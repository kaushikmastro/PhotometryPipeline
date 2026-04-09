import logging
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit

import pandas as pd
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataManager:
    """
    Manages the data ingestion, validation, and downloading process for the Vesta photometric survey.
    """

    def __init__(
        self,
        manifest_path: str,
        data_root: str,
        pds_base_url: str = "https://pds-imaging.jpl.nasa.gov/data/dawn/vesta/fc/DWNVFC2_3/DATA/",
    ):
        """
        Initializes the DataManager.

        Args:
            manifest_path (str): The path to the survey manifest CSV file.
            data_root (str): The root directory for data storage (e.g., '/scratch/kaushim07/vesta_data/').
            pds_base_url (str): The base URL for the PDS data archive.
        """
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)
        self.image_dir = self.data_root / "01_calibrated_images"
        self.spice_dir = self.data_root / "02_spice_kernels"
        self.dtm_dir = self.data_root / "03_dtm"
        self.dtm_geometry_label_url = (
            "https://sbnarchive.psi.edu/pds3/dawn/fc/DWNVSPG_2/GEOMETRY/dawn_vesta_SPG20160901.lbl"
        )
        self.dtm_geometry_pck_url = (
            "https://sbnarchive.psi.edu/pds3/dawn/fc/DWNVSPG_2/GEOMETRY/dawn_vesta_SPG20160901.tpc"
        )
        self.dtm_geometry_label_name = "dawn_vesta_SPG20160901.lbl"
        self.dtm_geometry_pck_name = "dawn_vesta_SPG20160901.tpc"
        self.pds_base_url = pds_base_url
        self.naif_base_url = "https://naif.jpl.nasa.gov/pub/naif/DAWN/kernels/"
        self.fc_base_url = "https://sbnarchive.psi.edu/pds3/dawn/fc/DWNVFC2_1B/DATA/IMG/"
        self.dtm_base_url = "https://sbnarchive.psi.edu/pds3/dawn/fc/DWNVSPG_2/DATA/"
        self.dtm_required_subpath = "/pds3/dawn/fc/DWNVSPG_2/DATA/"
        self.naif_pds_bundle_base = (
            "https://naif.jpl.nasa.gov/pub/naif/pds/data/dawn-m_a-spice-6-v1.0/dawnsp_1000/"
        )
        self._dir_cache: dict[str, tuple[list[str], list[str]]] = {}

        try:
            self.manifest = pd.read_csv(self.manifest_path)
            logging.info(f"Successfully loaded manifest from {self.manifest_path}")
        except FileNotFoundError:
            logging.error(f"Manifest file not found at {self.manifest_path}")
            self.manifest = pd.DataFrame()  # Empty dataframe

    def _normalize_image_stem(self, image_id: str) -> str:
        """Normalize manifest values to a stem usable for URL/file construction."""
        token = Path(str(image_id).strip()).name
        stem = Path(token).stem.upper()
        return stem

    def _manifest_image_column(self) -> str:
        """Resolve supported manifest column name for image IDs."""
        if "image_filename" in self.manifest.columns:
            return "image_filename"
        if "image_id" in self.manifest.columns:
            return "image_id"
        raise KeyError("Manifest must contain 'image_filename' or 'image_id' column.")

    def _construct_image_url(self, image_id: str) -> tuple[list[str], list[str]]:
        """
        Construct candidate IMG/LBL URLs for a Dawn FC2 image ID.

        The SBN/PDS3 folder layout is not always uniform across volumes, so this
        returns a prioritized candidate list. The downloader will try candidates
        in order until one succeeds.
        """
        base_url = self.fc_base_url
        stem = self._normalize_image_stem(image_id)

        # Candidate directory structure from most-specific to broad fallback.
        dirs = []
        if len(stem) >= 8:
            dirs.append(f"{stem[:5]}/{stem[:8]}/")
        if len(stem) >= 5:
            dirs.append(f"{stem[:5]}/")
        if len(stem) >= 3:
            dirs.append(f"{stem[:3]}/")
        dirs.append("")

        img_name = f"{stem}.IMG"
        lbl_name = f"{stem}.LBL"

        img_urls = [f"{base_url}{directory}{img_name}" for directory in dirs]
        lbl_urls = [f"{base_url}{directory}{lbl_name}" for directory in dirs]
        return img_urls, lbl_urls

    def _list_remote_directory_filtered(
        self, url: str, required_subpath: str | None = None
    ) -> tuple[list[str], list[str]]:
        """Return (file_hrefs, dir_hrefs) from a simple HTML directory listing."""
        if not url.endswith("/"):
            url = f"{url}/"

        cache_key = f"{url}|{required_subpath or '*'}"
        if cache_key in self._dir_cache:
            return self._dir_cache[cache_key]

        files: list[str] = []
        dirs: list[str] = []
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            hrefs = re.findall(r'href=["\']([^"\']+)["\']', response.text, flags=re.IGNORECASE)
            parsed_root = urlparse(url)
            for href in hrefs:
                if href in {"../", "./"}:
                    continue
                if href.startswith("?") or href.startswith("#"):
                    continue

                full = urljoin(url, href)
                parsed_full = urlparse(full)

                # Keep crawl constrained to same host and expected subtree.
                if parsed_full.netloc != parsed_root.netloc:
                    continue
                if required_subpath and required_subpath not in parsed_full.path:
                    continue

                if href.endswith("/"):
                    dirs.append(full)
                else:
                    files.append(full)
        except requests.exceptions.RequestException:
            # Quiet fallback: caller decides whether to continue exploring.
            pass

        self._dir_cache[cache_key] = (files, dirs)
        return files, dirs

    def _list_remote_directory(self, url: str) -> tuple[list[str], list[str]]:
        """FC image-specific directory listing helper."""
        return self._list_remote_directory_filtered(
            url,
            required_subpath="/pds3/dawn/fc/DWNVFC2_1B/DATA/IMG/",
        )

    def _find_and_download_dtm(self, max_depth: int = 6, max_dirs: int = 500) -> bool:
        """
        Crawl Dawn SPG DATA index and download HAMO DTM IMG files plus bundle geometry metadata.

        Returns:
            bool: True if DTM IMG files and mandatory geometry metadata are present.
        """
        try:
            resolved_data_root = self.data_root.resolve(strict=True)
            if not str(resolved_data_root).startswith("/scratch/"):
                logging.error(
                    "Refusing DTM download: data_root resolves to %s, expected /scratch/...",
                    resolved_data_root,
                )
                return False
        except FileNotFoundError:
            logging.error("Data root does not exist: %s", self.data_root)
            return False

        self.dtm_dir.mkdir(parents=True, exist_ok=True)
        existing_imgs = list(self.dtm_dir.glob("*.IMG"))
        geometry_ready = (self.dtm_dir / self.dtm_geometry_label_name).exists() and (
            self.dtm_dir / self.dtm_geometry_pck_name
        ).exists()

        # Idempotency: if authoritative geometry metadata exists, preserve IMG files.
        if existing_imgs and geometry_ready:
            logging.info("DTM foundation already present in %s", self.dtm_dir)
            return True

        # Legacy cleanup path for stale pre-fix states without authoritative metadata.
        if existing_imgs and not geometry_ready:
            orphaned_imgs = [img for img in existing_imgs if not img.with_suffix(".LBL").exists()]
            if orphaned_imgs:
                for orphan in orphaned_imgs:
                    logging.warning("Removing orphaned DTM IMG without matching LBL: %s", orphan)
                    orphan.unlink()

        existing_imgs = list(self.dtm_dir.glob("*.IMG"))
        if existing_imgs:
            if self._download_dtm_geometry_metadata():
                logging.info("DTM foundation ready in %s", self.dtm_dir)
                return True
            raise RuntimeError(
                "DTM geometry metadata is missing; cannot proceed with DTM foundation."
            )

        queue: list[tuple[str, int]] = [(self.dtm_base_url, 0)]
        visited = set()
        explored = 0
        img_candidates: list[str] = []

        while queue and explored < max_dirs:
            current_url, depth = queue.pop(0)
            if current_url in visited:
                continue
            visited.add(current_url)
            explored += 1

            files, dirs = self._list_remote_directory_filtered(
                current_url, required_subpath=self.dtm_required_subpath
            )
            for file_url in files:
                filename = Path(urlparse(file_url).path).name.upper()
                if "HAMO" in filename and "DTM" in filename:
                    if filename.endswith(".IMG"):
                        img_candidates.append(file_url)

            if depth >= max_depth:
                continue

            scored_dirs: list[tuple[int, str]] = []
            for d in dirs:
                dirname = Path(urlparse(d).path.rstrip("/")).name.upper()
                score = 0
                if "DTM" in dirname:
                    score += 3
                if "HAMO" in dirname:
                    score += 3
                if "SHAPE" in dirname:
                    score += 2
                if "DATA" in dirname:
                    score += 1
                scored_dirs.append((score, d))

            scored_dirs.sort(key=lambda item: item[0], reverse=True)
            for _, d in scored_dirs:
                if d not in visited:
                    queue.append((d, depth + 1))

        # De-duplicate while preserving order.
        img_candidates = list(dict.fromkeys(img_candidates))

        if not img_candidates:
            logging.critical(
                "DTM crawl failed to discover any HAMO/DTM IMG files under %s",
                self.dtm_base_url,
            )
            raise RuntimeError(
                f"DTM crawl failed to discover any HAMO/DTM IMG files under {self.dtm_base_url}"
            )

        img_by_stem: dict[str, list[str]] = {}
        for u in img_candidates:
            stem = Path(urlparse(u).path).stem.upper()
            img_by_stem.setdefault(stem, []).append(u)

        stems = sorted(img_by_stem.keys())

        for stem in stems:
            img_dest = self.dtm_dir / f"{stem}.IMG"

            if img_dest.exists():
                continue

            img_ok = self._download_from_candidate_urls(img_by_stem[stem], img_dest)
            if not img_ok:
                logging.warning("Failed to download DTM IMG for %s", stem)
                continue

        if not self._download_dtm_geometry_metadata():
            logging.critical("DTM geometry metadata download failed.")
            raise RuntimeError("DTM geometry metadata download failed.")

        logging.info("DTM foundation ready in %s", self.dtm_dir)
        return True

    def ensure_dtm_foundation(self) -> bool:
        """Public API to ensure DTM foundation exists for downstream geometry workflows."""
        return self._find_and_download_dtm()

    def _download_dtm_geometry_metadata(self) -> bool:
        """Download the authoritative DTM geometry label and PCK metadata files."""
        self.dtm_dir.mkdir(parents=True, exist_ok=True)

        label_dest = self.dtm_dir / self.dtm_geometry_label_name
        pck_dest = self.dtm_dir / self.dtm_geometry_pck_name

        label_ok = label_dest.exists() or self._download_file_with_retries(
            self.dtm_geometry_label_url, label_dest
        )
        if not label_ok:
            return False

        pck_ok = pck_dest.exists() or self._download_file_with_retries(
            self.dtm_geometry_pck_url, pck_dest
        )
        if not pck_ok:
            return False

        return True

    def _discover_image_urls(
        self, image_id: str, max_depth: int = 3, max_dirs: int = 200
    ) -> tuple[list[str], list[str]]:
        """
        Discover image URLs by crawling the archive listing when static paths fail.

        Keeps crawling bounded for HPC safety.
        """
        stem = self._normalize_image_stem(image_id)
        target_img = f"{stem}.IMG"
        target_lbl = f"{stem}.LBL"

        img_hits: list[str] = []
        lbl_hits: list[str] = []

        # Seed with base and likely prefix folders.
        seed_dirs = [self.fc_base_url]
        if len(stem) >= 5:
            seed_dirs.append(f"{self.fc_base_url}{stem[:5]}/")
        if len(stem) >= 3:
            seed_dirs.append(f"{self.fc_base_url}{stem[:3]}/")

        queue: list[tuple[str, int]] = [(d, 0) for d in list(dict.fromkeys(seed_dirs))]
        visited = set()
        explored = 0

        while queue and explored < max_dirs:
            current_url, depth = queue.pop(0)
            if current_url in visited:
                continue
            visited.add(current_url)
            explored += 1

            files, dirs = self._list_remote_directory(current_url)
            for file_url in files:
                name = Path(file_url).name.upper()
                if name == target_img and file_url not in img_hits:
                    img_hits.append(file_url)
                if name == target_lbl and file_url not in lbl_hits:
                    lbl_hits.append(file_url)

            if img_hits and lbl_hits:
                break

            if depth >= max_depth:
                continue

            # Prioritize subfolders that resemble the image prefix.
            scored_dirs = []
            for d in dirs:
                name = Path(d.rstrip("/")).name.upper()
                score = 0
                if len(stem) >= 5 and stem[:5] in name:
                    score += 3
                if len(stem) >= 3 and stem[:3] in name:
                    score += 2
                if "FC" in name:
                    score += 1
                scored_dirs.append((score, d))

            scored_dirs.sort(key=lambda item: item[0], reverse=True)
            for _, d in scored_dirs:
                if d not in visited:
                    queue.append((d, depth + 1))

        return img_hits, lbl_hits

    def _derive_lbl_urls_from_img_urls(self, img_urls: list[str]) -> list[str]:
        """Build likely LBL URLs from discovered IMG URLs in the same directories."""
        derived: list[str] = []
        for img_url in img_urls:
            parsed = urlsplit(img_url)
            path = Path(parsed.path)
            label_path = path.with_suffix(".LBL")
            derived.append(
                urlunsplit(
                    (parsed.scheme, parsed.netloc, str(label_path), parsed.query, parsed.fragment)
                )
            )
        return list(dict.fromkeys(derived))

    def _get_missing_images(self) -> list:
        """Identifies which image files from the manifest are missing."""
        if self.manifest.empty:
            return []

        col = self._manifest_image_column()
        missing_files = []
        for raw_value in self.manifest[col].astype(str):
            stem = self._normalize_image_stem(raw_value)
            if not (self.image_dir / f"{stem}.IMG").exists():
                missing_files.append(stem)
        return missing_files

    def download_missing_data(self):
        """
        Downloads missing image (.IMG and .LBL) files from the PDS.
        """
        try:
            resolved_data_root = self.data_root.resolve(strict=True)
            if not str(resolved_data_root).startswith("/scratch/"):
                logging.error(
                    "Refusing download: data_root resolves to %s, expected /scratch/...",
                    resolved_data_root,
                )
                return
        except FileNotFoundError:
            logging.error("Data root does not exist: %s", self.data_root)
            return

        missing_images = self._get_missing_images()
        if not missing_images:
            logging.info("No missing image files to download. Data is up to date.")
            return

        logging.info(f"Found {len(missing_images)} missing image files. Starting download...")
        self.image_dir.mkdir(parents=True, exist_ok=True)

        for image_id in missing_images:
            img_dest = self.image_dir / f"{image_id}.IMG"
            lbl_dest = self.image_dir / f"{image_id}.LBL"

            discovered_img_urls, discovered_lbl_urls = self._discover_image_urls(image_id)
            if discovered_img_urls and not discovered_lbl_urls:
                discovered_lbl_urls = self._derive_lbl_urls_from_img_urls(discovered_img_urls)
            logging.info(
                "Discovery for %s found %d IMG candidates and %d LBL candidates",
                image_id,
                len(discovered_img_urls),
                len(discovered_lbl_urls),
            )

            static_img_urls, static_lbl_urls = self._construct_image_url(image_id)
            img_urls = list(dict.fromkeys(discovered_img_urls + static_img_urls))
            lbl_urls = list(dict.fromkeys(discovered_lbl_urls + static_lbl_urls))

            img_ok = self._download_from_candidate_urls(img_urls, img_dest)
            lbl_ok = self._download_from_candidate_urls(lbl_urls, lbl_dest)

            if not img_ok:
                logging.error("Failed to download IMG for image_id=%s", image_id)
            if not lbl_ok:
                logging.warning(
                    "LBL not found for image_id=%s. Continuing with IMG-only ingestion.",
                    image_id,
                )

    def _download_from_candidate_urls(
        self, urls: list[str], destination: Path, max_retries: int = 3
    ) -> bool:
        """Try downloading from multiple candidate URLs until one succeeds."""
        for url in urls:
            if self._download_file_with_retries(url, destination, max_retries=max_retries):
                return True
        return False

    def _download_first_available(
        self, urls: list[str], destination_dir: Path, max_retries: int = 3
    ) -> Path | None:
        """Download the first reachable URL into destination_dir and return the saved path."""
        destination_dir.mkdir(parents=True, exist_ok=True)
        for url in urls:
            dest = destination_dir / Path(url).name
            if self._download_file_with_retries(url, dest, max_retries=max_retries):
                return dest
        return None

    def _discover_metakernel_urls(self) -> list[str]:
        """
        Discover candidate Vesta survey metakernel URLs from NAIF mk directory.

        Falls back to a small static candidate list if directory parsing fails.
        """
        mk_url = f"{self.naif_base_url}mk/"
        mk_fallback_url = f"{self.naif_pds_bundle_base}extras/mk/"
        candidates: list[str] = []

        for source_url in [mk_url, mk_fallback_url]:
            try:
                resp = requests.get(source_url, timeout=30)
                resp.raise_for_status()
                names = re.findall(r'href=["\']([^"\']+\.tm)["\']', resp.text, flags=re.IGNORECASE)
                names = [Path(name).name for name in names]

                preferred = [
                    n
                    for n in names
                    if "vesta" in n.lower()
                    and ("survey" in n.lower() or "phase3" in n.lower() or "v3" in n.lower())
                ]
                # Dawn annual metakernels are the real set in dawnsp_1000/extras/mk.
                annual = [n for n in names if re.match(r"(?i)^dawn_20\d\d_v\d\d\.tm$", n)]
                secondary = [n for n in names if "dawn" in n.lower()]
                ordered = preferred or annual or secondary or names

                candidates.extend([f"{source_url}{n}" for n in ordered])
            except requests.exceptions.RequestException as exc:
                logging.warning(
                    "Could not parse metakernel directory listing at %s: %s", source_url, exc
                )

        # Static fallback list for robustness in case listing format changes.
        static_fallback = [
            f"{mk_url}dawn_vesta_survey_v3.tm",
            f"{mk_url}dawn_vesta_phase3_survey.tm",
            f"{mk_url}dawn_vesta_survey.tm",
            f"{mk_url}dawn_vesta.tm",
            f"{mk_fallback_url}dawn_2011_v06.tm",
            f"{mk_fallback_url}dawn_2011_v05.tm",
            f"{mk_fallback_url}dawn_2012_v01.tm",
        ]

        # De-duplicate while preserving order.
        merged = candidates + static_fallback
        deduped = list(dict.fromkeys(merged))
        return deduped

    def _extract_kernel_paths_from_metakernel(self, metakernel_text: str) -> list[str]:
        """Extract referenced child-kernel paths from metakernel content."""
        ext_pattern = r"\.(?:bsp|bc|tf|ti|tsc)$"
        token_pattern = r"[A-Za-z0-9_\-./$]+\.(?:bsp|bc|tf|ti|tsc)"

        kernel_refs: list[str] = []
        for raw_line in metakernel_text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("\\"):
                continue

            for token in re.findall(token_pattern, line, flags=re.IGNORECASE):
                t = token.strip("'\"(),")
                if not re.search(ext_pattern, t, flags=re.IGNORECASE):
                    continue

                if t.startswith("$") and "/" in t:
                    t = t.split("/", 1)[1]
                t = t.lstrip("./")
                kernel_refs.append(t)

        return list(dict.fromkeys(kernel_refs))

    def _construct_kernel_url(self, kernel_ref: str) -> str:
        """Construct NAIF URL from a metakernel child reference."""
        ref = kernel_ref.strip()
        if ref.startswith("http://") or ref.startswith("https://"):
            return ref

        ref = ref.replace("\\", "/")
        while ref.startswith("../"):
            ref = ref[3:]
        ref = ref.lstrip("./")
        if ref.lower().startswith("data/"):
            ref = ref[5:]

        if "/" in ref:
            return f"{self.naif_base_url}{ref.lstrip('/')}"

        ext = Path(ref).suffix.lower()
        ext_dir = {
            ".bsp": "spk",
            ".bc": "ck",
            ".tf": "fk",
            ".ti": "ik",
            ".tsc": "sclk",
        }
        if ext not in ext_dir:
            raise ValueError(f"Unsupported kernel extension for reference: {ref}")

        return f"{self.naif_base_url}{ext_dir[ext]}/{ref}"

    def download_spice_kernels(self) -> bool:
        """
        Download SPICE kernels including metakernel, child kernels, and reconstructed SPK.

        This method:
        1. Downloads the survey metakernel and its referenced child kernels
        2. Downloads reconstructed spacecraft SPK kernels for mission coverage
        3. Generates a dynamic metakernel that includes all available kernels

        Returns:
            bool: True if kernels are ready (either downloaded or previously cached).
        """
        try:
            resolved_data_root = self.data_root.resolve(strict=True)
            if not str(resolved_data_root).startswith("/scratch/"):
                logging.error(
                    "Refusing SPICE download: data_root resolves to %s, expected /scratch/...",
                    resolved_data_root,
                )
                return False
        except FileNotFoundError:
            logging.error("Data root does not exist: %s", self.data_root)
            return False

        self.spice_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Check if we already have full-enough kernels from previous runs.
        existing_spk = list(self.spice_dir.glob("*.bsp"))
        existing_ck = list(self.spice_dir.glob("*.bc"))
        dynamic_mk = self.spice_dir / "dawn_dynamic.tm"
        has_2011_sc_ck = any(
            re.search(r"(?i)^dawn_sc_11\d{4}_\d{6}.*\.bc$", p.name) for p in existing_ck
        )
        has_fc2_ck = any(re.search(r"(?i)^dawn_fc2_.*\.bc$", p.name) for p in existing_ck)
        if existing_spk and existing_ck and dynamic_mk.exists() and (has_2011_sc_ck or has_fc2_ck):
            logging.info(
                "SPICE kernels already present with CK coverage markers; refreshing SCLK/CK and metakernel."
            )
            self._ensure_latest_sclk_kernel()
            self._download_reconstructed_ck_kernels()
            self._generate_dynamic_metakernel()
            return True

        # Step 2: Download metakernel (.tm).
        metakernel_urls = self._discover_metakernel_urls()
        metakernel_path = self._download_first_available(metakernel_urls, self.spice_dir)
        if metakernel_path is None:
            logging.warning(
                "Unable to download survey metakernel from NAIF, will proceed with reconstructed SPK only"
            )
        else:
            logging.info("Downloaded metakernel: %s", metakernel_path)

            # Step 3: Parse metakernel for required child kernels.
            try:
                mk_text = metakernel_path.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:
                logging.error("Failed reading metakernel %s: %s", metakernel_path, exc)
                return False

            kernel_refs = self._extract_kernel_paths_from_metakernel(mk_text)
            if kernel_refs:
                # Step 4: Construct URLs and download only required kernels.
                for kernel_ref in kernel_refs:
                    try:
                        kernel_url = self._construct_kernel_url(kernel_ref)
                    except ValueError as exc:
                        logging.warning("Skipping kernel reference %s: %s", kernel_ref, exc)
                        continue

                    destination = self.spice_dir / Path(kernel_ref).name
                    if not self._download_file_with_retries(kernel_url, destination, max_retries=3):
                        logging.warning("Failed to download kernel %s (non-fatal)", kernel_ref)

        # Step 5: Download reconstructed spacecraft SPK kernels for mission coverage
        logging.info("Attempting to download reconstructed spacecraft SPK kernels...")
        self._download_reconstructed_spk_kernels()

        # Step 6: Download reconstructed spacecraft/instrument CK kernels for pointing coverage.
        logging.info("Attempting to download reconstructed CK kernels for attitude coverage...")
        self._download_reconstructed_ck_kernels()

        # Step 7: Ensure latest SCLK (.tsc) is available for late-2011 attitude unlock.
        logging.info("Ensuring latest DAWN SCLK kernel is available...")
        self._ensure_latest_sclk_kernel()

        # Step 8: Ensure de421.bsp (planetary positions) is available
        planetary_bsp = self.spice_dir / "de421.bsp"
        if not planetary_bsp.exists():
            try:
                de421_url = (
                    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de421.bsp"
                )
                self._download_file_with_retries(de421_url, planetary_bsp, max_retries=2)
            except Exception as exc:
                logging.warning(f"Could not ensure de421.bsp: {exc}")

        # Step 9: Generate dynamic metakernel that includes all available kernels
        self._generate_dynamic_metakernel()

        logging.info("SPICE kernel setup complete")
        return True

    def _discover_reconstructed_spk_urls(self) -> list[str]:
        """
        Discover reconstructed spacecraft SPK kernels from NAIF archive for mission phases.

        Returns:
            List of candidate SPK URLs, prioritized by relevance.
        """
        candidates: list[str] = []

        # Check NAIF spk directory for reconstructed spacecraft position kernels
        spk_base = f"{self.naif_base_url}spk/"

        try:
            resp = requests.get(spk_base, timeout=30)
            resp.raise_for_status()

            # Find files matching dawn*rec*.bsp or dawn_sc*.bsp patterns
            rec_files = re.findall(
                r'href=["\']([^"\']*(?:dawn[^"\']*rec[^"\']*|dawn_sc[\w_]*?)\.bsp)["\']',
                resp.text,
                flags=re.IGNORECASE,
            )
            rec_files = list(dict.fromkeys([Path(f).name for f in rec_files]))  # De-dup

            for f in rec_files:
                candidates.append(f"{spk_base}{f}")

            logging.info(f"Discovered {len(candidates)} reconstructed SPK kernel candidates")
        except requests.exceptions.RequestException as exc:
            logging.warning(f"Could not scan NAIF spk directory: {exc}")

        # Static fallback list for common reconstructed SPK files covering various mission phases
        static_candidates = [
            f"{spk_base}dawn_rec_070927_120201.bsp",
            f"{spk_base}dawn_070927_071031_rec.bsp",
            f"{spk_base}dawn_071101_120201_rec.bsp",
            f"{spk_base}dawn_sc_071001_080106.bsp",
            f"{spk_base}dawn_rec_2011_v1.bsp",
            f"{spk_base}dawn_rec_combined.bsp",
        ]

        candidates.extend(static_candidates)
        return list(dict.fromkeys(candidates))  # De-duplicate while preserving order

    def _download_reconstructed_spk_kernels(self) -> int:
        """
        Download reconstructed spacecraft SPK kernels for the full mission timeline.

        Returns:
            int: Number of successfully downloaded SPK kernels.
        """
        spk_urls = self._discover_reconstructed_spk_urls()
        downloaded_count = 0

        for url in spk_urls:
            destination = self.spice_dir / Path(url).name

            if destination.exists():
                logging.info(f"Reconstructed SPK kernel already present: {destination.name}")
                downloaded_count += 1
                continue

            if self._download_file_with_retries(url, destination, max_retries=2):
                logging.info(f"Successfully downloaded reconstructed SPK: {destination.name}")
                downloaded_count += 1

        if downloaded_count > 0:
            logging.info(f"Downloaded {downloaded_count} reconstructed SPK kernel(s)")
        else:
            logging.warning("No reconstructed SPK kernels were successfully downloaded")

        return downloaded_count

    def _discover_reconstructed_ck_urls(self) -> list[str]:
        """Discover CK kernels needed for 2011 Vesta FC2 pointing coverage."""
        ck_base = f"{self.naif_base_url}ck/"
        candidates: list[str] = []

        try:
            resp = requests.get(ck_base, timeout=30)
            resp.raise_for_status()
            names = sorted(
                set(re.findall(r'href=["\']([^"\']+\.bc)["\']', resp.text, flags=re.IGNORECASE))
            )

            # Mission-week attitude kernels around Vesta encounter (2011) for spacecraft and solar arrays.
            weekly_2011 = [
                n for n in names if re.search(r"(?i)^dawn_(?:sc|sa)_11\d{4}_\d{6}.*\.bc$", n)
            ]
            # FC2 CK is mission-interval instrument pointing and should be included when available.
            fc2 = [n for n in names if re.search(r"(?i)^dawn_fc2_.*\.bc$", n)]
            # Include quick-look 2011 segments as fallback/augment.
            ql_2011 = [n for n in names if re.search(r"(?i)^dawn_ql_11\d{4}_\d{6}\.bc$", n)]

            for n in weekly_2011 + fc2 + ql_2011:
                candidates.append(f"{ck_base}{Path(n).name}")

            logging.info(
                "Discovered %d CK kernel candidates for 2011/FC2 coverage", len(candidates)
            )
        except requests.exceptions.RequestException as exc:
            logging.warning("Could not scan NAIF ck directory: %s", exc)

        # Conservative static fallback list (core weekly and FC2 coverage file).
        static_fallback = [
            f"{ck_base}dawn_sc_110801_110807.bc",
            f"{ck_base}dawn_sc_110808_110814.bc",
            f"{ck_base}dawn_sc_110815_110821.bc",
            f"{ck_base}dawn_sc_110822_110828.bc",
            f"{ck_base}dawn_sa_110801_110807.bc",
            f"{ck_base}dawn_sa_110808_110814.bc",
            f"{ck_base}dawn_sa_110815_110821.bc",
            f"{ck_base}dawn_sa_110822_110828.bc",
            f"{ck_base}dawn_fc2_110723_120725_grv221108_v1.bc",
        ]
        candidates.extend(static_fallback)
        return list(dict.fromkeys(candidates))

    def _discover_latest_sclk_url(self) -> str | None:
        """Discover the latest DAWN SCLK kernel URL from NAIF sclk directory."""
        sclk_base = f"{self.naif_base_url}sclk/"
        try:
            resp = requests.get(sclk_base, timeout=30)
            resp.raise_for_status()
            names = re.findall(r'href=["\']([^"\']+\.tsc)["\']', resp.text, flags=re.IGNORECASE)
            names = [Path(n).name for n in names]
            dawn_names = [n for n in names if re.match(r"(?i)^DAWN_203_SCLKSCET\.\d+\.tsc$", n)]
            if not dawn_names:
                return None

            def _version_num(name: str) -> int:
                m = re.search(r"\.(\d+)\.tsc$", name, flags=re.IGNORECASE)
                return int(m.group(1)) if m else -1

            latest = max(dawn_names, key=_version_num)
            return f"{sclk_base}{latest}"
        except requests.exceptions.RequestException as exc:
            logging.warning("Could not discover latest SCLK from NAIF: %s", exc)
            return None

    def _ensure_latest_sclk_kernel(self) -> Path | None:
        """Download the latest DAWN SCLK if available; return the local path."""
        latest_url = self._discover_latest_sclk_url()
        fallback_name = "DAWN_203_SCLKSCET.00091.tsc"

        if latest_url is None:
            latest_url = f"{self.naif_base_url}sclk/{fallback_name}"

        dest = self.spice_dir / Path(latest_url).name
        if not dest.exists():
            if not self._download_file_with_retries(latest_url, dest, max_retries=3):
                logging.warning("Could not download latest SCLK from %s", latest_url)
                return None

        logging.info("Latest SCLK available: %s", dest.name)
        return dest

    def _download_reconstructed_ck_kernels(self) -> int:
        """Download reconstructed CK kernels needed for FC2 frame connectivity at 2011 epochs."""
        ck_urls = self._discover_reconstructed_ck_urls()
        downloaded_count = 0

        for url in ck_urls:
            destination = self.spice_dir / Path(url).name

            if destination.exists():
                downloaded_count += 1
                continue

            if self._download_file_with_retries(url, destination, max_retries=2):
                logging.info("Successfully downloaded CK: %s", destination.name)
                downloaded_count += 1

        if downloaded_count > 0:
            logging.info("Downloaded/present CK kernels: %d", downloaded_count)
        else:
            logging.warning("No reconstructed CK kernels were successfully downloaded")

        return downloaded_count

    def _generate_dynamic_metakernel(self) -> Path:
        """
        Generate a dynamic metakernel (.tm file) that includes all available kernels.

        This creates a metakernel text file that explicitly lists all .bsp, .bc, .tf, .ti
        and .tsc files currently in the spice_dir, ensuring proper SPICE kernel loading
        without path ambiguities.

        Returns:
            Path: The generated metakernel file path.
        """
        metakernel_path = self.spice_dir / "dawn_dynamic.tm"

        # Collect all kernel files by type
        kernel_patterns = {
            ".tls": "LSK",
            ".tsc": "SCLK",
            ".tpc": "PCK",
            ".tf": "FRAMES",
            ".ti": "INSTRUMENT",
            ".bsp": "SPK",
            ".bc": "CK",
        }

        kernel_files: dict[str, list[Path]] = {ext: [] for ext in kernel_patterns}

        for ext, _ in kernel_patterns.items():
            found = sorted(self.spice_dir.glob(f"*{ext}"))
            # Also include DTM-specific geometry files
            if ext == ".tpc":
                found.extend(sorted(self.dtm_dir.glob(f"*{ext}")))
                found = sorted(set(found))  # De-duplicate
            kernel_files[ext] = found

        spice_root = str(self.spice_dir.resolve())
        dtm_root = str(self.dtm_dir.resolve())

        # Generate metakernel content with proper KPL format.
        lines = [
            "KPL/MK",
            "",
            "\\begindata",
            "",
            "PATH_VALUES = (",
            f"  '{spice_root}',",
            f"  '{dtm_root}'",
            ")",
            "",
            "PATH_SYMBOLS = (",
            "  'SPICE',",
            "  'DTM'",
            ")",
            "",
            "KERNELS_TO_LOAD = (",
        ]

        kernel_list = []

        # Load in SPICE-recommended order
        order = [".tls", ".tsc", ".tpc", ".tf", ".ti", ".bsp", ".bc"]
        for ext in order:
            for kernel_path in kernel_files.get(ext, []):
                symbol = (
                    "DTM" if kernel_path.resolve().parent == self.dtm_dir.resolve() else "SPICE"
                )
                kernel_list.append(f"  '${symbol}/{kernel_path.name}'")

        if kernel_list:
            lines.extend([",\n".join(kernel_list)])

        lines.extend(
            [
                ")",
                "\\begintext",
                "Dynamic metakernel generated for Vesta geometry computation.",
                "Includes all available kernels in spice_dir and DTM geometry files.",
                "\\endtext",
            ]
        )

        metakernel_content = "\n".join(lines)
        metakernel_path.write_text(metakernel_content, encoding="utf-8")

        logging.info(f"Generated dynamic metakernel: {metakernel_path.name}")
        logging.debug(f"Metakernel includes {len(kernel_list)} kernels")

        return metakernel_path

    def _download_file_with_retries(
        self, url: str, destination: Path, max_retries: int = 3
    ) -> bool:
        """
        Downloads a file from a URL with a retry mechanism.

        Args:
            url (str): The URL to download from.
            destination (Path): The local path to save the file.
            max_retries (int): The maximum number of download attempts.
        """
        for attempt in range(max_retries):
            try:
                logging.info(
                    f"Downloading {url} to {destination} (Attempt {attempt + 1}/{max_retries})"
                )
                with requests.get(url, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with open(destination, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                logging.info(f"Successfully downloaded {destination.name}")
                return True
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else None
                logging.warning(f"Failed to download {url} on attempt {attempt + 1}: {e}")
                if status_code == 404:
                    logging.error(f"Stopping retries for {url}: received 404 Not Found.")
                    return False
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    logging.error(f"All {max_retries} attempts to download {url} failed.")
            except requests.exceptions.RequestException as e:
                logging.warning(f"Failed to download {url} on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)  # Wait for 5 seconds before retrying
                else:
                    logging.error(f"All {max_retries} attempts to download {url} failed.")
        return False

    def validate_data_ready(self) -> bool:
        """
        Checks if all required data files (images and SPICE kernels) are present.

        Returns:
            bool: True if all files are ready, False otherwise.
        """
        if self.manifest.empty:
            logging.error("Manifest is empty. Cannot validate data.")
            return False

        all_ready = True
        missing_images = self._get_missing_images()

        # 1. Validate calibrated images
        if missing_images:
            all_ready = False
            for f in missing_images:
                logging.warning(f"Missing image file: {self.image_dir / (f + '.IMG')}")
        else:
            logging.info("All calibrated image files are present.")

        # 2. Validate SPICE metakernel
        logging.info("Validating SPICE metakernel...")
        # Assuming there's only one metakernel needed for the project
        try:
            metakernel = next(self.spice_dir.glob("*.tm"))
            logging.info(f"Found SPICE metakernel: {metakernel}")
        except StopIteration:
            logging.error(f"No SPICE metakernel (.tm file) found in {self.spice_dir}")
            all_ready = False

        return all_ready


if __name__ == "__main__":
    # Example usage:
    # This assumes you are running this script from the project root directory.
    # In the HPC environment, you would pass the absolute paths.

    # Create dummy files for testing
    Path("configs").mkdir(exist_ok=True)
    with open("configs/survey_manifest.csv", "w") as f:
        f.write("image_filename\n")
        f.write("FC21B0000001_00320_1F2A.IMG\n")  # A real file for testing download
        f.write("image2.IMG\n")

    data_dir = Path("data")
    (data_dir / "01_calibrated_images").mkdir(parents=True, exist_ok=True)
    (data_dir / "02_spice_kernels").mkdir(parents=True, exist_ok=True)
    (data_dir / "02_spice_kernels" / "vesta_v01.tm").touch()

    manager = DataManager(manifest_path="configs/survey_manifest.csv", data_root=str(data_dir))
    if not manager.validate_data_ready():
        print("Data is missing. Attempting to download...")
        manager.download_missing_data()
    else:
        print("All data is ready for the pipeline.")

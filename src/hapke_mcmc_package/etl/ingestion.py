import pandas as pd
import logging
from pathlib import Path
import requests
import time
from typing import Dict, List, Tuple
import re
from urllib.parse import urljoin, urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataManager:
    """
    Manages the data ingestion, validation, and downloading process for the Vesta photometric survey.
    """

    def __init__(self, manifest_path: str, data_root: str, pds_base_url: str = "https://pds-imaging.jpl.nasa.gov/data/dawn/vesta/fc/DWNVFC2_3/DATA/"):
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
        self.pds_base_url = pds_base_url
        self.naif_base_url = "https://naif.jpl.nasa.gov/pub/naif/DAWN/kernels/"
        self.fc_base_url = "https://sbnarchive.psi.edu/pds3/dawn/fc/DWNVFC2_1B/DATA/IMG/"
        self.naif_pds_bundle_base = "https://naif.jpl.nasa.gov/pub/naif/pds/data/dawn-m_a-spice-6-v1.0/dawnsp_1000/"
        self._dir_cache: Dict[str, Tuple[List[str], List[str]]] = {}
        
        try:
            self.manifest = pd.read_csv(self.manifest_path)
            logging.info(f"Successfully loaded manifest from {self.manifest_path}")
        except FileNotFoundError:
            logging.error(f"Manifest file not found at {self.manifest_path}")
            self.manifest = pd.DataFrame() # Empty dataframe

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

    def _construct_image_url(self, image_id: str) -> Tuple[List[str], List[str]]:
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

    def _list_remote_directory(self, url: str) -> Tuple[List[str], List[str]]:
        """Return (file_hrefs, dir_hrefs) from a simple HTML directory listing."""
        if not url.endswith("/"):
            url = f"{url}/"

        if url in self._dir_cache:
            return self._dir_cache[url]

        files: List[str] = []
        dirs: List[str] = []
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

                # Keep crawl constrained to same host and under expected FC data tree.
                if parsed_full.netloc != parsed_root.netloc:
                    continue
                if "/pds3/dawn/fc/DWNVFC2_1B/DATA/IMG/" not in parsed_full.path:
                    continue

                if href.endswith("/"):
                    dirs.append(full)
                else:
                    files.append(full)
        except requests.exceptions.RequestException:
            # Quiet fallback: caller decides whether to continue exploring.
            pass

        self._dir_cache[url] = (files, dirs)
        return files, dirs

    def _discover_image_urls(self, image_id: str, max_depth: int = 3, max_dirs: int = 200) -> Tuple[List[str], List[str]]:
        """
        Discover image URLs by crawling the archive listing when static paths fail.

        Keeps crawling bounded for HPC safety.
        """
        stem = self._normalize_image_stem(image_id)
        target_img = f"{stem}.IMG"
        target_lbl = f"{stem}.LBL"

        img_hits: List[str] = []
        lbl_hits: List[str] = []

        # Seed with base and likely prefix folders.
        seed_dirs = [self.fc_base_url]
        if len(stem) >= 5:
            seed_dirs.append(f"{self.fc_base_url}{stem[:5]}/")
        if len(stem) >= 3:
            seed_dirs.append(f"{self.fc_base_url}{stem[:3]}/")

        queue: List[Tuple[str, int]] = [(d, 0) for d in list(dict.fromkeys(seed_dirs))]
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

    def _derive_lbl_urls_from_img_urls(self, img_urls: List[str]) -> List[str]:
        """Build likely LBL URLs from discovered IMG URLs in the same directories."""
        derived: List[str] = []
        for img_url in img_urls:
            p = Path(img_url)
            # Most PDS3 FC products use uppercase .LBL; include lowercase fallback.
            derived.append(str(p.with_suffix(".LBL")))
            derived.append(str(p.with_suffix(".lbl")))
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

    def _download_from_candidate_urls(self, urls: List[str], destination: Path, max_retries: int = 3) -> bool:
        """Try downloading from multiple candidate URLs until one succeeds."""
        for url in urls:
            if self._download_file_with_retries(url, destination, max_retries=max_retries):
                return True
        return False

    def _download_first_available(self, urls: List[str], destination_dir: Path, max_retries: int = 3) -> Path | None:
        """Download the first reachable URL into destination_dir and return the saved path."""
        destination_dir.mkdir(parents=True, exist_ok=True)
        for url in urls:
            dest = destination_dir / Path(url).name
            if self._download_file_with_retries(url, dest, max_retries=max_retries):
                return dest
        return None

    def _discover_metakernel_urls(self) -> List[str]:
        """
        Discover candidate Vesta survey metakernel URLs from NAIF mk directory.

        Falls back to a small static candidate list if directory parsing fails.
        """
        mk_url = f"{self.naif_base_url}mk/"
        mk_fallback_url = f"{self.naif_pds_bundle_base}extras/mk/"
        candidates: List[str] = []

        for source_url in [mk_url, mk_fallback_url]:
            try:
                resp = requests.get(source_url, timeout=30)
                resp.raise_for_status()
                names = re.findall(r'href=["\']([^"\']+\.tm)["\']', resp.text, flags=re.IGNORECASE)
                names = [Path(name).name for name in names]

                preferred = [
                    n for n in names
                    if "vesta" in n.lower() and (
                        "survey" in n.lower() or "phase3" in n.lower() or "v3" in n.lower()
                    )
                ]
                # Dawn annual metakernels are the real set in dawnsp_1000/extras/mk.
                annual = [n for n in names if re.match(r"(?i)^dawn_20\d\d_v\d\d\.tm$", n)]
                secondary = [n for n in names if "dawn" in n.lower()]
                ordered = preferred or annual or secondary or names

                candidates.extend([f"{source_url}{n}" for n in ordered])
            except requests.exceptions.RequestException as exc:
                logging.warning("Could not parse metakernel directory listing at %s: %s", source_url, exc)

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

    def _extract_kernel_paths_from_metakernel(self, metakernel_text: str) -> List[str]:
        """Extract referenced child-kernel paths from metakernel content."""
        ext_pattern = r"\.(?:bsp|bc|tf|ti|tsc)$"
        token_pattern = r"[A-Za-z0-9_\-./$]+\.(?:bsp|bc|tf|ti|tsc)"

        kernel_refs: List[str] = []
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
        Download the Vesta survey metakernel and only its required child kernels.

        Returns:
            bool: True if metakernel and all parsed child kernels are downloaded, else False.
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

        # Step 1: Download metakernel (.tm).
        metakernel_urls = self._discover_metakernel_urls()
        metakernel_path = self._download_first_available(metakernel_urls, self.spice_dir)
        if metakernel_path is None:
            logging.error("Unable to download any Vesta survey metakernel from NAIF.")
            return False

        logging.info("Downloaded metakernel: %s", metakernel_path)

        # Step 2: Parse metakernel for required child kernels.
        try:
            mk_text = metakernel_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            logging.error("Failed reading metakernel %s: %s", metakernel_path, exc)
            return False

        kernel_refs = self._extract_kernel_paths_from_metakernel(mk_text)
        if not kernel_refs:
            logging.warning("Metakernel parsed but no child kernel references were found.")
            return True

        # Step 3: Construct URLs and download only required kernels.
        all_ok = True
        for kernel_ref in kernel_refs:
            try:
                kernel_url = self._construct_kernel_url(kernel_ref)
            except ValueError as exc:
                logging.warning("Skipping kernel reference %s: %s", kernel_ref, exc)
                all_ok = False
                continue

            destination = self.spice_dir / Path(kernel_ref).name
            if not self._download_file_with_retries(kernel_url, destination, max_retries=3):
                logging.error("Failed to download required kernel %s", kernel_ref)
                all_ok = False

        return all_ok

    def _download_file_with_retries(self, url: str, destination: Path, max_retries: int = 3) -> bool:
        """
        Downloads a file from a URL with a retry mechanism.

        Args:
            url (str): The URL to download from.
            destination (Path): The local path to save the file.
            max_retries (int): The maximum number of download attempts.
        """
        for attempt in range(max_retries):
            try:
                logging.info(f"Downloading {url} to {destination} (Attempt {attempt + 1}/{max_retries})")
                with requests.get(url, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    with open(destination, 'wb') as f:
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

if __name__ == '__main__':
    # Example usage:
    # This assumes you are running this script from the project root directory.
    # In the HPC environment, you would pass the absolute paths.
    
    # Create dummy files for testing
    Path("configs").mkdir(exist_ok=True)
    with open("configs/survey_manifest.csv", "w") as f:
        f.write("image_filename\n")
        f.write("FC21B0000001_00320_1F2A.IMG\n") # A real file for testing download
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

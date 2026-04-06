from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
import spiceypy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _load_planetaryimage_module():
    """Load planetaryimage with a NumPy 2 compatibility shim when required."""
    if not hasattr(np, "product"):
        np.product = np.prod  # type: ignore[attr-defined]

    try:
        import planetaryimage

        return planetaryimage
    except ValueError as exc:
        if "fromstring" not in str(exc):
            raise

        old_fromstring = np.fromstring

        def _fromstring_compat(string, dtype=float, count=-1, sep=''):
            if sep == '' and isinstance(string, (bytes, bytearray, memoryview)):
                return np.frombuffer(string, dtype=dtype, count=count)
            return old_fromstring(string, dtype=dtype, count=count, sep=sep)

        np.fromstring = _fromstring_compat  # type: ignore[assignment]
        import planetaryimage

        logging.warning("Applied NumPy compatibility shim for planetaryimage import.")
        return planetaryimage


@dataclass
class DTMTile:
    """In-memory representation of one DTM tile and its precomputed gradients."""

    file_path: Path
    elevation: np.ndarray
    lat_min_deg: float
    lat_max_deg: float
    lon_min_deg: float
    lon_max_deg: float


def _as_str(value: Any) -> str:
    """Convert SPICE return values that may be bytes to plain strings."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value)


def _find_label_value(label: Any, key: str) -> Any:
    """Recursively search for a key in a nested PDS label structure."""
    key_upper = key.upper()

    def _search(obj: Any) -> Any:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).upper() == key_upper:
                    return v
                nested = _search(v)
                if nested is not None:
                    return nested
        return None

    return _search(label)


def _safe_float(label: Any, keys: Iterable[str], default: float) -> float:
    """Read a float value from label keys with fallback."""
    for key in keys:
        raw = _find_label_value(label, key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return default


def _to_spice_utc_string(value: Any) -> str:
    """Normalize label time values to a SPICE-friendly UTC string."""
    text = str(value).strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        return text.replace(" ", "T")

class GeometryEngine:
    """
    Core SPICE-based ray-tracing engine to calculate photometric angles.
    """

    def __init__(self, data_root: str, metakernel_path: str | None = None):
        """
        Initialize SPICE and load Vesta DTM products.

        Args:
            data_root (str): Root data directory (scratch-backed), containing
                02_spice_kernels, 03_dtm, and 04_geometry_tables subdirectories.
            metakernel_path (str | None): Optional path to explicit .tm file.
                If not provided, looks for dynamic metakernel first, then any .tm file.
        """
        self.data_root = Path(data_root)
        self.spice_dir = self.data_root / "02_spice_kernels"
        self.dtm_dir = self.data_root / "03_dtm"
        self.output_dir = self.data_root / "04_geometry_tables"
        self._planetaryimage = _load_planetaryimage_module()

        if metakernel_path is None:
            # Prefer dynamic metakernel if it exists
            dynamic_mk = self.spice_dir / "dawn_dynamic.tm"
            if dynamic_mk.exists():
                self.metakernel_path = dynamic_mk
            else:
                tm_files = sorted(self.spice_dir.glob("*.tm"))
                if not tm_files:
                    raise FileNotFoundError(f"No metakernel found in {self.spice_dir}")
                self.metakernel_path = tm_files[0]
        else:
            self.metakernel_path = Path(metakernel_path)
            if not self.metakernel_path.exists():
                raise FileNotFoundError(f"Metakernel not found: {self.metakernel_path}")

        self._initialize_spice()

        # Camera constants for Dawn's Framing Camera (FC2)
        self.instrument = 'DAWN_FC2'
        self.target = 'VESTA'
        self.aberration_correction = 'LT+S'
        self.target_frame = 'IAU_VESTA'
        self.observer = 'DAWN'
        
        # Get camera FOV details
        try:
            self.cam_id = spiceypy.bodn2c(self.instrument)
            _, self.cam_frame, self.boresight, self.num_bounds, self.bounds = spiceypy.getfov(self.cam_id, 4)
            self.cam_frame = _as_str(self.cam_frame)
            logging.info(f"Successfully loaded camera model for {self.instrument}")
        except Exception as e:
            logging.error(f"Could not get camera FOV for {self.instrument}. Check SPICE kernels. Error: {e}")
            raise

        self.vesta_radius_m = float(np.mean(spiceypy.bodvrd(self.target, "RADII", 3)[1])) * 1000.0
        self.dtm_tiles = self._load_dtm_tiles()

    def _initialize_spice(self) -> None:
        """Initialize SPICE with metakernel and robust local fallback for broken paths."""
        try:
            spiceypy.furnsh(str(self.metakernel_path))
            logging.info("Successfully loaded SPICE metakernel: %s", self.metakernel_path)
            return
        except spiceypy.support_types.SpiceyError as exc:
            logging.warning("Metakernel load failed, switching to fallback kernel load: %s", exc)
            spiceypy.kclear()

        self._ensure_minimum_spice_kernels()
        loaded_count = self._load_spice_fallback()
        if loaded_count == 0:
            raise RuntimeError("Failed to load any SPICE kernels via fallback path.")

        # Quick sanity check that time conversion is available (LSK loaded).
        try:
            spiceypy.utc2et("2000-01-01T12:00:00")
        except spiceypy.support_types.SpiceyError as exc:
            raise RuntimeError(f"SPICE fallback loaded kernels but time conversion failed: {exc}") from exc

        logging.info("SPICE fallback loader initialized %d kernels.", loaded_count)

    def _download_if_missing(self, url: str, destination: Path) -> None:
        """Download a kernel if it is missing locally."""
        if destination.exists():
            return

        destination.parent.mkdir(parents=True, exist_ok=True)
        logging.info("Downloading missing SPICE dependency %s", destination.name)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        destination.write_bytes(response.content)

    def _ensure_minimum_spice_kernels(self) -> None:
        """Ensure baseline kernels needed by SPICE time/frame operations are present."""
        required = [
            (
                "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
                self.spice_dir / "naif0012.tls",
            ),
            (
                "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc",
                self.spice_dir / "pck00010.tpc",
            ),
            (
                "https://naif.jpl.nasa.gov/pub/naif/DAWN/kernels/sclk/DAWN_203_SCLKSCET.00091.tsc",
                self.spice_dir / "DAWN_203_SCLKSCET.00091.tsc",
            ),
        ]

        for url, destination in required:
            try:
                self._download_if_missing(url, destination)
            except requests.exceptions.RequestException as exc:
                logging.warning("Could not download optional fallback kernel %s: %s", destination.name, exc)

    def _load_spice_fallback(self) -> int:
        """Load all available kernels from local directories in dependency-friendly order."""
        ordered_patterns = ["*.tls", "*.tsc", "*.tpc", "*.tf", "*.ti", "*.bsp", "*.bc"]
        load_paths: list[Path] = []
        for pattern in ordered_patterns:
            load_paths.extend(sorted(self.spice_dir.glob(pattern)))

        # Include DTM-specific planetary constants if available.
        load_paths.extend(sorted(self.dtm_dir.glob("*.tpc")))

        # De-duplicate while preserving order.
        seen = set()
        unique_paths: list[Path] = []
        for p in load_paths:
            if p in seen:
                continue
            seen.add(p)
            unique_paths.append(p)

        loaded = 0
        for kernel_path in unique_paths:
            try:
                spiceypy.furnsh(str(kernel_path))
                loaded += 1
            except spiceypy.support_types.SpiceyError as exc:
                logging.warning("Skipping kernel %s due to SPICE error: %s", kernel_path.name, exc)

        return loaded

    def _load_dtm_tiles(self) -> list[DTMTile]:
        """Load all available DTM IMG tiles and precompute gradients for normals."""
        dtm_paths = sorted(self.dtm_dir.glob("*_DTM.IMG"))
        if not dtm_paths:
            raise FileNotFoundError(f"No DTM IMG files found in {self.dtm_dir}")

        tiles: list[DTMTile] = []
        for dtm_path in dtm_paths:
            pds = self._planetaryimage.PDS3Image.open(str(dtm_path))
            elevation = pds.image.astype(np.float32)
            label = pds.label

            invalid_mask = ~np.isfinite(elevation) | (elevation < -1e30)
            elevation[invalid_mask] = np.nan

            lat_min = _safe_float(label, ["MINIMUM_LATITUDE"], -90.0)
            lat_max = _safe_float(label, ["MAXIMUM_LATITUDE"], 90.0)
            lon_min = _safe_float(label, ["WESTERNMOST_LONGITUDE", "MINIMUM_LONGITUDE"], 0.0)
            lon_max = _safe_float(label, ["EASTERNMOST_LONGITUDE", "MAXIMUM_LONGITUDE"], 360.0)

            tile = DTMTile(
                file_path=dtm_path,
                elevation=elevation,
                lat_min_deg=lat_min,
                lat_max_deg=lat_max,
                lon_min_deg=lon_min,
                lon_max_deg=lon_max,
            )
            tiles.append(tile)
            logging.info("Loaded DTM tile %s (%s)", dtm_path.name, elevation.shape)

        return tiles

    def _extract_observation_et(self, image_label: Any) -> float:
        """Extract observation UTC from image label and convert to ET."""
        for key in [
            "START_TIME",
            "STOP_TIME",
            "IMAGE_TIME",
            "DAWN:ALT_START_TIME",
            "DAWN:ALT_STOP_TIME",
            "SPACECRAFT_CLOCK_START_COUNT",
            "SPACECRAFT_CLOCK_STOP_COUNT",
        ]:
            value = _find_label_value(image_label, key)
            if value is None:
                continue
            try:
                return float(spiceypy.utc2et(_to_spice_utc_string(value)))
            except Exception:
                continue
        raise ValueError("Could not resolve observation time from image label.")

    def _pixel_rays(self, image_shape: tuple[int, int]) -> np.ndarray:
        """Build per-pixel camera rays by bilinear interpolation of FOV corner vectors."""
        bounds = np.asarray(self.bounds, dtype=np.float64)
        if bounds.shape[0] == 3:
            corners = bounds.T
        else:
            corners = bounds

        if corners.shape[0] < 4:
            raise RuntimeError("Camera FOV does not provide rectangular bounds.")

        c0, c1, c2, c3 = corners[:4]
        ny, nx = image_shape
        u = np.linspace(0.0, 1.0, nx, dtype=np.float64)
        v = np.linspace(0.0, 1.0, ny, dtype=np.float64)

        top = (1.0 - u)[:, None] * c0 + u[:, None] * c1
        bottom = (1.0 - u)[:, None] * c3 + u[:, None] * c2

        rays = (1.0 - v)[:, None, None] * top[None, :, :] + v[:, None, None] * bottom[None, :, :]
        rays = np.transpose(rays, (0, 1, 2))
        rays /= np.linalg.norm(rays, axis=2, keepdims=True)
        return rays

    @staticmethod
    def _normalize_rows(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0.0] = np.nan
        return vecs / norms

    @staticmethod
    def _bilinear_sample(grid: np.ndarray, rowf: np.ndarray, colf: np.ndarray) -> np.ndarray:
        """Bilinear sampling on a 2D grid at floating row/column coordinates."""
        nrows, ncols = grid.shape
        r0 = np.floor(rowf).astype(np.int64)
        c0 = np.floor(colf).astype(np.int64)
        r1 = np.clip(r0 + 1, 0, nrows - 1)
        c1 = np.clip(c0 + 1, 0, ncols - 1)

        r0 = np.clip(r0, 0, nrows - 1)
        c0 = np.clip(c0, 0, ncols - 1)

        dr = rowf - r0
        dc = colf - c0

        v00 = grid[r0, c0]
        v01 = grid[r0, c1]
        v10 = grid[r1, c0]
        v11 = grid[r1, c1]

        return (
            (1.0 - dr) * (1.0 - dc) * v00
            + (1.0 - dr) * dc * v01
            + dr * (1.0 - dc) * v10
            + dr * dc * v11
        )

    def _sample_topography(self, lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample elevation and gradients from loaded DTM tiles for each lat/lon point."""
        n = lat_deg.size
        elev = np.full(n, np.nan, dtype=np.float64)
        dh_dlat = np.full(n, np.nan, dtype=np.float64)
        dh_dlon = np.full(n, np.nan, dtype=np.float64)

        lon_mod = np.mod(lon_deg, 360.0)
        assigned = np.zeros(n, dtype=bool)

        for tile in self.dtm_tiles:
            tile_lon_min = tile.lon_min_deg
            tile_lon_max = tile.lon_max_deg
            if tile_lon_max <= tile_lon_min:
                tile_lon_max += 360.0

            work_lon = lon_mod.copy()
            if tile_lon_max > 360.0:
                work_lon[work_lon < tile_lon_min] += 360.0

            inside = (
                (~assigned)
                & (lat_deg >= tile.lat_min_deg)
                & (lat_deg <= tile.lat_max_deg)
                & (work_lon >= tile_lon_min)
                & (work_lon <= tile_lon_max)
            )
            if not np.any(inside):
                continue

            idx = np.where(inside)[0]
            nrows, ncols = tile.elevation.shape
            rowf = (tile.lat_max_deg - lat_deg[idx]) / max(1e-12, (tile.lat_max_deg - tile.lat_min_deg)) * (nrows - 1)
            colf = (work_lon[idx] - tile_lon_min) / max(1e-12, (tile_lon_max - tile_lon_min)) * (ncols - 1)

            elev[idx] = self._bilinear_sample(tile.elevation, rowf, colf)
            row_plus = rowf + 1.0
            row_minus = rowf - 1.0
            col_plus = colf + 1.0
            col_minus = colf - 1.0

            lat_step = np.deg2rad((tile.lat_max_deg - tile.lat_min_deg) / max(1, nrows - 1))
            lon_step = np.deg2rad((tile_lon_max - tile_lon_min) / max(1, ncols - 1))

            e_row_plus = self._bilinear_sample(tile.elevation, row_plus, colf)
            e_row_minus = self._bilinear_sample(tile.elevation, row_minus, colf)
            e_col_plus = self._bilinear_sample(tile.elevation, rowf, col_plus)
            e_col_minus = self._bilinear_sample(tile.elevation, rowf, col_minus)

            # row index increases toward lower latitude, hence negative sign for dH/dlat.
            dh_dlat[idx] = -((e_row_plus - e_row_minus) / max(2.0 * lat_step, 1e-12))
            dh_dlon[idx] = (e_col_plus - e_col_minus) / max(2.0 * lon_step, 1e-12)
            assigned[idx] = True

        return elev, dh_dlat, dh_dlon

    def _surface_normals_from_dtm(self, spoints: np.ndarray) -> np.ndarray:
        """Compute DTM-aware local normals at intercept points in body-fixed frame."""
        x = spoints[:, 0]
        y = spoints[:, 1]
        z = spoints[:, 2]
        r = np.linalg.norm(spoints, axis=1)

        lat = np.arcsin(np.clip(z / r, -1.0, 1.0))
        lon = np.arctan2(y, x)
        lat_deg = np.rad2deg(lat)
        lon_deg = np.mod(np.rad2deg(lon), 360.0)

        elev, dh_dlat, dh_dlon = self._sample_topography(lat_deg, lon_deg)
        valid = np.isfinite(elev) & np.isfinite(dh_dlat) & np.isfinite(dh_dlon)

        normals = np.full_like(spoints, np.nan, dtype=np.float64)
        if not np.any(valid):
            return normals

        lat_v = lat[valid]
        lon_v = lon[valid]
        cos_lat = np.cos(lat_v)
        sin_lat = np.sin(lat_v)
        cos_lon = np.cos(lon_v)
        sin_lon = np.sin(lon_v)

        up = np.column_stack((cos_lat * cos_lon, cos_lat * sin_lon, sin_lat))
        east = np.column_stack((-sin_lon, cos_lon, np.zeros_like(sin_lon)))
        north = np.column_stack((-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat))

        radius = np.maximum(self.vesta_radius_m + elev[valid], 1.0)
        tangent_lon = east * (radius * cos_lat)[:, None] + up * dh_dlon[valid][:, None]
        tangent_lat = north * radius[:, None] + up * dh_dlat[valid][:, None]

        nvec = np.cross(tangent_lon, tangent_lat)
        nvec = self._normalize_rows(nvec)
        normals[valid] = nvec
        return normals

    def _log_pointing_diagnostics(self, et: float, image_shape: tuple[int, int], rays: np.ndarray, image_id: str) -> None:
        """Log frame, boresight, and FOV diagnostics for one image/time."""
        cy = image_shape[0] // 2
        cx = image_shape[1] // 2

        center_ray_cam = np.asarray(rays[cy, cx], dtype=np.float64)
        boresight_cam = np.asarray(self.boresight, dtype=np.float64)

        logging.info(
            "Frame check for %s: instrument=%s cam_frame=%s target_frame=%s",
            image_id,
            self.instrument,
            self.cam_frame,
            self.target_frame,
        )
        logging.info(
            "Center-pixel ray in %s for %s: x=%.9f y=%.9f z=%.9f",
            self.cam_frame,
            image_id,
            center_ray_cam[0],
            center_ray_cam[1],
            center_ray_cam[2],
        )
        logging.info(
            "Boresight vector in %s for %s: x=%.9f y=%.9f z=%.9f",
            self.cam_frame,
            image_id,
            boresight_cam[0],
            boresight_cam[1],
            boresight_cam[2],
        )

        try:
            rot = spiceypy.pxform(self.cam_frame, self.target_frame, et)
            center_ray_target = np.asarray(spiceypy.mxv(rot, center_ray_cam), dtype=np.float64)
            boresight_target = np.asarray(spiceypy.mxv(rot, boresight_cam), dtype=np.float64)
            logging.info(
                "Center-pixel ray in %s for %s: x=%.9f y=%.9f z=%.9f",
                self.target_frame,
                image_id,
                center_ray_target[0],
                center_ray_target[1],
                center_ray_target[2],
            )
            logging.info(
                "Boresight vector in %s for %s: x=%.9f y=%.9f z=%.9f",
                self.target_frame,
                image_id,
                boresight_target[0],
                boresight_target[1],
                boresight_target[2],
            )
        except spiceypy.support_types.SpiceyError as exc:
            logging.warning("Could not transform center/boresight vectors to %s: %s", self.target_frame, exc)

        try:
            obs_pos, _ = spiceypy.spkpos(
                self.observer,
                et,
                self.target_frame,
                self.aberration_correction,
                self.target,
            )
            center_range_km = float(np.linalg.norm(np.asarray(obs_pos, dtype=np.float64)))

            spoint, _, srfvec = spiceypy.subpnt(
                "Intercept: ellipsoid",
                self.target,
                et,
                self.target_frame,
                self.aberration_correction,
                self.observer,
            )
            subpnt_range_km = float(np.linalg.norm(np.asarray(srfvec, dtype=np.float64)))
            spoint = np.asarray(spoint, dtype=np.float64)

            logging.info(
                "subpnt diagnostics for %s: spacecraft_to_center=%.3f km spacecraft_to_subpnt=%.3f km subpnt=(%.3f, %.3f, %.3f) km",
                image_id,
                center_range_km,
                subpnt_range_km,
                spoint[0],
                spoint[1],
                spoint[2],
            )
        except spiceypy.support_types.SpiceyError as exc:
            logging.warning("subpnt diagnostics failed for %s: %s", image_id, exc)

        try:
            in_fov = bool(
                spiceypy.fovtrg(
                    self.instrument,
                    self.target,
                    "ELLIPSOID",
                    self.target_frame,
                    self.aberration_correction,
                    self.observer,
                    et,
                )
            )
            logging.info("FOV test for %s: target=%s in_%s_FOV=%s", image_id, self.target, self.instrument, in_fov)
            if not in_fov:
                logging.warning(
                    "fovtrg returned False for %s at this ET. This points to timing/frame/kernel inconsistency.",
                    image_id,
                )
        except spiceypy.support_types.SpiceyError as exc:
            logging.warning("fovtrg diagnostics failed for %s: %s", image_id, exc)

    def compute_geometry(self, image_file_path: str) -> pd.DataFrame:
        """
        Compute incidence/emission/phase geometry for one calibrated image.

        Args:
            image_file_path (str): Path to one Dawn FC image (.IMG).

        Returns:
            pd.DataFrame: Flattened valid geometry table with columns:
            [pixel_x, pixel_y, iof, incidence, emission, phase]
        """
        image_path = Path(image_file_path)
        image_id = image_path.stem
        logging.info("Computing geometry for image: %s", image_id)

        try:
            pds_img = self._planetaryimage.PDS3Image.open(str(image_path))
            iof_data = pds_img.image.astype(np.float32)
            et = self._extract_observation_et(pds_img.label)
            logging.info("Loaded I/F and observation time from %s", image_path)
        except Exception as e:
            raise RuntimeError(f"Unable to read image/time for {image_path}: {e}")

        image_shape = iof_data.shape
        if iof_data.ndim != 2:
            raise ValueError(f"Expected 2D image array, got shape {image_shape}")

        rays = self._pixel_rays(image_shape)
        rays_flat = rays.reshape(-1, 3)

        self._log_pointing_diagnostics(et, image_shape, rays, image_id)

        n_pix = rays_flat.shape[0]
        spoints = np.full((n_pix, 3), np.nan, dtype=np.float64)
        valid = np.zeros(n_pix, dtype=bool)

        method = "ELLIPSOID"
        logging.info("Tracing %d rays with SPICE sincpt...", n_pix)
        for idx in range(n_pix):
            try:
                spoint, _, _ = spiceypy.sincpt(
                    method,
                    self.target,
                    et,
                    self.target_frame,
                    self.aberration_correction,
                    self.observer,
                    self.cam_frame,
                    rays_flat[idx],
                )
                spoints[idx] = spoint
                valid[idx] = True
            except spiceypy.support_types.SpiceyError:
                continue

        n_intercepts = int(np.count_nonzero(valid))
        if n_intercepts == 0:
            logging.warning(
                "Geometry diagnostics for %s: total=%d, intercepts=0 (method=%s)",
                image_id,
                n_pix,
                method,
            )

        normals = self._surface_normals_from_dtm(spoints[valid])
        normals_finite = np.isfinite(normals).all(axis=1)
        n_normals_finite = int(np.count_nonzero(normals_finite))

        sun_pos, _ = spiceypy.spkpos("SUN", et, self.target_frame, self.aberration_correction, self.target)
        obs_pos, _ = spiceypy.spkpos(self.observer, et, self.target_frame, self.aberration_correction, self.target)

        v_spoints = spoints[valid]
        to_sun = self._normalize_rows(np.asarray(sun_pos)[None, :] - v_spoints)
        to_obs = self._normalize_rows(np.asarray(obs_pos)[None, :] - v_spoints)
        sun_obs_finite = np.isfinite(to_sun).all(axis=1) & np.isfinite(to_obs).all(axis=1)
        n_sun_obs_finite = int(np.count_nonzero(sun_obs_finite))

        # Angle computations are fully vectorized once intercepts are known.
        incidence = np.rad2deg(np.arccos(np.clip(np.einsum("ij,ij->i", normals, to_sun), -1.0, 1.0)))
        emission = np.rad2deg(np.arccos(np.clip(np.einsum("ij,ij->i", normals, to_obs), -1.0, 1.0)))
        phase = np.rad2deg(np.arccos(np.clip(np.einsum("ij,ij->i", to_sun, to_obs), -1.0, 1.0)))
        angles_finite = np.isfinite(incidence) & np.isfinite(emission) & np.isfinite(phase)
        n_angles_finite = int(np.count_nonzero(angles_finite))

        iof_flat = iof_data.reshape(-1)
        iof_finite_hits = np.isfinite(iof_flat[valid])
        n_iof_finite_hits = int(np.count_nonzero(iof_finite_hits))

        incidence_all = np.full(n_pix, np.nan, dtype=np.float64)
        emission_all = np.full(n_pix, np.nan, dtype=np.float64)
        phase_all = np.full(n_pix, np.nan, dtype=np.float64)
        incidence_all[valid] = incidence
        emission_all[valid] = emission
        phase_all[valid] = phase

        valid_pixels = (
            valid
            & np.isfinite(iof_flat)
            & np.isfinite(incidence_all)
            & np.isfinite(emission_all)
            & np.isfinite(phase_all)
        )
        n_final = int(np.count_nonzero(valid_pixels))

        logging.info(
            (
                "Geometry diagnostics for %s: total=%d, intercepts=%d (%.1f%%), "
                "finite_normals=%d, finite_sun_obs=%d, finite_angles=%d, finite_iof_on_hits=%d, "
                "final_rows=%d (%.1f%%)"
            ),
            image_id,
            n_pix,
            n_intercepts,
            100.0 * n_intercepts / max(1, n_pix),
            n_normals_finite,
            n_sun_obs_finite,
            n_angles_finite,
            n_iof_finite_hits,
            n_final,
            100.0 * n_final / max(1, n_pix),
        )

        if n_final == 0:
            logging.warning(
                (
                    "All rows filtered for %s. Drop-off summary: no_intercept=%d, bad_normal_or_vectors=%d, "
                    "bad_angles=%d, nonfinite_iof_on_hits=%d"
                ),
                image_id,
                n_pix - n_intercepts,
                max(0, n_intercepts - min(n_normals_finite, n_sun_obs_finite)),
                max(0, n_intercepts - n_angles_finite),
                max(0, n_intercepts - n_iof_finite_hits),
            )

        yy, xx = np.indices(image_shape)
        pixel_x = xx.reshape(-1)[valid_pixels]
        pixel_y = yy.reshape(-1)[valid_pixels]

        df = pd.DataFrame(
            {
                "pixel_x": pixel_x.astype(np.int32),
                "pixel_y": pixel_y.astype(np.int32),
                "iof": iof_flat[valid_pixels].astype(np.float32),
                "incidence": incidence_all[valid_pixels].astype(np.float32),
                "emission": emission_all[valid_pixels].astype(np.float32),
                "phase": phase_all[valid_pixels].astype(np.float32),
            }
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{image_id}_geometry.parquet"
        df.to_parquet(output_path, engine="pyarrow", index=False)
        logging.info("Saved geometry table: %s (%d rows)", output_path, len(df))
        return df

    def __del__(self):
        """Unload SPICE kernels when the object is destroyed."""
        try:
            spiceypy.kclear()
            logging.info("SPICE kernels have been unloaded.")
        except Exception as e:
            logging.error(f"Error unloading SPICE kernels: {e}")

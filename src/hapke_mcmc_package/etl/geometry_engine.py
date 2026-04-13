from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import spiceypy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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

        def _fromstring_compat(string, dtype=float, count=-1, sep=""):
            if sep == "" and isinstance(string, (bytes, bytearray, memoryview)):
                return np.frombuffer(string, dtype=dtype, count=count)
            return old_fromstring(string, dtype=dtype, count=count, sep=sep)

        np.fromstring = _fromstring_compat  # type: ignore[assignment]
        import planetaryimage

        logging.warning("Applied NumPy compatibility shim for planetaryimage import.")
        return planetaryimage


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
            dt = dt.replace(tzinfo=UTC)
        dt_utc = dt.astimezone(UTC)
        return dt_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        return text.replace(" ", "T")


def _log_fatal_geometry_missing(context: str, exc: Exception) -> None:
    """Log a fatal geometry failure and preserve the original exception."""
    logging.critical("FATAL GEOMETRY MISSING: %s: %s", context, exc, exc_info=True)


class GeometryEngine:
    """
    Core SPICE-based ray-tracing engine to calculate photometric angles.
    """

    def __init__(self, data_root: str, metakernel_path: str, body_fixed_frame: str = "IAU_VESTA"):
        """
        Initialize SPICE and load Vesta DTM products.

        Args:
            data_root (str): Root data directory (scratch-backed), containing
                02_spice_kernels, 03_dtm, and 04_geometry_tables subdirectories.
            metakernel_path (str): Explicit path to the metakernel (.tm) file.
            body_fixed_frame (str): Body-fixed reference frame for all geometry calculations.
                Default is 'IAU_VESTA'. For high-resolution DSK models with localized frames
                (e.g., Claudia Double-Prime), pass the DSK-native frame name.
        """
        self.data_root = Path(data_root)
        self.spice_dir = self.data_root / "02_spice_kernels"
        self.output_dir = self.data_root / "04_geometry_tables"
        self._planetaryimage = _load_planetaryimage_module()

        self.metakernel_path = Path(metakernel_path)
        if not self.metakernel_path.exists():
            raise FileNotFoundError(f"Metakernel not found: {self.metakernel_path}")

        self._initialize_spice()

        # Camera constants for Dawn's Framing Camera (FC2)
        self.instrument = "DAWN_FC2"
        self.target = "VESTA"
        self.aberration_correction = "LT+S"
        self.body_fixed_frame = body_fixed_frame
        self.observer = "DAWN"
        # SCIENTIFIC DECISION - surface model is ELLIPSOID.
        # This was confirmed valid by runtime log analysis of job 25268078 on 2026-04-11.
        # Do not change this to DSK without discussing with supervisor and updating all existing parquet outputs.
        self.surface_intercept_method = "ELLIPSOID"

        # Get camera FOV details
        try:
            self.cam_id = spiceypy.bodn2c(self.instrument)
            _, self.cam_frame, self.boresight, self.num_bounds, self.bounds = spiceypy.getfov(
                self.cam_id, 4
            )
            self.cam_frame = _as_str(self.cam_frame)
            logging.info(f"Successfully loaded camera model for {self.instrument}")
        except Exception as e:
            _log_fatal_geometry_missing(f"camera FOV lookup failed for {self.instrument}", e)
            raise

    def _initialize_spice(self) -> None:
        """Initialize SPICE from the explicitly configured metakernel."""
        try:
            spiceypy.furnsh(str(self.metakernel_path))
        except spiceypy.support_types.SpiceyError as exc:
            _log_fatal_geometry_missing(f"metakernel load failed: {self.metakernel_path}", exc)
            raise

        # Quick sanity check that time conversion is available (LSK loaded).
        try:
            spiceypy.utc2et("2000-01-01T12:00:00")
        except spiceypy.support_types.SpiceyError as exc:
            raise RuntimeError(
                f"SPICE fallback loaded kernels but time conversion failed: {exc}"
            ) from exc

        logging.info("Successfully loaded SPICE metakernel: %s", self.metakernel_path)

    def _extract_observation_et(self, image_label: Any) -> float:
        """Extract mid-exposure ET from SCLK start count plus exposure duration."""
        sclk_start = _find_label_value(image_label, "SPACECRAFT_CLOCK_START_COUNT")
        if sclk_start is None:
            raise ValueError("Could not resolve SPACECRAFT_CLOCK_START_COUNT from image label.")

        exposure_duration = _find_label_value(image_label, "EXPOSURE_DURATION")
        if exposure_duration is None:
            raise ValueError("Could not resolve EXPOSURE_DURATION from image label.")

        try:
            start_et = float(spiceypy.scs2e(-203, _as_str(sclk_start)))
        except spiceypy.support_types.SpiceyError as exc:
            _log_fatal_geometry_missing("scs2e failed for SPACECRAFT_CLOCK_START_COUNT", exc)
            raise

        try:
            exposure_seconds = float(exposure_duration)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid EXPOSURE_DURATION value: {exposure_duration!r}") from exc

        return start_et + 0.5 * exposure_seconds

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

    def _log_pointing_diagnostics(
        self, et: float, image_shape: tuple[int, int], rays: np.ndarray, image_id: str
    ) -> None:
        """Log frame, boresight, and FOV diagnostics for one image/time."""
        cy = image_shape[0] // 2
        cx = image_shape[1] // 2

        center_ray_cam = np.asarray(rays[cy, cx], dtype=np.float64)
        boresight_cam = np.asarray(self.boresight, dtype=np.float64)

        logging.info(
            "Frame check for %s: instrument=%s cam_frame=%s body_fixed_frame=%s",
            image_id,
            self.instrument,
            self.cam_frame,
            self.body_fixed_frame,
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
            rot = spiceypy.pxform(self.cam_frame, self.body_fixed_frame, et)
            center_ray_target = np.asarray(spiceypy.mxv(rot, center_ray_cam), dtype=np.float64)
            boresight_target = np.asarray(spiceypy.mxv(rot, boresight_cam), dtype=np.float64)
            logging.info(
                "Center-pixel ray in %s for %s: x=%.9f y=%.9f z=%.9f",
                self.body_fixed_frame,
                image_id,
                center_ray_target[0],
                center_ray_target[1],
                center_ray_target[2],
            )
            logging.info(
                "Boresight vector in %s for %s: x=%.9f y=%.9f z=%.9f",
                self.body_fixed_frame,
                image_id,
                boresight_target[0],
                boresight_target[1],
                boresight_target[2],
            )
        except spiceypy.support_types.SpiceyError as exc:
            _log_fatal_geometry_missing(f"pxform failed for {image_id}", exc)
            raise

        try:
            obs_pos, _ = spiceypy.spkpos(
                self.observer,
                et,
                self.body_fixed_frame,
                self.aberration_correction,
                self.target,
            )
            center_range_km = float(np.linalg.norm(np.asarray(obs_pos, dtype=np.float64)))

            spoint, _, srfvec = spiceypy.subpnt(
                "Intercept: ellipsoid",
                self.target,
                et,
                self.body_fixed_frame,
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
                    self.body_fixed_frame,
                    self.aberration_correction,
                    self.observer,
                    et,
                )
            )
            logging.info(
                "FOV test for %s: target=%s in_%s_FOV=%s",
                image_id,
                self.target,
                self.instrument,
                in_fov,
            )
            if not in_fov:
                logging.warning(
                    "fovtrg returned False for %s at this ET. This points to timing/frame/kernel inconsistency.",
                    image_id,
                )
        except spiceypy.support_types.SpiceyError as exc:
            _log_fatal_geometry_missing(f"fovtrg diagnostics failed for {image_id}", exc)
            raise

    @staticmethod
    def _phase_subdir_from_image_path(image_path: Path) -> str:
        """Resolve phase output subdir from an image path; defaults to survey."""
        phase_names = ("rc", "survey", "hamo", "lamo")
        parts_lower = [part.lower() for part in image_path.parts]
        for phase in phase_names:
            if phase in parts_lower:
                return phase
        return "survey"

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
        phase_subdir = self._phase_subdir_from_image_path(image_path)
        logging.info("Computing geometry for image: %s", image_id)

        try:
            pds_img = self._planetaryimage.PDS3Image.open(str(image_path))
            iof_data = pds_img.image.astype(np.float32)
            et = self._extract_observation_et(pds_img.label)
            logging.info("Loaded I/F and observation time from %s", image_path)
        except Exception as e:
            _log_fatal_geometry_missing(f"unable to read image/time for {image_path}", e)
            raise

        image_shape = iof_data.shape
        if iof_data.ndim != 2:
            raise ValueError(f"Expected 2D image array, got shape {image_shape}")

        rays = self._pixel_rays(image_shape)
        rays_flat = rays.reshape(-1, 3)

        self._log_pointing_diagnostics(et, image_shape, rays, image_id)

        n_pix = rays_flat.shape[0]
        spoints = np.full((n_pix, 3), np.nan, dtype=np.float64)

        method = self.surface_intercept_method
        logging.info("Tracing %d rays with SPICE sincpt using method=%s...", n_pix, method)
        phase = np.full(n_pix, np.nan, dtype=np.float64)
        incidence = np.full(n_pix, np.nan, dtype=np.float64)
        emission = np.full(n_pix, np.nan, dtype=np.float64)
        latitude = np.full(n_pix, np.nan, dtype=np.float64)
        longitude = np.full(n_pix, np.nan, dtype=np.float64)
        for idx in range(n_pix):
            try:
                spoint, _, _ = spiceypy.sincpt(
                    self.surface_intercept_method,
                    self.target,
                    et,
                    self.body_fixed_frame,
                    self.aberration_correction,
                    self.observer,
                    self.cam_frame,
                    rays_flat[idx],
                )
            except spiceypy.utils.exceptions.NotFoundError:
                # Pixel looks at background space; leave NaNs in place and continue.
                continue
            except spiceypy.support_types.SpiceyError as exc:
                _log_fatal_geometry_missing(
                    f"sincpt failed for image={image_id} pixel={idx} method={method}", exc
                )
                raise
            spoints[idx] = spoint

            # Extract lat/lon from surface point
            try:
                radius, lon_rad, lat_rad = spiceypy.reclat(spoint)
                longitude[idx] = np.degrees(float(lon_rad))
                latitude[idx] = np.degrees(float(lat_rad))
            except spiceypy.support_types.SpiceyError:
                pass

            try:
                illumf_result = spiceypy.illumf(
                    self.surface_intercept_method,
                    self.target,
                    "SUN",
                    et,
                    self.body_fixed_frame,
                    self.aberration_correction,
                    self.observer,
                    spoint,
                )
                phase_rad = illumf_result[2]
                incdnc_rad = illumf_result[3]
                emissn_rad = illumf_result[4]
            except spiceypy.support_types.SpiceyError as exc:
                _log_fatal_geometry_missing(
                    f"illumf failed for image={image_id} pixel={idx} method={method}", exc
                )
                raise

            phase[idx] = float(phase_rad)
            incidence[idx] = float(incdnc_rad)
            emission[idx] = float(emissn_rad)

        if not np.isfinite(spoints).any():
            raise RuntimeError(
                f"FATAL GEOMETRY MISSING: non-finite intercept vector produced for {image_id}"
            )

        phase = np.rad2deg(phase)
        incidence = np.rad2deg(incidence)
        emission = np.rad2deg(emission)

        if (
            not np.isfinite(phase).any()
            or not np.isfinite(incidence).any()
            or not np.isfinite(emission).any()
        ):
            raise RuntimeError(
                f"FATAL GEOMETRY MISSING: non-finite illumination angles computed for {image_id}"
            )

        iof_flat = iof_data.reshape(-1)
        if not np.isfinite(iof_flat).all():
            raise RuntimeError(f"FATAL GEOMETRY MISSING: non-finite I/F values in {image_id}")

        if not np.all(np.isfinite(spoints)):
            raise RuntimeError(f"FATAL GEOMETRY MISSING: invalid intercept rows in {image_id}")

        yy, xx = np.indices(image_shape)
        pixel_x = xx.reshape(-1)
        pixel_y = yy.reshape(-1)

        df = pd.DataFrame(
            {
                "image_id": np.full(n_pix, image_id, dtype=object),
                "pixel_x": pixel_x.astype(np.int32),
                "pixel_y": pixel_y.astype(np.int32),
                "iof": iof_flat.astype(np.float32),
                "incidence": incidence.astype(np.float32),
                "emission": emission.astype(np.float32),
                "phase": phase.astype(np.float32),
                "latitude": latitude.astype(np.float32),
                "longitude": longitude.astype(np.float32),
            }
        )

        output_phase_dir = self.output_dir / phase_subdir
        output_phase_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_phase_dir / f"{image_id}_geometry.parquet"
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

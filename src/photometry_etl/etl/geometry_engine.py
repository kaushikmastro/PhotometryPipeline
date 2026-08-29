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

# FORMALIZED EXCEPTION POLICY CONSTANTS
# These represent physically expected boundary conditions where an infinitely thin pixel ray
# grazing the rugged limb of Vesta misses or is obstructed by local micro-topography.
ALLOWABLE_DSK_SHORT_CODES = {
    "SPICE(DSKXTAIL)",       # Ray traces past the back/tail end of a DSK component or limb facet
    "SPICE(NOFRAMECONNECT)", # Occurs on deep-space background pixel transitions near the limb
    "SPICE(POINTNOTONFACE)"  # Calculated intercept point falls slightly outside the local bounding facet
}


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


class _DetachedFitsImage:
    """pds_img-compatible adapter for .FIT images with a detached PDS3 .LBL label.

    Exposes `.label` (dict-like PVL structure) and `.image` (2D pixel array),
    the only two attributes compute_geometry() reads off a PDS3Image.
    """

    def __init__(self, image_path: Path):
        import pvl
        from astropy.io import fits

        label_path = image_path.with_suffix(".LBL")
        if not label_path.exists():
            label_path = image_path.with_suffix(".lbl")
        if not label_path.exists():
            raise FileNotFoundError(f"No detached PDS3 label found for {image_path}")

        self.label = pvl.load(str(label_path))

        with fits.open(str(image_path)) as hdul:
            # Verified single-HDU file: PRIMARY, (1024, 1024), float32.
            # Copy out of the mmap before the file closes.
            self.image = np.array(hdul[0].data)


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


def calibrate_iof_data(raw_image: np.ndarray, image_id: str, distance_au: float, f_solar: float) -> np.ndarray:

    """Convert radiance to I/F, masking PDS flags, cosmic rays, and dead pixels."""

    # Physical Radiance-to-I/F equation
    iof_data = (np.asarray(raw_image, dtype=np.float32) * np.pi * (distance_au ** 2)) / f_solar 

    # 2. Cosmic Ray & PDS Fill Value Masking
    # Physical I/F for Vesta will not exceed ~1.5 even at opposition. 
    # We set a strict physical ceiling of 2.0. Anything above is a cosmic ray or PDS flag.
    # We set a floor of -0.05 to allow for slight CCD noise floor fluctuations.
    valid_mask = np.isfinite(iof_data) & (iof_data >= -0.05) & (iof_data <= 2.0)
    
    # Set all invalid pixels (space, flags, cosmic rays) to NaN
    iof_data[~valid_mask] = np.nan
    
    # 3. Extract only the physically valid pixels for logging
    valid_iof = iof_data[valid_mask]

    if valid_iof.size == 0:
        raise RuntimeError(f"FATAL GEOMETRY MISSING: I/F array contains no valid physical pixels for {image_id}")

    iof_min = float(np.min(valid_iof))
    max_iof = float(np.max(valid_iof))

    # 4. Gentle Clipping for slight noise floor
    clip_limit = 1.5
    if iof_min < 0.0 or max_iof > clip_limit:
        logging.debug(
            "I/F minor bounds adjustment for %s (min=%.6f, max=%.6f); clipping to [0, %.1f].",
            image_id, iof_min, max_iof, clip_limit
        )
        # Only clip the valid pixels, leave NaNs alone
        iof_data[valid_mask] = np.clip(valid_iof, 0.0, clip_limit)

    return iof_data


def compute_pixel_rays(bounds: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Build per-pixel camera ray unit vectors by bilinear interpolation of the 4 FOV
    corner vectors returned by spiceypy.getfov(). Pure function of camera geometry only
    (no target/epoch dependence), factored out of GeometryEngine._pixel_rays so it can be
    unit-tested without a live SPICE session.

    Returns an (ny, nx, 3) array of unit vectors in the camera frame.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    corners = bounds.T if bounds.shape[0] == 3 else bounds

    if corners.shape[0] < 4:
        raise RuntimeError("Camera FOV does not provide rectangular bounds.")

    c0, c1, c2, c3 = corners[:4]
    ny, nx = image_shape
    u = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    v = np.linspace(0.0, 1.0, ny, dtype=np.float64)

    top = (1.0 - u)[:, None] * c0 + u[:, None] * c1
    bottom = (1.0 - u)[:, None] * c3 + u[:, None] * c2

    rays = (1.0 - v)[:, None, None] * top[None, :, :] + v[:, None, None] * bottom[None, :, :]
    rays /= np.linalg.norm(rays, axis=2, keepdims=True)
    return rays


def compute_pixel_solid_angles(rays: np.ndarray) -> np.ndarray:
    """Per-pixel solid angle (steradians) from a camera ray-direction grid.

    Small-angle parallelogram approximation: at each pixel, the solid angle is the
    magnitude of the cross product of the local one-pixel-step tangent vectors along the
    two image axes (|d_col x d_row|). This depends only on camera geometry (the ray grid
    from compute_pixel_rays), not on the target intercept or epoch, so it is the same for
    every image from a given instrument/detector-window combination.

    The last row and last column have no "next" neighbor to forward-difference against;
    rather than pad with a duplicated ray (which would zero out that edge's difference and
    understate its solid angle), they reuse the adjacent interior pixel's step vector — a
    much closer approximation than an artificial zero, and the FOV interpolation is smooth
    enough that neighboring pixels' angular steps do not differ meaningfully.

    Returns an (ny, nx) array of solid angles in steradians.
    """
    rays = np.asarray(rays, dtype=np.float64)
    ny, nx, _ = rays.shape
    if ny < 2 or nx < 2:
        raise ValueError("Need at least a 2x2 pixel grid to estimate per-pixel solid angle.")

    d_col = np.empty_like(rays)
    d_col[:, :-1, :] = rays[:, 1:, :] - rays[:, :-1, :]
    d_col[:, -1, :] = d_col[:, -2, :]

    d_row = np.empty_like(rays)
    d_row[:-1, :, :] = rays[1:, :, :] - rays[:-1, :, :]
    d_row[-1, :, :] = d_row[-2, :, :]

    cross = np.cross(d_col, d_row)
    return np.linalg.norm(cross, axis=2)


def compute_pixel_area_km2(
    pixel_solid_angle_sr: np.ndarray,
    range_km: np.ndarray,
    emission_deg: np.ndarray,
    max_emission_for_area_deg: float,
) -> np.ndarray:
    """TRUE (unforeshortened) surface area per pixel on the body: solid_angle *
    range^2 / cos(emission) — dividing by cos(emission) undoes the foreshortening that
    solid_angle*range^2 alone gives you (see compute_projected_area_km2), recovering the
    actual physical footprint on Vesta's surface. Useful for mapping/resolution questions
    ("how many km^2 of surface does this pixel sample") — NOT the right weight for
    disk-integrated photometry; see photometry.aggregation.disk_integrate's module
    docstring for why, and use compute_projected_area_km2 for that instead.

    NaN, not clipped, wherever emission_deg > max_emission_for_area_deg or either input
    is non-finite: the projection genuinely diverges as emission -> 90 deg, so silently
    clipping would hand downstream area-weighted aggregation a plausible-looking but
    wrong finite number. Callers must handle NaN areas explicitly.

    All three array arguments must be the same shape; returns that same shape.
    """
    pixel_solid_angle_sr = np.asarray(pixel_solid_angle_sr, dtype=np.float64)
    range_km = np.asarray(range_km, dtype=np.float64)
    emission_deg = np.asarray(emission_deg, dtype=np.float64)

    area = np.full(emission_deg.shape, np.nan, dtype=np.float64)
    computable = (
        np.isfinite(emission_deg)
        & (emission_deg <= max_emission_for_area_deg)
        & np.isfinite(range_km)
        & np.isfinite(pixel_solid_angle_sr)
    )
    cos_emission = np.cos(np.deg2rad(emission_deg[computable]))
    area[computable] = (
        pixel_solid_angle_sr[computable] * range_km[computable] ** 2 / cos_emission
    )
    return area


def compute_projected_area_km2(pixel_solid_angle_sr: np.ndarray, range_km: np.ndarray) -> np.ndarray:
    """Projected (sky-plane) area per pixel: solid_angle * range^2 — the area a flat
    patch perpendicular to the line of sight would need to subtend the same angular size
    at that range. This IS the foreshortened footprint of the pixel as the observer
    actually sees it (no cos(emission) division), and is the correct integration weight
    for disk-integrated photometry: total flux received by a distant observer is
    proportional to sum(reflectance * projected_area), matching the standard definition
    used in Li et al. 2013 and published disk-integrated phase curves. See
    photometry.aggregation.disk_integrate's module docstring for the full reasoning and
    the analytic Lambertian-sphere test that pins this down.

    No singularity as emission -> 90 deg (unlike compute_pixel_area_km2): a limb pixel's
    true surface area diverges, but its projected/foreshortened area shrinks smoothly to
    zero, which is physically correct — a grazing-emission surface element contributes
    ~nothing to observed flux. NaN propagates only from NaN inputs.
    """
    pixel_solid_angle_sr = np.asarray(pixel_solid_angle_sr, dtype=np.float64)
    range_km = np.asarray(range_km, dtype=np.float64)
    return pixel_solid_angle_sr * range_km**2


class GeometryEngine:

    """Core SPICE-based ray-tracing engine supporting parameterized Ellipsoid and DSK structures."""

    def __init__(
        self, 
        data_root: str | Path, 
        metakernel_path: str | Path, 
        surface_intercept_method: str = "ELLIPSOID",
        output_subdir: str = "04_geometry_tables",
        body_fixed_frame: str = "IAU_VESTA",
        aberration_correction: str = "LT+S",
        f_solar: float = 892.0, # 1473.4
        max_emission_for_area_deg: float = 89.0
    ):
        """Initialize SPICE environment. Backward-compatible signature matching for standalone scripts.

        max_emission_for_area_deg: above this emission angle, pixel_area_km2 is emitted as
        NaN rather than a clipped/finite value, since the projected-area formula
        (solid_angle * range^2 / cos(emission)) diverges as emission -> 90 deg. Do not
        change the default without updating the corresponding schema documentation in
        compute_geometry()'s docstring.
        """
        self.data_root = Path(data_root)
        self.spice_dir = self.data_root / "spice_kernels"
        self.output_dir = self.data_root / output_subdir
        self._planetaryimage = _load_planetaryimage_module()

        requested_metakernel_path = Path(metakernel_path)
        dynamic_metakernel_path = self.spice_dir / "dawn_dynamic.tm"
        
        if dynamic_metakernel_path.exists():
            self.metakernel_path = dynamic_metakernel_path
            logging.info("Preferring dynamic SPICE metakernel: %s", self.metakernel_path)
        else:
            self.metakernel_path = requested_metakernel_path
            logging.info("Dynamic metakernel not found; using requested: %s", self.metakernel_path)

        if not self.metakernel_path.exists():
            raise FileNotFoundError(f"Metakernel not found: {self.metakernel_path}")

        self._initialize_spice()

        self.instrument = "DAWN_FC2"
        self.target = "VESTA"
        self.observer = "DAWN"
        
        self.surface_intercept_method = surface_intercept_method
        self.body_fixed_frame = body_fixed_frame
        self.aberration_correction = aberration_correction
        self.f_solar = f_solar
        self.max_emission_for_area_deg = max_emission_for_area_deg

        # Fail fast if DSK mode is requested but no .bds kernels are loaded in the current pool
        method_upper = self.surface_intercept_method.upper()
        if "DSK" in method_upper:
            try:
                num_dsk_kernels = spiceypy.ktotal("DSK")
                if num_dsk_kernels == 0:
                    raise RuntimeError(
                        f"Architecture Mismatch: Tracking mode set to '{self.surface_intercept_method}', "
                        "but zero Digital Shape Kernels (.bds) are loaded in the current SPICE pool. "
                        "Verify your dynamic metakernel configuration."
                    )
                logging.info("DSK pool introspection complete. Located %d active shape kernel(s).", num_dsk_kernels)
            except spiceypy.support_types.SpiceyError as exc:
                _log_fatal_geometry_missing("Failed to query loaded SPICE pool for active DSK counts", exc)
                raise

        logging.info("GeometryEngine operational. Method=%s, Saving to=%s", 
                     self.surface_intercept_method, self.output_dir)

        try:
            self.cam_id = spiceypy.bodn2c(self.instrument)
            _, self.cam_frame, self.boresight, self.num_bounds, self.bounds = spiceypy.getfov(self.cam_id, 4)
            self.cam_frame = _as_str(self.cam_frame)
            logging.info("Successfully loaded camera model for %s", self.instrument)
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

        try:
            spiceypy.utc2et("2000-01-01T12:00:00")
        except spiceypy.support_types.SpiceyError as exc:
            raise RuntimeError(f"SPICE loaded but time conversion failed: {exc}") from exc

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
        return compute_pixel_rays(self.bounds, image_shape)

    def _log_pointing_diagnostics(self, et: float, image_shape: tuple[int, int], rays: np.ndarray, image_id: str) -> None:
        """Log frame, boresight, and FOV diagnostics for one image/time (Non-fatal)."""
        cy, cx = image_shape[0] // 2, image_shape[1] // 2
        center_ray_cam = np.asarray(rays[cy, cx], dtype=np.float64)
        boresight_cam = np.asarray(self.boresight, dtype=np.float64)

        logging.info("Frame check for %s: instrument=%s cam_frame=%s body_fixed_frame=%s",
                     image_id, self.instrument, self.cam_frame, self.body_fixed_frame)

        try:
            rot = spiceypy.pxform(self.cam_frame, self.body_fixed_frame, et)
            center_ray_target = np.asarray(spiceypy.mxv(rot, center_ray_cam), dtype=np.float64)
            boresight_target = np.asarray(spiceypy.mxv(rot, boresight_cam), dtype=np.float64)
            
            logging.info("Center-pixel ray in %s for %s: x=%.9f y=%.9f z=%.9f",
                         self.body_fixed_frame, image_id, center_ray_target[0], center_ray_target[1], center_ray_target[2])
            logging.info("Boresight vector in %s for %s: x=%.9f y=%.9f z=%.9f",
                         self.body_fixed_frame, image_id, boresight_target[0], boresight_target[1], boresight_target[2])
        except spiceypy.support_types.SpiceyError as exc:
            _log_fatal_geometry_missing(f"pxform failed for {image_id}", exc)
            raise

        # Mode-aware diagnostic routing to isolate log execution issues completely from execution truth
        method_upper = self.surface_intercept_method.upper()
        if "ELLIPSOID" in method_upper:
            subpnt_method = "Intercept: ellipsoid"
        elif "DSK" in method_upper:
            subpnt_method = "Intercept: dsk/unprioritized"
        else:
            logging.warning("Unknown tracking intercept configuration context: %s. Skipping subpnt logging.", self.surface_intercept_method)
            subpnt_method = None

        if subpnt_method:
            try:
                obs_pos, _ = spiceypy.spkpos(self.observer, et, self.body_fixed_frame, self.aberration_correction, self.target)
                center_range_km = float(np.linalg.norm(np.asarray(obs_pos, dtype=np.float64)))

                spoint, _, srfvec = spiceypy.subpnt(subpnt_method, self.target, et, self.body_fixed_frame, self.aberration_correction, self.observer)
                subpnt_range_km = float(np.linalg.norm(np.asarray(srfvec, dtype=np.float64)))
                spoint = np.asarray(spoint, dtype=np.float64)

                logging.info("subpnt diagnostics (%s) for %s: obs_to_center=%.3f km obs_to_subpnt=%.3f km subpnt=(%.3f, %.3f, %.3f) km",
                             subpnt_method, image_id, center_range_km, subpnt_range_km, spoint[0], spoint[1], spoint[2])
            except spiceypy.support_types.SpiceyError as exc:
                logging.warning("Non-fatal mode subpnt diagnostics check skipped for %s: %s", image_id, exc)

        try:
            fov_shape_method = "ELLIPSOID" if "ELLIPSOID" in method_upper else "DSK/UNPRIORITIZED"
            in_fov = bool(spiceypy.fovtrg(self.instrument, self.target, fov_shape_method, self.body_fixed_frame, self.aberration_correction, self.observer, et))
            logging.info("FOV test for %s: target=%s in_%s_FOV=%s", image_id, self.target, self.instrument, in_fov)
        except spiceypy.support_types.SpiceyError as exc:
            logging.warning("Non-fatal fovtrg validation diagnostic skipped for %s: %s", image_id, exc)

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
        """Compute incidence/emission/phase geometry for one calibrated image.

        Output schema adds four area-related columns (float64, ingredients kept
        separately from the derived values so either projection can be re-audited or
        revised without re-running SPICE):
          - range_km: spacecraft-to-surface-intercept distance for that pixel
            (norm of sincpt's srfvec), km.
          - pixel_solid_angle_sr: per-pixel angular footprint as seen from the spacecraft,
            steradians. Depends only on camera geometry (FOV + detector window), not on
            the target intercept, so it is identical across images from the same
            instrument/window.
          - pixel_area_km2: TRUE (unforeshortened) surface area on the body, km^2 —
            pixel_solid_angle_sr * range_km^2 / cos(emission). NaN (not clipped) where
            emission > max_emission_for_area_deg, since this diverges as emission ->
            90 deg. Useful for mapping/resolution questions. Do NOT use this to weight a
            disk integral — see projected_area_km2 and
            photometry.aggregation.disk_integrate's module docstring for why.
          - projected_area_km2: projected (sky-plane) area, km^2 — pixel_solid_angle_sr *
            range_km^2, i.e. the same two ingredients WITHOUT dividing by cos(emission).
            No singularity near emission=90 deg. This is the correct weight for
            disk-integrated photometry (total flux to a distant observer is
            proportional to sum(reflectance * projected_area_km2) — the Li et al. 2013 /
            standard disk-integrated-phase-curve convention).
        """
        image_path = Path(image_file_path)
        image_id = image_path.stem
        phase_subdir = self._phase_subdir_from_image_path(image_path)
        logging.info("Computing geometry for image: %s", image_id)

        # GRANULAR ERROR OBSERVABILITY EXTRACTION STAGES
        # Stage 1: PDS Image File IO
        try:
            if image_path.suffix.lower() == ".fit":
                pds_img = _DetachedFitsImage(image_path)
            else:
                pds_img = self._planetaryimage.PDS3Image.open(str(image_path))
        except Exception as e:
            _log_fatal_geometry_missing(f"Label/Image IO operation read failed for target path: {image_path}", e)
            raise

        # Stage 2: Ephemeris Time SCLK Derivation
        try:
            et = self._extract_observation_et(pds_img.label)
        except Exception as e:
            _log_fatal_geometry_missing(f"Spacecraft Clock to ET execution lookup failed for image id: {image_id}", e)
            raise

        # Stage 3: Heliocentric Vectors and Distance Conversions
        try:
            sun_pos, _ = spiceypy.spkpos("SUN", et, "J2000", self.aberration_correction, self.target)
            distance_au = spiceypy.convrt(float(spiceypy.vnorm(sun_pos)), "KM", "AU")
        except spiceypy.support_types.SpiceyError as exc:
            _log_fatal_geometry_missing(f"SPK target vector distance computation failed for image {image_id} at ET {et}", exc)
            raise


        print(f"DEBUG: PDS Image data shape: {pds_img.image.shape}")
        print(f"DEBUG: PDS Image data type: {pds_img.image.dtype}")
        print(f"DEBUG: PDS Image data min/max: {np.min(pds_img.image)}, {np.max(pds_img.image)}")

        # Stage 4: Radiometric IOF Scaling Execution
        try:
            iof_data = calibrate_iof_data(pds_img.image, image_id, distance_au, f_solar=self.f_solar)
        except Exception as e:
            _log_fatal_geometry_missing(f"Radiometric I/F scaling engine processing failed for image id: {image_id}", e)
            raise

        image_shape = iof_data.shape
        if iof_data.ndim != 2:
            raise ValueError(f"Expected 2D image array, got shape {image_shape}")

        pixel_rays_grid = self._pixel_rays(image_shape)
        rays_flat = pixel_rays_grid.reshape(-1, 3)
        self._log_pointing_diagnostics(et, image_shape, pixel_rays_grid, image_id)

        # Per-pixel solid angle depends only on camera geometry (the ray grid above), not
        # on the per-pixel SPICE intercept loop below, so compute it once up front.
        pixel_solid_angle_sr = compute_pixel_solid_angles(pixel_rays_grid).reshape(-1)

        n_pix = rays_flat.shape[0]
        spoints = np.full((n_pix, 3), np.nan, dtype=np.float64)

        phase = np.full(n_pix, np.nan, dtype=np.float64)
        incidence = np.full(n_pix, np.nan, dtype=np.float64)
        emission = np.full(n_pix, np.nan, dtype=np.float64)
        latitude = np.full(n_pix, np.nan, dtype=np.float64)
        longitude = np.full(n_pix, np.nan, dtype=np.float64)
        range_km = np.full(n_pix, np.nan, dtype=np.float64)

        logging.info("Tracing %d rays with SPICE sincpt using method=%s...", n_pix, self.surface_intercept_method)
        for idx in range(n_pix):
            try:
                spoint, _, srfvec = spiceypy.sincpt(
                    self.surface_intercept_method, self.target, et, self.body_fixed_frame,
                    self.aberration_correction, self.observer, self.cam_frame, rays_flat[idx]
                )
            except spiceypy.utils.exceptions.NotFoundError:
                continue
            except spiceypy.support_types.SpiceyError as exc:
                # Structured, policy-driven short-code parsing instead of ad-hoc text queries
                if "DSK" in self.surface_intercept_method.upper():
                    if getattr(exc, "short", "") in ALLOWABLE_DSK_SHORT_CODES:
                        continue
                _log_fatal_geometry_missing(f"sincpt failed for image={image_id} pixel={idx}", exc)
                raise

            spoints[idx] = spoint
            # srfvec is the observer->surface-point vector sincpt already computes
            # internally; its norm is the exact per-pixel range, previously discarded.
            range_km[idx] = float(np.linalg.norm(srfvec))

            try:
                _, lon_rad, lat_rad = spiceypy.reclat(spoint)
                longitude[idx] = np.degrees(float(lon_rad))
                latitude[idx] = np.degrees(float(lat_rad))
            except spiceypy.support_types.SpiceyError:
                pass

            try:
                illumf_result = spiceypy.illumf(
                    self.surface_intercept_method, self.target, "SUN", et,
                    self.body_fixed_frame, self.aberration_correction, self.observer, spoint
                )
                phase[idx] = float(illumf_result[2])
                incidence[idx] = float(illumf_result[3])
                emission[idx] = float(illumf_result[4])
            except spiceypy.support_types.SpiceyError as exc:
                # SYMMETRIC SHADOW/LIMB ROBUSTNESS GRACEFUL BYPASS
                if "DSK" in self.surface_intercept_method.upper():
                    if getattr(exc, "short", "") in ALLOWABLE_DSK_SHORT_CODES:
                        spoints[idx] = np.nan
                        continue
                _log_fatal_geometry_missing(f"illumf failed for image={image_id} pixel={idx}", exc)
                raise

        phase = np.rad2deg(phase)
        incidence = np.rad2deg(incidence)
        emission = np.rad2deg(emission)
        iof_flat = iof_data.reshape(-1)
        yy, xx = np.indices(image_shape)

        pixel_area_km2 = compute_pixel_area_km2(
            pixel_solid_angle_sr, range_km, emission, self.max_emission_for_area_deg
        )
        projected_area_km2 = compute_projected_area_km2(pixel_solid_angle_sr, range_km)

        spoints_finite = np.all(np.isfinite(spoints), axis=1)
        phase_valid = np.isfinite(phase) & (phase >= 0.0) & (phase <= 180.0)
        incidence_valid = np.isfinite(incidence) & (incidence >= 0.0) & (incidence <= 180.0)
        emission_valid = np.isfinite(emission) & (emission >= 0.0) & (emission <= 180.0)
        iof_valid = np.isfinite(iof_flat)

        valid_mask = (spoints_finite & phase_valid & incidence_valid & emission_valid & iof_valid)
        n_valid = int(np.count_nonzero(valid_mask))
        if n_valid == 0:
            raise RuntimeError(f"FATAL GEOMETRY MISSING: zero valid geometry pixels inside bounding intercept for image {image_id}")

        df = pd.DataFrame({
            "image_id": np.full(n_valid, image_id, dtype=object),
            "pixel_x": xx.reshape(-1)[valid_mask].astype(np.int32),
            "pixel_y": yy.reshape(-1)[valid_mask].astype(np.int32),
            "iof": iof_flat[valid_mask].astype(np.float32),
            "incidence": incidence[valid_mask].astype(np.float32),
            "emission": emission[valid_mask].astype(np.float32),
            "phase": phase[valid_mask].astype(np.float32),
            "latitude": latitude[valid_mask].astype(np.float32),
            "longitude": longitude[valid_mask].astype(np.float32),
            # float64, not float32 like the columns above: these exist specifically so
            # the projected area can be recomputed/audited later without re-running
            # SPICE, so precision is kept rather than traded for storage compactness.
            "range_km": range_km[valid_mask].astype(np.float64),
            "pixel_solid_angle_sr": pixel_solid_angle_sr[valid_mask].astype(np.float64),
            "pixel_area_km2": pixel_area_km2[valid_mask].astype(np.float64),
            "projected_area_km2": projected_area_km2[valid_mask].astype(np.float64),
        })

        output_phase_dir = self.output_dir / phase_subdir
        output_phase_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_phase_dir / f"{image_id}_geometry.parquet"
        tmp_path = output_path.with_suffix(".tmp")

        # POSIX ATOMIC TRANSFORMATION
        try:
            df.to_parquet(tmp_path, engine="pyarrow", index=False)
            tmp_path.replace(output_path) 
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            logging.critical("POSIX atomic write transaction failed on disk for path: %s. Exception: %s", output_path, e)
            raise

        logging.info("Saved geometry table: %s (%d rows)", output_path, len(df))
        return df

    def __del__(self):
        """Unload SPICE kernels when the object is destroyed."""
        try:
            spiceypy.kclear()
            logging.info("SPICE kernels have been unloaded.")
        except Exception as e:
            logging.error(f"Error unloading SPICE kernels: {e}")
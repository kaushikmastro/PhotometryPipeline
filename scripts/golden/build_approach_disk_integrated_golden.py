"""Disk-integrated golden layer for approach-phase F1B images, via aperture photometry.

SPICE-SEEDED SEARCH (v3 redesign). Earlier versions searched the WHOLE frame for
"whatever pixel clears a background-noise threshold" and centroided on it. That failed
decisively and was caught by a provenance-based validation (aperture-vs-per-pixel
comparison, PDS label inspection, and visual inspection -- see the decisive-test record
in git history around commit 29c3b00, now superseded): on genuinely faint point-source
frames, the real target signal sits AT the noise floor while the background has fat,
non-Gaussian tails (cosmic rays, hot pixels, background stars) -- a whole-frame search
reliably locks onto the brightest unrelated feature instead. Visually confirmed: the
code centroided on a background star with a visible CCD-blooming streak, 69px from
Vesta's true (SPICE-predicted, per-pixel-geometry-confirmed) position, which happened to
sit near the noise floor and was invisible even on an aggressive stretch. A "perfect"
Spearman phase-trend and tight repeat-frame agreement both survived on this wrong
target, because the star's brightness varies coherently with time too -- a reminder that
trend-based validation without provenance is not validation.

This version predicts the target's expected pixel position and angular size from SPICE
ephemeris FIRST (predict_target_pixel / estimate_target_diameter_px -- one coordinate
transform each, still no per-pixel ray tracing), then restricts BOTH detection/
centroiding (locate_target_in_window) and the aperture radius to that predicted
neighborhood, sized from predicted geometry rather than a blind whole-frame search or
curve-of-growth (the growth-curve approach had its own failure mode: it consistently
plateaued around ~136px regardless of the true target's size, chasing extended PSF/
scattered-light structure rather than the disk edge -- visually confirmed on a well-
detected mid-tier frame where the found centroid WAS correct but the growth-curve
aperture still over-included background well beyond the sharp disk boundary).

Range varies ~300x across the approach dataset (10,700 km to 1.22M km), so Vesta's
apparent size is NOT uniform: it spans from ~4.6 px (genuinely point-like, ~1.22M km) up
to larger than the 1024x1024 frame itself (~10,700 km, essentially a resolved disk-limb
shot). Two regimes, by estimated target diameter (see estimate_target_diameter_px):
  - <= MAX_ADAPTIVE_TARGET_DIAMETER_PX: aperture photometry applies. Search window and
    aperture radius both scale with predicted target size (see predict_target_pixel,
    locate_target_in_window, POINTING_MARGIN_PX, APERTURE_PADDING_FACTOR).
  - above that: fully resolved (in some cases larger than the frame). Aperture
    photometry of any radius is the wrong tool here -- these need the standard per-pixel
    geometry pipeline (GeometryEngine.compute_geometry), same as RC/Survey/HAMO/LAMO.
    Rows are still emitted (photometry_method="skipped_fully_resolved") for
    auditability, with NaN photometry.

Either way, this reuses GeometryEngine's existing camera-FOV and radiometric-calibration
machinery (per-pixel solid angle depends only on the camera FOV, not on a target
intercept; I/F conversion via calibrate_iof_data needs only Sun-target distance) without
ever calling its per-pixel sincpt ray-tracing loop -- avoiding ~1e6 wasted SPICE rays per
frame to find a target occupying a tiny fraction of the pixels, in the tiers where that
loop would be wasteful. (It is NOT wasteful in the fully-resolved tier -- that's exactly
why those frames are routed elsewhere instead of forced through a small aperture.)

VALIDATION HERE MUST BE PROVENANCE-BASED, NOT TREND-BASED: every row records
predicted_row/predicted_col alongside centroid_row/centroid_col and centroid_offset_px,
specifically so "does the measured centroid land near where SPICE says the target is"
can be checked directly, per frame, rather than inferred from whether the aggregate
trend looks physically plausible.

RULE 1/2 (CLAUDE.md): running this for real (609 frames, SPICE + image IO) MUST go
through srun/sbatch -- see scripts/submit/submit_approach_golden.sh.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import spiceypy

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from photometry_etl.etl.geometry_engine import (  # noqa: E402
    GeometryEngine,
    _safe_float,
    calibrate_iof_data,
    compute_pixel_solid_angles,
    compute_projected_area_km2,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SUBPOINT_METHOD = "Intercept: dsk/unprioritized"
ILLUM_METHOD = "DSK/UNPRIORITIZED"

DEFAULT_APERTURE_RADIUS_PX = 8.0
DEFAULT_DETECTION_SIGMA = 5.0
DEFAULT_BG_SIGMA_CLIP = 3.0
DEFAULT_BG_MAX_ITER = 5

VESTA_RADIUS_KM = 262.0
# Tier boundary, in estimated target diameter (px). See module docstring.
MAX_ADAPTIVE_TARGET_DIAMETER_PX = 200.0  # generous headroom above the largest
# aperture-photometry-tier target actually observed (~80 px) while still comfortably
# excluding the fully-resolved tier (~400-1400 px).

# SPICE-seeded search parameters (see module docstring for why the old whole-frame
# search / curve-of-growth were replaced).
POINTING_MARGIN_PX = 30.0  # generous fixed margin added to the predicted target radius
# when sizing the detection/centroiding search window, for OpNav-era ephemeris/pointing
# uncertainty -- not tuned against real pointing-error statistics yet, deliberately
# generous rather than tight.
MIN_WINDOW_RADIUS_PX = 20.0  # floor on the search window radius so a genuinely
# point-like target (predicted radius ~2px) still gets a reasonable neighborhood to
# search, not a search window barely bigger than the pointing margin alone.
APERTURE_PADDING_FACTOR = 1.3  # aperture radius = predicted target radius * this,
# to include near-target PSF wings just outside the sharp disk edge without the
# unbounded over-inclusion the old curve-of-growth showed (converged to ~136px
# regardless of true target size on real data -- see module docstring).

# Retained for its own test coverage / potential future use (e.g. refining
# APERTURE_PADDING_FACTOR against real curves within a bounded, SPICE-seeded window)
# but NO LONGER used as process_frame's primary aperture-sizing method -- see module
# docstring for why blind curve-of-growth failed on real data.
GROWTH_CURVE_RADII_PX = tuple(np.arange(4.0, 140.0, 4.0))
GROWTH_CURVE_PLATEAU_SIGMA = 2.0
GROWTH_CURVE_PLATEAU_CONSECUTIVE = 3

# Same YYDDDHHMMSS filename timestamp pattern proven in .tmp/approach_29_audit.py and
# .tmp/approach_phase_coverage.py. Only used here as a fallback for `campaign` parsing /
# sanity cross-checks -- the actual per-frame ET now comes from the real downloaded .LBL
# (GeometryEngine._extract_observation_et), which is more precise (true SCLK + exposure
# midpoint) now that these frames have labels.
TS_RE = re.compile(r"_(\d{2})(\d{3})(\d{2})(\d{2})(\d{2})F1")
CAMPAIGN_RE = re.compile(r"APPROACH/(\d+_[A-Za-z0-9_]+)/")


# ---------------------------------------------------------------------------
# Pure, unit-testable numpy functions -- no SPICE, no file IO.
# ---------------------------------------------------------------------------


def estimate_background(
    image: np.ndarray,
    sigma_clip: float = DEFAULT_BG_SIGMA_CLIP,
    max_iter: int = DEFAULT_BG_MAX_ITER,
) -> tuple[float, float]:
    """Iteratively sigma-clipped median/std of the whole frame.

    Vesta occupies ~17 of ~1e6 pixels at approach range, so the frame is >99.99%
    background -- a single whole-frame robust statistic is a fair background estimate
    without needing a separate sky annulus. Returns (background_level, background_sigma).
    NaN/non-finite pixels are ignored throughout.

    calibrate_iof_data() clips any negative-noise excursion up to exactly 0.0 (its noise
    floor), which pins a large fraction of a faint approach frame's pixels to that one
    value -- verified at 56% for a representative point-source frame. Once more than half
    the population shares one value, iterative sigma-clipping around the median
    inevitably collapses onto that degenerate cluster and converges to sigma=0 (found via
    the decisive aperture-vs-per-pixel test: a zero sigma makes locate_target's threshold
    degenerate to exactly background_level, so it "detects" and centroids on whatever
    noise pixel happens to dominate the flux-weighted sum, nowhere near the real target).
    When that happens, fall back to a percentile-based spread ((P84-P16)/2, the standard
    robust "half the 68% interval" estimator) over the ORIGINAL unclipped population --
    robust to floor-pinning as long as under 84% of pixels sit at the floor, comfortably
    true here.
    """
    valid = image[np.isfinite(image)]
    if valid.size == 0:
        return float("nan"), float("nan")

    original_valid = valid
    for _ in range(max_iter):
        level = float(np.median(valid))
        sigma = float(np.std(valid))
        if sigma == 0.0:
            break
        keep = np.abs(valid - level) <= sigma_clip * sigma
        if keep.all() or not keep.any():
            break
        valid = valid[keep]

    level = float(np.median(valid))
    sigma = float(np.std(valid))
    if sigma == 0.0:
        p16, p84 = np.percentile(original_valid, [16.0, 84.0])
        sigma = float((p84 - p16) / 2.0)

    return level, sigma


def locate_target(
    image: np.ndarray,
    background_level: float,
    background_sigma: float,
    detection_sigma: float = DEFAULT_DETECTION_SIGMA,
) -> tuple[float, float] | None:
    """Flux-weighted centroid of pixels above background_level + detection_sigma *
    background_sigma. Returns (row, col) in fractional pixel coordinates, or None if no
    pixel clears the detection threshold (target off-frame, or a failed/blank exposure).
    """
    if not np.isfinite(background_sigma) or background_sigma <= 0:
        return None
    threshold = background_level + detection_sigma * background_sigma
    detected = np.isfinite(image) & (image > threshold)
    if not detected.any():
        return None

    rows, cols = np.nonzero(detected)
    weights = np.clip(image[detected] - background_level, 0.0, None)
    if weights.sum() <= 0:
        return float(rows.mean()), float(cols.mean())
    row_c = float(np.sum(rows * weights) / weights.sum())
    col_c = float(np.sum(cols * weights) / weights.sum())
    return row_c, col_c


def locate_target_in_window(
    image: np.ndarray,
    predicted_row: float,
    predicted_col: float,
    window_radius_px: float,
    background_level: float,
    background_sigma: float,
    detection_sigma: float = DEFAULT_DETECTION_SIGMA,
) -> tuple[float, float] | None:
    """Flux-weighted centroid, restricted to a circular window around a SPICE-predicted
    target position -- the fix for locate_target's whole-frame failure mode (see module
    docstring): with a target this faint, a naive sigma threshold can clear on
    thousands of unrelated frame-wide pixels (heavy-tailed real-world noise, cosmic
    rays, background stars), and an unrestricted flux-weighted centroid locks onto
    whichever such feature dominates -- observed concretely as a background star with a
    CCD-blooming streak, 69px from the true target, on real data. Restricting the
    search to a window around where SPICE says the target actually is excludes
    everything outside that window from consideration, independent of how bright it is.

    Otherwise identical to locate_target: same threshold formula, same flux-weighted
    centroid, just masked to `window_radius_px` of (predicted_row, predicted_col) first.
    """
    if not np.isfinite(background_sigma) or background_sigma <= 0:
        return None
    window = aperture_mask(image.shape, (predicted_row, predicted_col), window_radius_px)
    threshold = background_level + detection_sigma * background_sigma
    detected = window & np.isfinite(image) & (image > threshold)
    if not detected.any():
        return None

    rows, cols = np.nonzero(detected)
    weights = np.clip(image[detected] - background_level, 0.0, None)
    if weights.sum() <= 0:
        return float(rows.mean()), float(cols.mean())
    row_c = float(np.sum(rows * weights) / weights.sum())
    col_c = float(np.sum(cols * weights) / weights.sum())
    return row_c, col_c


def aperture_mask(
    image_shape: tuple[int, int], centroid: tuple[float, float], aperture_radius_px: float
) -> np.ndarray:
    """Boolean circular aperture mask of the given radius around centroid (row, col)."""
    ny, nx = image_shape
    rows, cols = np.mgrid[0:ny, 0:nx]
    dist2 = (rows - centroid[0]) ** 2 + (cols - centroid[1]) ** 2
    return dist2 <= aperture_radius_px**2


def aperture_sum(image: np.ndarray, mask: np.ndarray, background_level: float) -> tuple[float, int]:
    """Background-subtracted sum of `image` within `mask`.

    NaN aperture pixels are excluded from both the sum and n_pix (n_pix reflects usable
    pixels only, so aperture_uncertainty's sqrt(n_pix) term stays honest).
    """
    usable = mask & np.isfinite(image)
    n_pix = int(usable.sum())
    if n_pix == 0:
        return float("nan"), 0
    return float(np.sum(image[usable] - background_level)), n_pix


def estimate_target_diameter_px(
    range_km: float, pixel_solid_angle_sr: float, target_radius_km: float = VESTA_RADIUS_KM
) -> float:
    """Rough estimate of the target's apparent diameter in pixels, used only to pick a
    photometry method (see module docstring's three tiers) -- not a science quantity.

    sqrt(pixel_solid_angle_sr) approximates one pixel's angular width (exact for a
    square pixel; the FC2 detector window is close enough to square that this is fine
    for tier selection). target angular diameter = 2 * target_radius_km / range_km.
    """
    if not np.isfinite(range_km) or range_km <= 0 or not np.isfinite(pixel_solid_angle_sr) or pixel_solid_angle_sr <= 0:
        return float("nan")
    pixel_angular_width_rad = np.sqrt(pixel_solid_angle_sr)
    target_angular_diameter_rad = 2.0 * target_radius_km / range_km
    return float(target_angular_diameter_rad / pixel_angular_width_rad)


def curve_of_growth_radius(
    image: np.ndarray,
    centroid: tuple[float, float],
    background_level: float,
    background_sigma: float,
    radii: tuple[float, ...] = GROWTH_CURVE_RADII_PX,
    plateau_sigma: float = GROWTH_CURVE_PLATEAU_SIGMA,
    consecutive: int = GROWTH_CURVE_PLATEAU_CONSECUTIVE,
) -> float:
    """Pick an aperture radius from a curve of growth: cumulative background-subtracted
    flux at each radius in `radii` (ascending), stopping at the smallest radius after
    which the marginal flux gained by `consecutive` further steps is each consistent
    with pure background noise (|delta_flux| <= plateau_sigma * sqrt(delta_n_pix) *
    background_sigma) rather than real additional signal.

    Standard aperture-photometry practice for an extended/partially-resolved source: a
    fixed radius either truncates real flux (too small) or adds pure noise (too big);
    the curve of growth finds where the two trade off. Falls back to the largest radius
    tested if the curve never plateaus within `radii` (the target is likely still
    growing at the cap -- callers should treat that as a maxed-out, non-plateaued
    aperture, not a confident measurement).
    """
    radii_sorted = np.asarray(sorted(radii), dtype=np.float64)
    fluxes = np.empty_like(radii_sorted)
    n_pix = np.empty_like(radii_sorted)
    for i, r in enumerate(radii_sorted):
        mask = aperture_mask(image.shape, centroid, r)
        total, n = aperture_sum(image, mask, background_level)
        fluxes[i] = total
        n_pix[i] = n

    plateau_run = 0
    chosen_idx = len(radii_sorted) - 1
    for i in range(1, len(radii_sorted)):
        d_flux = fluxes[i] - fluxes[i - 1]
        d_pix = max(n_pix[i] - n_pix[i - 1], 1.0)
        if not np.isfinite(d_flux) or not np.isfinite(background_sigma):
            plateau_run = 0
            continue
        noise = np.sqrt(d_pix) * background_sigma
        if noise > 0 and abs(d_flux) <= plateau_sigma * noise:
            plateau_run += 1
            if plateau_run >= consecutive:
                chosen_idx = i - consecutive + 1
                break
        else:
            plateau_run = 0

    return float(radii_sorted[chosen_idx])


def aperture_sensitivity_term(
    image: np.ndarray,
    centroid: tuple[float, float],
    aperture_radius_px: float,
    background_level: float,
    delta_px: float = 1.0,
) -> float:
    """Half the spread between aperture sums at radius-delta_px and radius+delta_px.

    An estimate of how much the photometry would move under a plausible alternative
    aperture choice -- relevant here because the target is only a handful of pixels
    across, so aperture-edge placement is a real error source, not just background
    scatter. Folded into aperture_uncertainty() below.
    """
    shape = image.shape
    inner = aperture_mask(shape, centroid, max(aperture_radius_px - delta_px, 0.5))
    outer = aperture_mask(shape, centroid, aperture_radius_px + delta_px)
    sum_inner, _ = aperture_sum(image, inner, background_level)
    sum_outer, _ = aperture_sum(image, outer, background_level)
    if not (np.isfinite(sum_inner) and np.isfinite(sum_outer)):
        return 0.0
    return abs(sum_outer - sum_inner) / 2.0


def aperture_uncertainty(
    n_pix_aperture: int, background_sigma: float, aperture_sensitivity: float = 0.0
) -> float:
    """Background-noise-dominated aperture photometry uncertainty.

    sqrt(n_pix) * background_sigma (standard aperture-photometry shot/background-noise
    formula) combined in quadrature with the aperture-choice sensitivity term above.
    """
    if n_pix_aperture <= 0 or not np.isfinite(background_sigma):
        return float("nan")
    shot_term = np.sqrt(n_pix_aperture) * background_sigma
    return float(np.sqrt(shot_term**2 + aperture_sensitivity**2))


def campaign_from_file_spec(file_specification_name: str) -> str | None:
    """Extract the PDS campaign subfolder name (e.g. '2011123_OPNAV_001') from an
    approach-phase FILE_SPECIFICATION_NAME."""
    m = CAMPAIGN_RE.search(str(file_specification_name))
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# SPICE/IO-dependent per-frame orchestration
# ---------------------------------------------------------------------------


def predict_target_pixel(
    engine: GeometryEngine, et: float, image_shape: tuple[int, int]
) -> tuple[float, float]:
    """SPICE-predicted (row, col) of the target's center, via nearest-ray lookup against
    the camera's per-pixel ray grid (compute_pixel_rays -- camera-FOV geometry only, NOT
    per-pixel ray tracing against the DSK). Rotates the observer->target J2000 direction
    into the camera frame, then finds which pixel's ray direction has the largest dot
    product with it (i.e. the smallest angular separation). One pxform + one spkpos
    (both already computed elsewhere in process_frame for other purposes -- kept
    self-contained here for testability) + a vectorized argmax over the ray grid that
    process_frame builds anyway for I/F calibration -- no added per-pixel SPICE cost.
    """
    target_vec_j2000, _ = spiceypy.spkpos(
        engine.target, et, "J2000", engine.aberration_correction, engine.observer
    )
    j2000_to_cam = spiceypy.pxform("J2000", engine.cam_frame, et)
    target_vec_cam = np.asarray(j2000_to_cam) @ np.asarray(target_vec_j2000)
    target_vec_cam = target_vec_cam / np.linalg.norm(target_vec_cam)

    pixel_rays_grid = engine._pixel_rays(image_shape)
    dot = np.einsum("ijk,k->ij", pixel_rays_grid, target_vec_cam)
    row, col = np.unravel_index(np.argmax(dot), dot.shape)
    return float(row), float(col)


@dataclass
class ApertureFrameResult:
    image_id: str
    campaign: str | None
    utc: str
    exposure_duration_ms: float
    phase_deg: float
    sub_observer_lat_deg: float
    sub_observer_lon_deg: float
    sub_solar_lat_deg: float
    sub_solar_lon_deg: float
    range_km: float
    estimated_target_diameter_px: float
    predicted_row: float
    predicted_col: float
    photometry_method: str
    centroid_row: float
    centroid_col: float
    centroid_offset_px: float
    aperture_radius_px: float
    n_pix_aperture: int
    background_iof: float
    background_iof_sigma: float
    integrated_flux: float
    integrated_flux_uncertainty: float
    integrated_iof_km2: float
    integrated_iof_km2_uncertainty: float
    detected: bool


def process_frame(
    engine: GeometryEngine,
    image_path: Path,
    campaign: str | None,
    min_aperture_radius_px: float = DEFAULT_APERTURE_RADIUS_PX,
    detection_sigma: float = DEFAULT_DETECTION_SIGMA,
) -> ApertureFrameResult:
    """Aperture-photometer one approach-phase frame, SPICE-seeded (see module
    docstring). SPICE point-geometry calls per frame (no per-pixel ray tracing):
    spkpos x3 (Sun-target, observer-target, and predict_target_pixel's own
    observer-target), pxform once, subpnt/subslr once each, illumf once.
    """
    image_id = image_path.stem
    pds_img = engine._planetaryimage.PDS3Image.open(str(image_path))

    et = engine._extract_observation_et(pds_img.label)
    utc = spiceypy.et2utc(et, "ISOC", 3)
    exposure_duration_ms = _safe_float(pds_img.label, ["EXPOSURE_DURATION"], default=float("nan"))

    sun_pos, _ = spiceypy.spkpos("SUN", et, "J2000", engine.aberration_correction, engine.target)
    distance_au = spiceypy.convrt(float(spiceypy.vnorm(sun_pos)), "KM", "AU")

    iof = calibrate_iof_data(pds_img.image, image_id, distance_au, f_solar=engine.f_solar)

    # Per-pixel solid angle depends only on camera FOV geometry -- no ray tracing.
    pixel_rays_grid = engine._pixel_rays(iof.shape)
    pixel_solid_angle_sr = compute_pixel_solid_angles(pixel_rays_grid)

    # ONE spacecraft-to-target range for the whole frame. Valid because the target
    # subtends ~4-5 px; range variation across that angular extent is negligible
    # (~262 km body radius vs ~1.2e6 km range).
    obs_to_target, _ = spiceypy.spkpos(
        engine.target, et, "J2000", engine.aberration_correction, engine.observer
    )
    range_km = float(spiceypy.vnorm(obs_to_target))

    spoint_obs, _, _ = spiceypy.subpnt(
        SUBPOINT_METHOD, engine.target, et, engine.body_fixed_frame,
        engine.aberration_correction, engine.observer,
    )
    _, lon_obs, lat_obs = spiceypy.reclat(spoint_obs)

    spoint_sun, _, _ = spiceypy.subslr(
        SUBPOINT_METHOD, engine.target, et, engine.body_fixed_frame,
        engine.aberration_correction, engine.observer,
    )
    _, lon_sun, lat_sun = spiceypy.reclat(spoint_sun)

    illumf_result = spiceypy.illumf(
        ILLUM_METHOD, engine.target, "SUN", et, engine.body_fixed_frame,
        engine.aberration_correction, engine.observer, spoint_obs,
    )
    phase_deg = float(np.degrees(float(illumf_result[2])))

    # Tier selection (see module docstring): estimated from geometry alone, before any
    # detection attempt, so a maxed-out/oversized target never gets forced through a
    # method that can't measure it.
    center_solid_angle_sr = float(
        pixel_solid_angle_sr[pixel_solid_angle_sr.shape[0] // 2, pixel_solid_angle_sr.shape[1] // 2]
    )
    estimated_diameter_px = estimate_target_diameter_px(range_km, center_solid_angle_sr)

    # SPICE-predicted pixel position -- reuses the same ray grid built above for I/F
    # calibration, so this is one argmax over an already-computed array, not new SPICE
    # ray-tracing cost.
    predicted_row, predicted_col = predict_target_pixel(engine, et, iof.shape)

    common = dict(
        image_id=image_id, campaign=campaign, utc=utc,
        exposure_duration_ms=exposure_duration_ms, phase_deg=phase_deg,
        sub_observer_lat_deg=float(np.degrees(lat_obs)),
        sub_observer_lon_deg=float(np.degrees(lon_obs)),
        sub_solar_lat_deg=float(np.degrees(lat_sun)),
        sub_solar_lon_deg=float(np.degrees(lon_sun)),
        range_km=range_km, estimated_target_diameter_px=estimated_diameter_px,
        predicted_row=predicted_row, predicted_col=predicted_col,
    )
    not_measured = dict(
        centroid_row=float("nan"), centroid_col=float("nan"), centroid_offset_px=float("nan"),
        aperture_radius_px=float("nan"),
        n_pix_aperture=0, background_iof=float("nan"), background_iof_sigma=float("nan"),
        integrated_flux=float("nan"), integrated_flux_uncertainty=float("nan"),
        integrated_iof_km2=float("nan"), integrated_iof_km2_uncertainty=float("nan"),
        detected=False,
    )

    if not np.isfinite(estimated_diameter_px) or estimated_diameter_px > MAX_ADAPTIVE_TARGET_DIAMETER_PX:
        # Fully resolved (sometimes larger than the frame) -- aperture photometry of any
        # radius is the wrong tool. Leave it out of this golden layer; route it through
        # GeometryEngine.compute_geometry instead.
        return ApertureFrameResult(**common, photometry_method="skipped_fully_resolved", **not_measured)

    predicted_radius_px = estimated_diameter_px / 2.0
    window_radius_px = max(predicted_radius_px + POINTING_MARGIN_PX, MIN_WINDOW_RADIUS_PX)

    background_level, background_sigma = estimate_background(iof)
    centroid = locate_target_in_window(
        iof, predicted_row, predicted_col, window_radius_px,
        background_level, background_sigma, detection_sigma,
    )

    if centroid is None:
        return ApertureFrameResult(
            **common, photometry_method="no_detection",
            **{**not_measured, "background_iof": background_level, "background_iof_sigma": background_sigma},
        )

    centroid_offset_px = float(np.hypot(centroid[0] - predicted_row, centroid[1] - predicted_col))
    # Aperture radius scales with predicted target size (not a blind curve-of-growth --
    # see module docstring), bounded below by min_aperture_radius_px (the point-source
    # floor) and above by the search window itself.
    radius_px = min(
        max(predicted_radius_px * APERTURE_PADDING_FACTOR, min_aperture_radius_px),
        window_radius_px,
    )

    mask = aperture_mask(iof.shape, centroid, radius_px)
    integrated_flux, n_pix = aperture_sum(iof, mask, background_level)
    sensitivity = aperture_sensitivity_term(iof, centroid, radius_px, background_level)
    integrated_flux_unc = aperture_uncertainty(n_pix, background_sigma, sensitivity)

    # Area-weighted, disk-integrated-photometry-convention quantity (projected area,
    # per CLAUDE.md's "Disk-Integrated Photometry" decision), using the single
    # frame-level range_km in place of a per-pixel range.
    projected_area_km2 = compute_projected_area_km2(
        pixel_solid_angle_sr, np.full_like(pixel_solid_angle_sr, range_km)
    )
    usable = mask & np.isfinite(iof)
    integrated_iof_km2 = float(np.sum((iof[usable] - background_level) * projected_area_km2[usable]))
    rel_unc = (
        integrated_flux_unc / integrated_flux
        if np.isfinite(integrated_flux) and integrated_flux != 0
        else float("nan")
    )
    integrated_iof_km2_unc = (
        abs(integrated_iof_km2 * rel_unc) if np.isfinite(rel_unc) else float("nan")
    )

    return ApertureFrameResult(
        **common, photometry_method="windowed_aperture",
        centroid_row=centroid[0], centroid_col=centroid[1], centroid_offset_px=centroid_offset_px,
        aperture_radius_px=radius_px, n_pix_aperture=n_pix,
        background_iof=background_level, background_iof_sigma=background_sigma,
        integrated_flux=integrated_flux, integrated_flux_uncertainty=integrated_flux_unc,
        integrated_iof_km2=integrated_iof_km2, integrated_iof_km2_uncertainty=integrated_iof_km2_unc,
        detected=True,
    )


def build_approach_golden(
    manifest_csv: str,
    data_root: str,
    output_path: str,
    metakernel_path: str | None = None,
    min_aperture_radius_px: float = DEFAULT_APERTURE_RADIUS_PX,
    detection_sigma: float = DEFAULT_DETECTION_SIGMA,
) -> pd.DataFrame:
    """Build the approach-phase disk-integrated golden layer: one row per F1B approach
    frame, via aperture photometry (see module docstring)."""
    manifest = pd.read_csv(manifest_csv)
    manifest["filter_letter"] = manifest["image_filename"].str.extract(r"F1([A-Z])")
    rows = manifest[
        (manifest["phase_subdir"] == "approach")
        & (manifest["filter_letter"] == "B")
        & (manifest["file_type"] == "IMG")
    ].copy()
    rows["campaign"] = rows["FILE_SPECIFICATION_NAME"].map(campaign_from_file_spec)
    logging.info("Processing %d F1B approach frames", len(rows))

    data_root_path = Path(data_root)
    engine = GeometryEngine(
        data_root=str(data_root_path),
        metakernel_path=metakernel_path or str(data_root_path / "spice_kernels" / "dawn_dynamic.tm"),
        surface_intercept_method="DSK/UNPRIORITIZED",
        output_subdir="geometry/approach_aperture_photometry",
    )

    image_dir = data_root_path / "calibrated_raw_images" / "approach"
    results: list[ApertureFrameResult] = []
    for _, row in rows.iterrows():
        image_path = image_dir / row["image_filename"]
        if not image_path.exists():
            logging.warning("Missing on disk, skipping: %s", image_path)
            continue
        try:
            result = process_frame(
                engine, image_path, row["campaign"],
                min_aperture_radius_px=min_aperture_radius_px, detection_sigma=detection_sigma,
            )
        except Exception:
            logging.exception("Failed to process %s", image_path)
            continue
        results.append(result)

    df = pd.DataFrame([asdict(r) for r in results])
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path_obj, index=False)
    logging.info("Wrote %d frame rows to %s (%d detected)", len(df), output_path_obj, int(df["detected"].sum()) if len(df) else 0)
    return df


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    build_approach_golden(
        manifest_csv=str(project_root / "configs" / "survey_manifest.csv"),
        data_root="/scratch/kaushim07/vesta_data",
        output_path="/scratch/kaushim07/vesta_data/golden/approach_disk_integrated_f1b.parquet",
    )


if __name__ == "__main__":
    main()

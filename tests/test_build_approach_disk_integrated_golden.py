from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "golden" / "build_approach_disk_integrated_golden.py"

_spec = importlib.util.spec_from_file_location("build_approach_disk_integrated_golden", SCRIPT_PATH)
_module = importlib.util.module_from_spec(_spec)
# Register before exec: the module's `from __future__ import annotations` + dataclass
# field types need `sys.modules[cls.__module__]` to resolve at class-creation time.
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)

estimate_background = _module.estimate_background
locate_target = _module.locate_target
locate_target_in_window = _module.locate_target_in_window
aperture_mask = _module.aperture_mask
aperture_sum = _module.aperture_sum
aperture_sensitivity_term = _module.aperture_sensitivity_term
aperture_uncertainty = _module.aperture_uncertainty
campaign_from_file_spec = _module.campaign_from_file_spec
estimate_target_diameter_px = _module.estimate_target_diameter_px
curve_of_growth_radius = _module.curve_of_growth_radius
MAX_ADAPTIVE_TARGET_DIAMETER_PX = _module.MAX_ADAPTIVE_TARGET_DIAMETER_PX
POINTING_MARGIN_PX = _module.POINTING_MARGIN_PX
MIN_WINDOW_RADIUS_PX = _module.MIN_WINDOW_RADIUS_PX
APERTURE_PADDING_FACTOR = _module.APERTURE_PADDING_FACTOR


def _synthetic_frame(
    shape=(64, 64), background=0.0005, noise_sigma=0.00005, target_row=32.0, target_col=32.0,
    target_peak=0.05, target_sigma_px=1.5, seed=0,
):
    """A small Gaussian-blob-on-background frame standing in for a barely-resolved
    approach frame (real frames are 1024x1024 with ~17 on-target pixels; this keeps
    tests fast while preserving the same background::target size ratio character)."""
    rng = np.random.default_rng(seed)
    image = background + rng.normal(0.0, noise_sigma, size=shape).astype(np.float64)
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    blob = target_peak * np.exp(
        -(((rows - target_row) ** 2 + (cols - target_col) ** 2) / (2 * target_sigma_px**2))
    )
    return image + blob


# ---------------------------------------------------------------------------
# estimate_background
# ---------------------------------------------------------------------------


def test_estimate_background_recovers_known_level_and_sigma_ignoring_bright_blob():
    image = _synthetic_frame(background=0.001, noise_sigma=0.0001, target_peak=0.05)
    level, sigma = estimate_background(image)
    assert level == pytest.approx(0.001, abs=0.0005)
    assert sigma == pytest.approx(0.0001, rel=0.5)


def test_estimate_background_all_nan_returns_nan():
    image = np.full((8, 8), np.nan)
    level, sigma = estimate_background(image)
    assert np.isnan(level)
    assert np.isnan(sigma)


def test_estimate_background_ignores_nan_pixels():
    image = np.full((16, 16), 0.001)
    image[0, 0] = np.nan
    level, sigma = estimate_background(image)
    assert level == pytest.approx(0.001)
    assert sigma == pytest.approx(0.0)


def test_estimate_background_falls_back_to_percentile_spread_when_floor_pinned():
    # Reproduces the real bug found via the decisive aperture-vs-per-pixel test:
    # calibrate_iof_data clips negative-noise excursions to exactly 0.0, which pins
    # >50% of a faint approach frame's pixels to that one value (measured 56% on a real
    # frame). Iterative sigma-clipping around the median then collapses onto that
    # degenerate cluster and converges to sigma=0 -- which made locate_target's
    # threshold degenerate to exactly background_level, "detecting" and centroiding on
    # whatever noise pixel summed highest, nowhere near the real ~17-pixel target.
    rng = np.random.default_rng(3)
    image = rng.normal(0.0, 1e-7, size=(200, 200))
    image[image < 0] = 0.0  # the same floor-clip calibrate_iof_data applies
    # ~50% in expectation for a symmetric distribution clipped at its median; a lenient
    # >0.4 avoids the assertion being a coin-flip around exactly 0.5 while still
    # confirming a real degenerate-majority precondition (real frames measured at 56%).
    assert (image == 0.0).mean() > 0.4

    level, sigma = estimate_background(image)
    assert level == pytest.approx(0.0, abs=1e-12)
    # The old code returned sigma=0.0 here (the bug); the fix must recover a small but
    # real, positive noise scale from the percentile fallback.
    assert sigma > 0.0
    assert sigma < 1e-5  # sane order of magnitude for this noise level, not a fluke


def test_estimate_background_floor_pinned_still_detects_real_faint_target():
    # End-to-end regression for the bug: with the fix, a faint compact source should
    # still be found correctly even when the background is floor-pinned, instead of
    # locate_target degenerating to "detect everywhere."
    rng = np.random.default_rng(4)
    image = rng.normal(0.0, 1e-7, size=(200, 200))
    image[image < 0] = 0.0
    image[100, 100] = 0.01  # a real, unambiguous source many sigma above the noise
    level, sigma = estimate_background(image)
    centroid = locate_target(image, level, sigma, detection_sigma=5.0)
    assert centroid is not None
    assert centroid[0] == pytest.approx(100.0, abs=1.0)
    assert centroid[1] == pytest.approx(100.0, abs=1.0)


# ---------------------------------------------------------------------------
# locate_target
# ---------------------------------------------------------------------------


def test_locate_target_finds_known_centroid():
    image = _synthetic_frame(target_row=20.0, target_col=40.0, noise_sigma=1e-6)
    level, sigma = estimate_background(image)
    centroid = locate_target(image, level, sigma, detection_sigma=5.0)
    assert centroid is not None
    row, col = centroid
    assert row == pytest.approx(20.0, abs=0.5)
    assert col == pytest.approx(40.0, abs=0.5)


def test_locate_target_returns_none_when_nothing_above_threshold():
    image = np.random.default_rng(1).normal(0.001, 0.0001, size=(32, 32))
    level, sigma = estimate_background(image)
    centroid = locate_target(image, level, sigma, detection_sigma=50.0)
    assert centroid is None


def test_locate_target_returns_none_for_non_finite_background_sigma():
    image = _synthetic_frame()
    centroid = locate_target(image, background_level=0.0, background_sigma=float("nan"))
    assert centroid is None


# ---------------------------------------------------------------------------
# locate_target_in_window -- regression coverage for the real bug (see module
# docstring): a whole-frame search locks onto a bright unrelated feature instead of a
# faint real target near the SPICE-predicted position. locate_target (unwindowed)
# reproduces the failure; locate_target_in_window is the fix.
# ---------------------------------------------------------------------------


def _frame_with_bright_decoy_and_faint_true_target(
    shape=(200, 200), background=0.0, noise_sigma=1e-8,
    decoy_row=100.0, decoy_col=170.0, decoy_peak=0.05,
    true_row=100.0, true_col=100.0, true_peak=5e-7, seed=7,
):
    """A bright, unrelated 'decoy' feature (standing in for the real background star
    with a CCD-blooming streak found on real data) far from a much fainter 'true
    target' -- the exact shape of the bug this function fixes."""
    rng = np.random.default_rng(seed)
    image = background + rng.normal(0.0, noise_sigma, size=shape)
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]]
    decoy = decoy_peak * np.exp(-(((rows - decoy_row) ** 2 + (cols - decoy_col) ** 2) / (2 * 3.0**2)))
    true_target = true_peak * np.exp(
        -(((rows - true_row) ** 2 + (cols - true_col) ** 2) / (2 * 1.5**2))
    )
    return image + decoy + true_target


def test_locate_target_unwindowed_reproduces_the_bug_locking_onto_the_decoy():
    image = _frame_with_bright_decoy_and_faint_true_target()
    level, sigma = estimate_background(image)
    centroid = locate_target(image, level, sigma, detection_sigma=5.0)
    assert centroid is not None
    # Whole-frame search finds the bright decoy, not the faint true target 70px away.
    assert centroid[1] == pytest.approx(170.0, abs=2.0)


def test_locate_target_in_window_excludes_the_decoy_and_finds_the_true_target():
    image = _frame_with_bright_decoy_and_faint_true_target()
    level, sigma = estimate_background(image)
    # Window centered on the true target's (SPICE-predicted) position, small enough to
    # exclude the decoy at col=170 (70px away).
    centroid = locate_target_in_window(
        image, predicted_row=100.0, predicted_col=100.0, window_radius_px=30.0,
        background_level=level, background_sigma=sigma, detection_sigma=3.0,
    )
    assert centroid is not None
    assert centroid[0] == pytest.approx(100.0, abs=2.0)
    assert centroid[1] == pytest.approx(100.0, abs=2.0)


def test_locate_target_in_window_returns_none_when_window_has_no_signal():
    image = _synthetic_frame(target_row=10.0, target_col=10.0)  # real target far outside window
    level, sigma = estimate_background(image)
    centroid = locate_target_in_window(
        image, predicted_row=50.0, predicted_col=50.0, window_radius_px=10.0,
        background_level=level, background_sigma=sigma, detection_sigma=5.0,
    )
    assert centroid is None


def test_locate_target_in_window_returns_none_for_non_finite_background_sigma():
    image = _synthetic_frame()
    centroid = locate_target_in_window(
        image, predicted_row=32.0, predicted_col=32.0, window_radius_px=20.0,
        background_level=0.0, background_sigma=float("nan"),
    )
    assert centroid is None


# ---------------------------------------------------------------------------
# aperture_mask / aperture_sum
# ---------------------------------------------------------------------------


def test_aperture_mask_is_circular_and_centered():
    mask = aperture_mask((21, 21), centroid=(10.0, 10.0), aperture_radius_px=3.0)
    assert mask[10, 10]
    assert mask[10, 12]
    assert not mask[10, 14]
    # roughly pi*r^2 pixels, generous tolerance for pixelation
    assert abs(mask.sum() - np.pi * 9) < 6


def test_aperture_sum_subtracts_background_and_counts_pixels():
    image = np.full((10, 10), 5.0)
    mask = aperture_mask((10, 10), (5.0, 5.0), 2.0)
    total, n_pix = aperture_sum(image, mask, background_level=2.0)
    assert n_pix == int(mask.sum())
    assert total == pytest.approx(3.0 * n_pix)


def test_aperture_sum_excludes_nan_pixels_from_sum_and_count():
    image = np.full((10, 10), 5.0)
    mask = aperture_mask((10, 10), (5.0, 5.0), 2.0)
    image[5, 5] = np.nan
    total, n_pix = aperture_sum(image, mask, background_level=0.0)
    assert n_pix == int(mask.sum()) - 1
    assert np.isfinite(total)


def test_aperture_sum_all_nan_returns_nan_and_zero_pixels():
    image = np.full((10, 10), np.nan)
    mask = aperture_mask((10, 10), (5.0, 5.0), 2.0)
    total, n_pix = aperture_sum(image, mask, background_level=0.0)
    assert np.isnan(total)
    assert n_pix == 0


# ---------------------------------------------------------------------------
# aperture_sensitivity_term / aperture_uncertainty
# ---------------------------------------------------------------------------


def test_aperture_sensitivity_term_zero_for_flat_background_only_frame():
    image = np.full((20, 20), 1.0)
    term = aperture_sensitivity_term(image, (10.0, 10.0), aperture_radius_px=5.0, background_level=1.0)
    assert term == pytest.approx(0.0, abs=1e-9)


def test_aperture_sensitivity_term_positive_for_a_real_source():
    image = _synthetic_frame(shape=(40, 40), target_row=20.0, target_col=20.0, noise_sigma=1e-9)
    level, _ = estimate_background(image)
    term = aperture_sensitivity_term(image, (20.0, 20.0), aperture_radius_px=5.0, background_level=level)
    assert term > 0.0


def test_aperture_uncertainty_matches_sqrt_n_shot_noise_formula():
    unc = aperture_uncertainty(n_pix_aperture=100, background_sigma=0.01, aperture_sensitivity=0.0)
    assert unc == pytest.approx(np.sqrt(100) * 0.01)


def test_aperture_uncertainty_combines_shot_and_sensitivity_in_quadrature():
    shot = np.sqrt(25) * 0.02
    unc = aperture_uncertainty(n_pix_aperture=25, background_sigma=0.02, aperture_sensitivity=0.1)
    assert unc == pytest.approx(np.sqrt(shot**2 + 0.1**2))


def test_aperture_uncertainty_nan_for_zero_pixels():
    unc = aperture_uncertainty(n_pix_aperture=0, background_sigma=0.01)
    assert np.isnan(unc)


# ---------------------------------------------------------------------------
# campaign_from_file_spec
# ---------------------------------------------------------------------------


def test_campaign_from_file_spec_extracts_opnav_campaign():
    spec = "/DATA/IMG/2011123_APPROACH/2011123_OPNAV_001/FC21B0001898_11123133516F1B.IMG"
    assert campaign_from_file_spec(spec) == "2011123_OPNAV_001"


def test_campaign_from_file_spec_extracts_mosaic_campaign():
    spec = "/DATA/IMG/2011123_APPROACH/2011218_C0_EQUATORIAL_MOSAIC/FC21B0003910_11218125522F1D.IMG"
    assert campaign_from_file_spec(spec) == "2011218_C0_EQUATORIAL_MOSAIC"


def test_campaign_from_file_spec_returns_none_when_no_approach_segment_present():
    # This extractor is only ever called on rows already filtered to
    # phase_subdir == "approach" by build_approach_golden -- it doesn't itself
    # distinguish RC-under-*_APPROACH from true approach imaging (that's
    # DataManager._phase_from_file_spec's job upstream). It simply has nothing to
    # extract when the path contains no "APPROACH/<campaign>/" segment at all.
    spec = "/DATA/IMG/2011246_SURVEY/FC21B0004000_11246000000F1B.IMG"
    assert campaign_from_file_spec(spec) is None


# ---------------------------------------------------------------------------
# estimate_target_diameter_px
# ---------------------------------------------------------------------------


def test_estimate_target_diameter_px_matches_far_campaign_scale():
    # 2011123_OPNAV_001: range ~1.2177e6 km, verified target diameter ~4.6 px
    # (see Stage-1/2 report). FC2's per-pixel angular width from spiceypy.getfov is
    # ~19.2 arcsec, so pixel_solid_angle_sr ~ (19.2 arcsec)^2 in steradians.
    arcsec_to_rad = np.pi / (180.0 * 3600.0)
    pixel_angular_width_rad = 19.2 * arcsec_to_rad
    pixel_solid_angle_sr = pixel_angular_width_rad**2
    diameter_px = estimate_target_diameter_px(1.2177e6, pixel_solid_angle_sr)
    assert diameter_px == pytest.approx(4.6, rel=0.1)


def test_estimate_target_diameter_px_scales_inversely_with_range():
    pixel_solid_angle_sr = (19.2 * np.pi / (180.0 * 3600.0)) ** 2
    near = estimate_target_diameter_px(1000.0, pixel_solid_angle_sr)
    far = estimate_target_diameter_px(10000.0, pixel_solid_angle_sr)
    assert near == pytest.approx(far * 10.0, rel=1e-6)


def test_estimate_target_diameter_px_nan_for_invalid_inputs():
    assert np.isnan(estimate_target_diameter_px(float("nan"), 1e-8))
    assert np.isnan(estimate_target_diameter_px(1000.0, 0.0))
    assert np.isnan(estimate_target_diameter_px(-1.0, 1e-8))


def test_tier_boundary_matches_observed_dataset_scale():
    # The observed aperture-photometry-tier target peaks ~80 px diameter;
    # MAX_ADAPTIVE_TARGET_DIAMETER_PX must clear that with real headroom while staying
    # well under the observed fully-resolved tier's ~400 px floor.
    assert 80.0 < MAX_ADAPTIVE_TARGET_DIAMETER_PX < 400.0


# ---------------------------------------------------------------------------
# curve_of_growth_radius
# ---------------------------------------------------------------------------


def test_curve_of_growth_radius_plateaus_near_a_compact_sources_true_size():
    image = _synthetic_frame(
        shape=(200, 200), background=0.001, noise_sigma=1e-5,
        target_row=100.0, target_col=100.0, target_peak=0.5, target_sigma_px=6.0,
    )
    level, sigma = estimate_background(image)
    radius = curve_of_growth_radius(image, (100.0, 100.0), level, sigma)
    # A Gaussian blob with sigma=6px has ~99% of its flux within ~3*sigma=18px;
    # the curve of growth should settle well short of the radii array's 140px cap.
    assert 12.0 <= radius <= 40.0


def test_curve_of_growth_radius_falls_back_to_largest_radius_when_never_plateaus():
    # A source that fills the whole test frame never lets the curve flatten within
    # the radii sampled here -- falls back to the largest one tested, not a false
    # plateau partway through.
    image = np.full((60, 60), 0.5)
    radii = (4.0, 8.0, 12.0, 16.0, 20.0)
    radius = curve_of_growth_radius(image, (30.0, 30.0), background_level=0.0,
                                     background_sigma=0.0001, radii=radii)
    assert radius == max(radii)


def test_curve_of_growth_radius_small_for_pure_background_frame():
    image = np.random.default_rng(2).normal(0.001, 0.0001, size=(100, 100))
    level, sigma = estimate_background(image)
    radius = curve_of_growth_radius(image, (50.0, 50.0), level, sigma)
    # No real source: the curve should plateau almost immediately, at or near the
    # smallest radius sampled.
    assert radius <= 12.0

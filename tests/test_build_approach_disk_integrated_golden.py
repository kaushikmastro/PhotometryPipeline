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
aperture_mask = _module.aperture_mask
aperture_sum = _module.aperture_sum
aperture_sensitivity_term = _module.aperture_sensitivity_term
aperture_uncertainty = _module.aperture_uncertainty
campaign_from_file_spec = _module.campaign_from_file_spec


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

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from photometry.aggregation.disk_integrate import (  # noqa: E402
    integrate_modeled,
    integrate_observed,
)
from photometry.core.types import GeometryBatch  # noqa: E402
from photometry.models.baselines import (  # noqa: E402
    LambertianModel,
    LommelSeeligerModel,
    MinnaertModel,
)

# Each case: (model class, parameters). Used for the model-agnostic contract test — to
# cover a new model (e.g. AkimovModel) later, add one more pytest.param here.
MODEL_CASES = [
    pytest.param(LambertianModel, {"albedo": 0.6}, id="lambertian"),
    pytest.param(LommelSeeligerModel, {"w": 0.5}, id="lommel-seeliger"),
    pytest.param(MinnaertModel, {"albedo": 0.5, "k": 0.6}, id="minnaert"),
]


def _lambert_phase_function(alpha_rad: np.ndarray) -> np.ndarray:
    """Phi_L(alpha) = (1/pi)*(sin(alpha) + (pi-alpha)*cos(alpha)), normalized to
    Phi_L(0)=1. The classical closed form for a Lambertian sphere's disk-integrated
    brightness (e.g. Russell 1916); verified numerically against independent grid
    quadrature during development (see conversation record / CLAUDE.md), not merely
    assumed."""
    return (1.0 / np.pi) * (np.sin(alpha_rad) + (np.pi - alpha_rad) * np.cos(alpha_rad))


def _sphere_disk_geometry(
    alpha_rad: float, n_theta: int = 300, n_phi: int = 300, radius_km: float = 100.0
):
    """Synthetic sphere sampled on a (theta, phi) grid, sun and observer directions
    separated by phase angle alpha_rad (both distant, so alpha is constant over the
    body -- the standard disk-integrated-phase-curve simplification). Returns a
    GeometryBatch plus BOTH area weightings for the same grid, so a test can compare
    them directly:
      - projected_area_km2: mu * R^2 * dOmega  (correct disk-integration weight)
      - true_area_km2:            R^2 * dOmega  (mapping/resolution weight, wrong one
        for disk integration -- included so tests can demonstrate the difference)

    Only pixels with mu0>0 and mu>0 (illuminated AND visible) get a finite incidence/
    emission; everything else is NaN, matching how a real geometry table only carries
    rows for on-body, illuminated, visible pixels.
    """
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]
    TH, PH = np.meshgrid(theta, phi, indexing="ij")

    sin_th = np.sin(TH)
    z = np.cos(TH)
    x = sin_th * np.cos(PH)

    s = np.sin(alpha_rad / 2.0)
    c = np.cos(alpha_rad / 2.0)
    mu0 = x * s + z * c
    mu = -x * s + z * c

    lit_and_visible = (mu0 > 0) & (mu > 0)

    d_omega = sin_th * dtheta * dphi
    projected_area_km2 = np.where(lit_and_visible, mu * radius_km**2 * d_omega, np.nan)
    true_area_km2 = np.where(lit_and_visible, radius_km**2 * d_omega, np.nan)

    incidence = np.where(lit_and_visible, np.arccos(np.clip(mu0, -1.0, 1.0)), np.nan)
    emission = np.where(lit_and_visible, np.arccos(np.clip(mu, -1.0, 1.0)), np.nan)
    phase = np.full_like(incidence, alpha_rad)

    geometry = GeometryBatch(
        incidence=incidence.reshape(-1),
        emission=emission.reshape(-1),
        phase=phase.reshape(-1),
    )
    return geometry, projected_area_km2.reshape(-1), true_area_km2.reshape(-1)


# ---------------------------------------------------------------------------
# Load-bearing correctness test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha_deg", [0.0, 15.0, 30.0, 60.0, 90.0, 120.0, 150.0])
def test_lambertian_sphere_matches_phase_function_closed_form(alpha_deg: float) -> None:
    """The load-bearing correctness test: if this fails, nothing downstream (including
    the physics decision to weight by projected_area_km2, not pixel_area_km2) is
    trustworthy. See disk_integrate.py's module docstring for the full reasoning."""
    albedo = 0.6
    radius_km = 100.0
    alpha_rad = np.deg2rad(max(alpha_deg, 1e-6))  # avoid the exact-0 boundary singularity

    geometry, projected_area_km2, _ = _sphere_disk_geometry(alpha_rad, radius_km=radius_km)
    model = LambertianModel()
    model.parameters["albedo"] = albedo

    result = integrate_modeled(model, geometry, projected_area_km2)

    expected = albedo * radius_km**2 * (2.0 / 3.0) * _lambert_phase_function(alpha_rad)

    # Grid quadrature error at this resolution is on the order of ~1%, characterized the
    # same way as the geometry_engine solid-angle grid test; a real bug in the
    # aggregation logic (wrong weight, wrong sign, off-by-something) would be off by a
    # large factor, not a percent.
    assert result.integrated_value == pytest.approx(expected, rel=0.02)


def test_true_surface_area_weighting_does_not_match_phase_function() -> None:
    """Negative-result companion to the test above: weighting by TRUE surface area
    (what pixel_area_km2 would give) does NOT reproduce Phi_L(alpha) -- pinned here so a
    future refactor can't silently swap the integration weight back to pixel_area_km2."""
    albedo = 0.6
    radius_km = 100.0
    alpha_rad = np.deg2rad(90.0)

    geometry, _, true_area_km2 = _sphere_disk_geometry(alpha_rad, radius_km=radius_km)
    model = LambertianModel()
    model.parameters["albedo"] = albedo

    result = integrate_modeled(model, geometry, true_area_km2)

    expected_correct = albedo * radius_km**2 * (2.0 / 3.0) * _lambert_phase_function(alpha_rad)
    relative_diff = abs(result.integrated_value - expected_correct) / expected_correct

    # Measured ~57% at alpha=90 deg during development; assert it's grossly wrong (>20%),
    # not pinning the exact percentage so the test doesn't become a second copy of the
    # closed-form derivation.
    assert relative_diff > 0.20, (
        f"true-surface-area weighting should NOT reproduce the Lambert phase function "
        f"(got {result.integrated_value:.4f} vs correct {expected_correct:.4f}, "
        f"{relative_diff:.1%} off) -- if this now passes, projected_area_km2 and "
        f"pixel_area_km2 may have been swapped somewhere"
    )


# ---------------------------------------------------------------------------
# Area-weighting vs equal-weighting
# ---------------------------------------------------------------------------


def test_area_weighting_differs_from_equal_weighting() -> None:
    """This is the test that would have caught the original gap: constructs a case
    with deliberately non-uniform pixel footprints (near-limb pixels are heavily
    foreshortened relative to near-center pixels) and confirms area-weighted and
    equal-weighted aggregation give measurably different answers, with the
    area-weighted one matching the known-correct result."""
    albedo = 0.6
    radius_km = 100.0
    alpha_rad = np.deg2rad(60.0)

    geometry, projected_area_km2, _ = _sphere_disk_geometry(alpha_rad, radius_km=radius_km)
    model = LambertianModel()
    model.parameters["albedo"] = albedo

    area_weighted = integrate_modeled(model, geometry, projected_area_km2)

    # Equal-weighting = the bug being fixed: every included pixel gets the SAME area
    # (mean area), i.e. area-weighted-sum degenerates to (mean_area * sum(reflectance)).
    included = np.isfinite(projected_area_km2)
    equal_area = np.where(included, np.nanmean(projected_area_km2), np.nan)
    equal_weighted = integrate_modeled(model, geometry, equal_area)

    expected = albedo * radius_km**2 * (2.0 / 3.0) * _lambert_phase_function(alpha_rad)

    relative_gap = abs(area_weighted.integrated_value - equal_weighted.integrated_value) / expected
    assert relative_gap > 0.05, "equal- and area-weighting should measurably differ on this limb-heavy geometry"
    assert area_weighted.integrated_value == pytest.approx(expected, rel=0.02)


# ---------------------------------------------------------------------------
# Model-agnostic contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_cls, params", MODEL_CASES)
def test_integrate_modeled_matches_manual_weighted_sum(model_cls, params) -> None:
    """integrate_modeled must be pure aggregation on top of model.reflectance() -- no
    model-specific branching -- so it should exactly match hand-computing the same
    weighted sum from model.reflectance() directly, for any BasePhotometricModel."""
    rng = np.random.default_rng(5)
    n = 50
    incidence = rng.uniform(np.deg2rad(5.0), np.deg2rad(80.0), size=n)
    emission = rng.uniform(np.deg2rad(5.0), np.deg2rad(80.0), size=n)
    phase = np.array(
        [rng.uniform(abs(i - e) + 1e-3, i + e) for i, e in zip(incidence, emission, strict=True)]
    )
    geometry = GeometryBatch(incidence=incidence, emission=emission, phase=phase)
    areas = rng.uniform(1.0, 5.0, size=n)

    model = model_cls()
    model.parameters.update(params)

    result = integrate_modeled(model, geometry, areas)
    manual = float(np.sum(np.asarray(model.reflectance(geometry)) * areas))

    assert result.integrated_value == pytest.approx(manual, rel=1e-10)
    assert result.total_area_km2 == pytest.approx(float(areas.sum()), rel=1e-10)
    assert result.n_pixels_included == n
    assert result.n_pixels_excluded_disk_selection == 0
    assert result.n_pixels_excluded_nan_area == 0


def test_integrate_observed_matches_manual_weighted_sum() -> None:
    rng = np.random.default_rng(6)
    n = 40
    incidence = rng.uniform(np.deg2rad(5.0), np.deg2rad(80.0), size=n)
    emission = rng.uniform(np.deg2rad(5.0), np.deg2rad(80.0), size=n)
    phase = np.array(
        [rng.uniform(abs(i - e) + 1e-3, i + e) for i, e in zip(incidence, emission, strict=True)]
    )
    geometry = GeometryBatch(incidence=incidence, emission=emission, phase=phase)
    observed = rng.uniform(0.05, 0.3, size=n)
    areas = rng.uniform(1.0, 5.0, size=n)

    result = integrate_observed(observed, geometry, areas)

    assert result.integrated_value == pytest.approx(float(np.sum(observed * areas)), rel=1e-10)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_disk_selection_gives_nan() -> None:
    """Every pixel fails the incidence/emission cutoffs."""
    n = 10
    geometry = GeometryBatch(
        incidence=np.full(n, np.deg2rad(85.0)),
        emission=np.full(n, np.deg2rad(85.0)),
        phase=np.zeros(n),
    )
    observed = np.full(n, 0.1)
    areas = np.full(n, 2.0)

    result = integrate_observed(
        observed, geometry, areas, max_incidence_deg=10.0, max_emission_deg=10.0
    )

    assert np.isnan(result.integrated_value)
    assert np.isnan(result.total_area_km2)
    assert result.n_pixels_included == 0
    assert result.n_pixels_excluded_disk_selection == n


def test_all_pixels_nan_area_gives_nan() -> None:
    """Pixels pass the disk selection, but every area is NaN (e.g. emission exceeded
    max_emission_for_area_deg upstream in geometry_engine, so pixel_area_km2 -- or here,
    a caller who mistakenly propagated NaN areas -- can't be used)."""
    n = 10
    geometry = GeometryBatch(
        incidence=np.full(n, np.deg2rad(20.0)),
        emission=np.full(n, np.deg2rad(20.0)),
        phase=np.zeros(n),
    )
    observed = np.full(n, 0.1)
    areas = np.full(n, np.nan)

    result = integrate_observed(observed, geometry, areas)

    assert np.isnan(result.integrated_value)
    assert np.isnan(result.total_area_km2)
    assert result.n_pixels_included == 0
    assert result.n_pixels_excluded_disk_selection == 0
    assert result.n_pixels_excluded_nan_area == n


def test_mixed_nan_areas_excludes_only_those_pixels() -> None:
    n = 6
    geometry = GeometryBatch(
        incidence=np.full(n, np.deg2rad(20.0)),
        emission=np.full(n, np.deg2rad(20.0)),
        phase=np.zeros(n),
    )
    observed = np.full(n, 0.1)
    areas = np.array([2.0, np.nan, 2.0, np.nan, 2.0, 2.0])

    result = integrate_observed(observed, geometry, areas)

    assert result.n_pixels_included == 4
    assert result.n_pixels_excluded_nan_area == 2
    assert result.integrated_value == pytest.approx(4 * 0.1 * 2.0, rel=1e-10)


def test_area_shape_mismatch_raises() -> None:
    geometry = GeometryBatch(
        incidence=np.zeros(5), emission=np.zeros(5), phase=np.zeros(5)
    )
    with pytest.raises(ValueError):
        integrate_observed(np.zeros(5), geometry, np.zeros(3))


def test_geometry_shape_mismatch_raises() -> None:
    geometry = GeometryBatch(
        incidence=np.zeros(5), emission=np.zeros(5), phase=np.zeros(5)
    )
    with pytest.raises(ValueError):
        integrate_observed(np.zeros(3), geometry, np.zeros(3))

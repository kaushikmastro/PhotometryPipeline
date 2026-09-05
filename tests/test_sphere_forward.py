from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from photometry.aggregation.sphere_forward import (  # noqa: E402
    DEFAULT_N_PHI,
    DEFAULT_N_THETA,
    sphere_forward_integrate,
)
from photometry.models.baselines import (  # noqa: E402
    LambertianModel,
    LommelSeeligerModel,
    MinnaertModel,
)

# Each case: (model class, parameters). Same style as the other aggregation test
# suites -- to cover a new model (e.g. AkimovModel) later, add one more pytest.param.
MODEL_CASES = [
    pytest.param(LambertianModel, {"albedo": 0.6}, id="lambertian"),
    pytest.param(LommelSeeligerModel, {"w": 0.5}, id="lommel-seeliger"),
    pytest.param(MinnaertModel, {"albedo": 0.5, "k": 0.6}, id="minnaert"),
]


def _lambert_phase_function(alpha_rad: np.ndarray) -> np.ndarray:
    """Phi_L(alpha) = (1/pi)*(sin(alpha) + (pi-alpha)*cos(alpha)), normalized to
    Phi_L(0)=1 -- the classical closed form for a Lambertian sphere's disk-integrated
    brightness. See test_disk_integrate.py for the same reference, independently
    re-derived/verified there."""
    return (1.0 / np.pi) * (np.sin(alpha_rad) + (np.pi - alpha_rad) * np.cos(alpha_rad))


def _expected_lambert_integral(albedo: float, radius_km: float, alpha_deg: float) -> float:
    alpha_rad = np.deg2rad(alpha_deg)
    return albedo * radius_km**2 * (2.0 / 3.0) * _lambert_phase_function(alpha_rad)


# ---------------------------------------------------------------------------
# Load-bearing correctness test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha_deg", [1e-3, 15.0, 30.0, 60.0, 90.0, 120.0, 150.0])
def test_lambertian_sphere_forward_matches_phase_function_closed_form(alpha_deg: float) -> None:
    """This should now be exact-ish (not just approximate) since we're forward-
    integrating an actual analytic sphere rather than sparse real image pixels --
    tolerance reflects only the measured grid-discretization error at the default
    resolution (~0.2-0.3%), not sampling noise."""
    albedo = 0.6
    radius_km = 262.0  # Vesta mean radius, used here as a concrete physical check

    model = LambertianModel()
    model.parameters["albedo"] = albedo

    curve = sphere_forward_integrate(model, [alpha_deg], radius_km=radius_km)
    expected = _expected_lambert_integral(albedo, radius_km, alpha_deg)

    assert curve.integrated_value[0] == pytest.approx(expected, rel=0.01)


def test_lambertian_sphere_forward_full_curve_shape() -> None:
    """Sanity on the array entry point: a full phase-angle array in one call, monotonic
    decrease with phase (true here, unlike the real-Dawn-frame case, because this is a
    genuine full-sphere integral at every phase angle -- no partial-disk-coverage
    confound)."""
    albedo = 0.6
    radius_km = 262.0
    phases = np.array([5.0, 30.0, 60.0, 90.0, 120.0, 150.0])

    model = LambertianModel()
    model.parameters["albedo"] = albedo

    curve = sphere_forward_integrate(model, phases, radius_km=radius_km)

    assert curve.phase_deg.shape == phases.shape
    assert curve.integrated_value.shape == phases.shape
    assert np.all(np.diff(curve.integrated_value) < 0), "disk-integrated brightness must decrease monotonically with phase for a full-sphere integral"


# ---------------------------------------------------------------------------
# Model-agnostic contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_cls, params", MODEL_CASES)
def test_sphere_forward_runs_for_all_models(model_cls, params) -> None:
    """No model-specific branching: every BasePhotometricModel subclass should produce
    a finite, positive integrated value at a mid-range phase angle."""
    model = model_cls()
    model.parameters.update(params)

    curve = sphere_forward_integrate(model, [45.0], radius_km=262.0)

    assert np.isfinite(curve.integrated_value[0])
    assert curve.integrated_value[0] > 0.0
    assert curve.total_area_km2[0] > 0.0


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


def test_default_discretization_is_adequate() -> None:
    """Pins the measured convergence behaviour: at the default resolution
    (n_theta=n_phi=300), relative error against the exact closed form is comfortably
    under 1% (measured ~0.2-0.3% during development), and refining the grid further
    moves the answer by much less than that -- i.e. the default is past the point of
    diminishing returns, not under-resolved."""
    albedo = 0.6
    radius_km = 262.0
    alpha_deg = 30.0
    expected = _expected_lambert_integral(albedo, radius_km, alpha_deg)

    model = LambertianModel()
    model.parameters["albedo"] = albedo

    default_curve = sphere_forward_integrate(
        model, [alpha_deg], radius_km=radius_km, n_theta=DEFAULT_N_THETA, n_phi=DEFAULT_N_PHI
    )
    finer_curve = sphere_forward_integrate(
        model, [alpha_deg], radius_km=radius_km, n_theta=2 * DEFAULT_N_THETA, n_phi=2 * DEFAULT_N_PHI
    )

    default_rel_err = abs(default_curve.integrated_value[0] - expected) / expected
    finer_rel_err = abs(finer_curve.integrated_value[0] - expected) / expected

    assert default_rel_err < 0.01, f"default resolution should be under 1% error, got {default_rel_err:.4%}"
    assert finer_rel_err < default_rel_err, "doubling resolution should reduce error (monotonic O(1/n) convergence)"
    # the improvement from doubling resolution should itself be small -- confirms the
    # default isn't leaving easy accuracy on the table
    assert default_rel_err - finer_rel_err < 0.01

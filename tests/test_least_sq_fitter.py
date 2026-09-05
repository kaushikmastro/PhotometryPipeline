from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from photometry.core.types import GeometryBatch  # noqa: E402
from photometry.fitting.least_sq import LeastSquaresFitter  # noqa: E402
from photometry.models.baselines import LambertianModel, LommelSeeligerModel  # noqa: E402


def _geometry(n: int = 20) -> GeometryBatch:
    rng = np.random.default_rng(1)
    incidence = rng.uniform(np.deg2rad(5.0), np.deg2rad(70.0), size=n)
    return GeometryBatch(
        incidence=incidence, emission=np.zeros(n), phase=np.zeros(n)
    )


def _synthetic_observed(albedo: float, geometry: GeometryBatch) -> np.ndarray:
    model = LambertianModel()
    model.parameters["albedo"] = albedo
    return np.asarray(model.reflectance(geometry))


def test_fit_populates_success_and_objective() -> None:
    geometry = _geometry()
    observed = _synthetic_observed(0.5, geometry)
    model = LambertianModel()
    model.parameters["albedo"] = 0.2

    result = LeastSquaresFitter().fit(
        model=model, geometry=geometry, observed_reflectance=observed
    )

    assert result.metadata["success"] is True
    assert isinstance(result.objective_value, float)
    assert result.objective_value >= 0.0
    assert result.fitted_parameters["albedo"] == pytest.approx(0.5, abs=1e-4)


def test_weights_none_is_unweighted() -> None:
    geometry = _geometry()
    observed = _synthetic_observed(0.5, geometry)
    model = LambertianModel()
    model.parameters["albedo"] = 0.2

    result = LeastSquaresFitter().fit(
        model=model, geometry=geometry, observed_reflectance=observed, weights=None
    )

    assert result.metadata["weighted"] is False
    assert result.metadata["weight_source"] is None
    assert result.fitted_parameters["albedo"] == pytest.approx(0.5, abs=1e-4)


def test_weights_raw_array() -> None:
    geometry = _geometry()
    observed = _synthetic_observed(0.5, geometry)
    weights = np.ones(len(observed))
    model = LambertianModel()
    model.parameters["albedo"] = 0.2

    result = LeastSquaresFitter().fit(
        model=model, geometry=geometry, observed_reflectance=observed, weights=weights
    )

    assert result.metadata["weighted"] is True
    assert result.metadata["weight_source"] == "array"
    assert result.fitted_parameters["albedo"] == pytest.approx(0.5, abs=1e-4)


def test_weights_dict_n_pixels_iof_iqr() -> None:
    geometry = _geometry()
    observed = _synthetic_observed(0.5, geometry)
    n = len(observed)
    weights = {"n_pixels": np.full(n, 100.0), "iof_iqr": np.full(n, 0.01)}
    model = LambertianModel()
    model.parameters["albedo"] = 0.2

    result = LeastSquaresFitter().fit(
        model=model, geometry=geometry, observed_reflectance=observed, weights=weights
    )

    assert result.metadata["weighted"] is True
    assert result.metadata["weight_source"] == "n_pixels/iof_iqr"
    assert result.fitted_parameters["albedo"] == pytest.approx(0.5, abs=1e-4)


def test_initial_guess_outside_bounds_is_clipped() -> None:
    geometry = _geometry()
    observed = _synthetic_observed(0.5, geometry)
    model = LambertianModel()
    model.parameters["albedo"] = 5.0  # outside the [0, 1] bound

    result = LeastSquaresFitter().fit(
        model=model, geometry=geometry, observed_reflectance=observed
    )

    assert result.metadata["success"] is True
    assert 0.0 <= result.fitted_parameters["albedo"] <= 1.0
    assert result.fitted_parameters["albedo"] == pytest.approx(0.5, abs=1e-4)


def test_model_parameters_restored_after_fit() -> None:
    """residuals() mutates model.parameters per-iteration; fit() must restore the original state."""
    geometry = _geometry()
    observed = _synthetic_observed(0.5, geometry)
    model = LambertianModel()
    model.parameters["albedo"] = 0.2

    LeastSquaresFitter().fit(model=model, geometry=geometry, observed_reflectance=observed)

    assert model.parameters["albedo"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# parameter_errors
# ---------------------------------------------------------------------------


def test_parameter_errors_field_present_on_fit_result() -> None:
    geometry = _geometry()
    observed = _synthetic_observed(0.5, geometry)
    model = LambertianModel()
    model.parameters["albedo"] = 0.2

    result = LeastSquaresFitter().fit(
        model=model, geometry=geometry, observed_reflectance=observed
    )

    assert result.parameter_errors is not None
    assert result.parameter_errors == result.metadata["parameter_errors"]
    assert result.parameter_errors["albedo"] > 0.0


def test_parameter_errors_matches_curve_fit_absolute_sigma_false() -> None:
    """Ground-truth cross-check: cov = inv(J^T J) * (2*cost/dof) should match
    scipy.optimize.curve_fit(..., absolute_sigma=False)'s pcov on the same problem,
    confirming weights (1/sigma) are treated as relative, not double-counted."""
    rng = np.random.default_rng(3)
    n = 40
    incidence = rng.uniform(np.deg2rad(5.0), np.deg2rad(70.0), size=n)
    geometry = GeometryBatch(incidence=incidence, emission=np.zeros(n), phase=np.zeros(n))

    true_albedo = 0.55
    truth = LambertianModel()
    truth.parameters["albedo"] = true_albedo
    clean = np.asarray(truth.reflectance(geometry))

    sigma_true = 0.01
    observed = clean + rng.normal(scale=sigma_true, size=n)

    model = LambertianModel()
    model.parameters["albedo"] = 0.2
    weights = np.full(n, 1.0 / sigma_true)
    result = LeastSquaresFitter().fit(
        model=model, geometry=geometry, observed_reflectance=observed, weights=weights
    )

    def f(mu0: np.ndarray, albedo: float) -> np.ndarray:
        return albedo * np.clip(mu0, 0.0, None) / np.pi

    mu0 = np.cos(incidence)
    popt, pcov = curve_fit(
        f, mu0, observed, p0=[0.2], sigma=np.full(n, sigma_true), absolute_sigma=False
    )

    assert result.fitted_parameters["albedo"] == pytest.approx(float(popt[0]), abs=1e-6)
    assert result.parameter_errors["albedo"] == pytest.approx(
        float(np.sqrt(pcov[0, 0])), rel=1e-4
    )


def test_underdetermined_fit_gives_nan_and_warning() -> None:
    """n_obs <= n_free_params: every parameter_errors entry must be NaN, with a warning."""
    geometry = GeometryBatch(
        incidence=np.array([np.deg2rad(30.0)]),
        emission=np.array([np.deg2rad(20.0)]),
        phase=np.zeros(1),
    )
    observed = np.array([0.1])
    model = LommelSeeligerModel()  # single free parameter, one observation -> dof = 0

    result = LeastSquaresFitter().fit(
        model=model, geometry=geometry, observed_reflectance=observed
    )

    assert np.isnan(result.parameter_errors["w"])
    assert result.metadata["error_estimation_warning"] is not None
    assert "underdetermined" in result.metadata["error_estimation_warning"]


def test_railed_parameter_gives_nan_and_warning() -> None:
    """A parameter pinned at its bound has no meaningful local-curvature uncertainty."""
    rng = np.random.default_rng(11)
    n = 30
    incidence = rng.uniform(np.deg2rad(5.0), np.deg2rad(70.0), size=n)
    emission = rng.uniform(np.deg2rad(5.0), np.deg2rad(70.0), size=n)
    geometry = GeometryBatch(incidence=incidence, emission=emission, phase=np.zeros(n))

    # w's upper bound is 1.0; a strong Lommel-Seeliger signal (w >> 1 if it were allowed
    # to go there) drives the fit to pin at the bound, same as the real Vesta LS result.
    truth = LommelSeeligerModel()
    truth.parameters["w"] = 1.0
    observed = np.asarray(truth.reflectance(geometry)) * 3.0  # push well past the bound

    model = LommelSeeligerModel()
    model.parameters["w"] = 0.5

    result = LeastSquaresFitter().fit(
        model=model, geometry=geometry, observed_reflectance=observed
    )

    assert result.metadata["boundary_hits"]["w"] is True
    assert np.isnan(result.parameter_errors["w"])
    assert result.metadata["error_estimation_warning"] is not None
    assert "railed" in result.metadata["error_estimation_warning"]

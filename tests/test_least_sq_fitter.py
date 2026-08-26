from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from photometry.core.types import GeometryBatch  # noqa: E402
from photometry.fitting.least_sq import LeastSquaresFitter  # noqa: E402
from photometry.models.baselines import LambertianModel  # noqa: E402


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

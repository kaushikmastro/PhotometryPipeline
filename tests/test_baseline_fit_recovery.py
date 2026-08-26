from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from photometry.core.types import Backend, GeometryBatch  # noqa: E402
from photometry.fitting.least_sq import LeastSquaresFitter  # noqa: E402
from photometry.models.baselines import (  # noqa: E402
    LambertianModel,
    LommelSeeligerModel,
    MinnaertModel,
)

# Each case: (model class, true parameters to recover, initial guess to fit from).
# To cover a new baseline model (e.g. AkimovModel), add one more pytest.param here —
# no new test function needed.
FIT_RECOVERY_CASES = [
    pytest.param(LambertianModel, {"albedo": 0.62}, {"albedo": 0.3}, id="lambertian"),
    pytest.param(LommelSeeligerModel, {"w": 0.55}, {"w": 0.2}, id="lommel-seeliger"),
    pytest.param(
        MinnaertModel,
        {"albedo": 0.8, "k": 0.65},
        {"albedo": 0.3, "k": 1.0},
        id="minnaert",
    ),
]


def _synthetic_geometry(rng: np.random.Generator, n: int = 200) -> GeometryBatch:
    incidence = rng.uniform(np.deg2rad(5.0), np.deg2rad(70.0), size=n)
    emission = rng.uniform(np.deg2rad(5.0), np.deg2rad(70.0), size=n)
    phase = np.array(
        [rng.uniform(abs(i - e) + 1e-3, i + e) for i, e in zip(incidence, emission, strict=True)]
    )
    return GeometryBatch(incidence=incidence, emission=emission, phase=phase)


@pytest.mark.parametrize("model_cls, true_params, initial_guess", FIT_RECOVERY_CASES)
def test_fit_recovers_known_parameters(model_cls, true_params, initial_guess) -> None:
    """Synthetic data generated from known parameters must be recovered by LeastSquaresFitter."""
    rng = np.random.default_rng(0)
    geometry = _synthetic_geometry(rng)

    truth_model = model_cls()
    truth_model.set_backend(Backend.NUMPY)
    truth_model.parameters.update(true_params)
    clean = np.asarray(truth_model.reflectance(geometry))

    noise = rng.normal(scale=0.01 * max(float(clean.mean()), 1e-6), size=clean.shape)
    observed = np.clip(clean + noise, 1e-6, None)

    fit_model = model_cls()
    fit_model.set_backend(Backend.NUMPY)
    fit_model.parameters.update(initial_guess)

    result = LeastSquaresFitter().fit(
        model=fit_model, geometry=geometry, observed_reflectance=observed
    )

    assert result.metadata["success"]
    for name, expected in true_params.items():
        assert result.fitted_parameters[name] == pytest.approx(expected, abs=0.03)

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
from photometry.models.baselines import MinnaertModel  # noqa: E402


def _geometry_batch(incidence: np.ndarray, emission: np.ndarray, phase: np.ndarray | None = None) -> GeometryBatch:
    if phase is None:
        phase = np.zeros_like(incidence)
    return GeometryBatch(incidence=incidence, emission=emission, phase=phase)


def test_minnaert_parameter_contract() -> None:
    model = MinnaertModel()

    assert model.parameter_names() == ["albedo", "k"]
    assert model.parameter_bounds() == {"albedo": (0.0, 10.0), "k": (0.0, 2.0)}
    assert model.parameters["albedo"] == 1.0
    assert model.parameters["k"] == 0.5


def test_minnaert_known_values_numpy() -> None:
    model = MinnaertModel()
    model.set_backend(Backend.NUMPY)
    model.parameters["albedo"] = 2.0
    model.parameters["k"] = 1.0

    geometry = _geometry_batch(
        incidence=np.array([0.0, np.pi / 3.0], dtype=np.float64),
        emission=np.array([0.0, np.pi / 3.0], dtype=np.float64),
    )
    result = model.reflectance(geometry)

    expected = np.array([2.0, 1.0], dtype=np.float64)
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-10)


def test_minnaert_terminator_safety_numpy() -> None:
    model = MinnaertModel()
    model.set_backend(Backend.NUMPY)
    model.parameters["albedo"] = 1.0
    model.parameters["k"] = 0.5

    geometry = _geometry_batch(
        incidence=np.array([0.0], dtype=np.float64),
        emission=np.array([np.pi / 2.0], dtype=np.float64),
    )
    result = model.reflectance(geometry)

    assert np.isfinite(result).all()
    assert result[0] > 0.0


def test_minnaert_dual_backend_equivalence() -> None:
    torch = pytest.importorskip("torch")

    incidence = np.array([0.1, 0.4, 0.7], dtype=np.float64)
    emission = np.array([0.2, 0.5, 0.8], dtype=np.float64)

    numpy_model = MinnaertModel()
    numpy_model.set_backend(Backend.NUMPY)
    numpy_model.parameters["albedo"] = 1.7
    numpy_model.parameters["k"] = 0.6
    numpy_result = numpy_model.reflectance(_geometry_batch(incidence, emission))

    torch_model = MinnaertModel()
    torch_model.set_backend(Backend.TORCH)
    torch_model.parameters["albedo"] = 1.7
    torch_model.parameters["k"] = 0.6
    torch_geometry = GeometryBatch(
        incidence=torch.as_tensor(incidence, dtype=torch.float64),
        emission=torch.as_tensor(emission, dtype=torch.float64),
        phase=torch.zeros_like(torch.as_tensor(incidence, dtype=torch.float64)),
    )
    torch_result = torch_model.reflectance(torch_geometry)

    np.testing.assert_allclose(numpy_result, torch_result.detach().cpu().numpy(), rtol=1e-6, atol=1e-6)

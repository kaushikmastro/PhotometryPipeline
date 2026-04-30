from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from photometry.core.types import Backend, GeometryBatch  # noqa: E402
from photometry.models.base import ModelRegistry, UnitError  # noqa: E402
from photometry.models.lambertian import LambertianModel  # noqa: E402


def _geometry_batch(
    incidence: np.ndarray,
    emission: np.ndarray | None = None,
    phase: np.ndarray | None = None,
) -> GeometryBatch:
    if emission is None:
        emission = np.zeros_like(incidence)
    if phase is None:
        phase = np.zeros_like(incidence)
    return GeometryBatch(incidence=incidence, emission=emission, phase=phase)


def _valid_geometry_rows_strategy():
    scalar = st.floats(
        min_value=0.0,
        max_value=float(np.pi / 2.0),
        allow_nan=False,
        allow_infinity=False,
        width=64,
    )
    return st.lists(st.tuples(scalar, scalar, scalar), min_size=1, max_size=64)


def _terminator_incidence_strategy():
    return st.lists(
        st.floats(
            min_value=float(np.pi / 2.0 + 1e-3),
            max_value=float(np.pi / 2.0 + 9e-3),
            allow_nan=False,
            allow_infinity=False,
            width=64,
        ),
        min_size=1,
        max_size=64,
    )


def test_model_registration_and_bounds() -> None:
    registry = ModelRegistry.get_registered_models()

    assert "lambertian" in registry
    assert registry["lambertian"] is LambertianModel

    model = LambertianModel()
    assert model.parameters["albedo"] == 1.0
    assert model.parameter_bounds()["albedo"] == (0.0, 1.0)


def test_unit_enforcement() -> None:
    model = LambertianModel()
    geometry = _geometry_batch(
        np.array([90.0], dtype=np.float64),
        emission=np.array([0.0], dtype=np.float64),
        phase=np.array([0.0], dtype=np.float64),
    )

    with pytest.raises(UnitError, match="angles appear to be in degrees"):
        model.reflectance(geometry)


def test_known_values_numpy() -> None:
    model = LambertianModel()
    model.set_backend(Backend.NUMPY)

    geometry = _geometry_batch(
        np.array([0.0, np.pi / 3.0], dtype=np.float64),
        emission=np.zeros(2, dtype=np.float64),
        phase=np.zeros(2, dtype=np.float64),
    )

    result = model.reflectance(geometry)
    expected = np.array([1.0 / np.pi, 0.5 / np.pi], dtype=np.float64)

    np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-12)


@settings(deadline=None, max_examples=50)
@given(incidence_values=_terminator_incidence_strategy())
def test_terminator_law(incidence_values: list[float]) -> None:
    model = LambertianModel()
    model.set_backend(Backend.NUMPY)

    incidence = np.asarray(incidence_values, dtype=np.float64)
    geometry = _geometry_batch(incidence)

    result = model.reflectance(geometry)

    np.testing.assert_array_equal(result, np.zeros_like(result))
    assert np.max(result) == 0.0


@settings(deadline=None, max_examples=50)
@given(rows=_valid_geometry_rows_strategy())
def test_dual_backend_equivalence(rows: list[tuple[float, float, float]]) -> None:
    torch = pytest.importorskip("torch")

    data = np.asarray(rows, dtype=np.float64)
    incidence = data[:, 0]
    emission = data[:, 1]
    phase = data[:, 2]

    numpy_model = LambertianModel()
    numpy_model.set_backend(Backend.NUMPY)
    numpy_result = numpy_model.reflectance(_geometry_batch(incidence, emission, phase))

    torch_model = LambertianModel()
    torch_model.set_backend(Backend.TORCH)
    torch_geometry = GeometryBatch(
        incidence=torch.as_tensor(incidence, dtype=torch.float64),
        emission=torch.as_tensor(emission, dtype=torch.float64),
        phase=torch.as_tensor(phase, dtype=torch.float64),
    )
    torch_result = torch_model.reflectance(torch_geometry)

    np.testing.assert_allclose(
        numpy_result,
        torch_result.detach().cpu().numpy(),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        torch_result,
        torch.as_tensor(numpy_result, dtype=torch.float64),
        rtol=1e-6,
        atol=1e-6,
    )

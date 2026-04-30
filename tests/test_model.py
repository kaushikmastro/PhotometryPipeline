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
from photometry.models.baselines import LambertianModel, LommelSeeligerModel  # noqa: E402


MODEL_CASES = [
    pytest.param(LambertianModel, "lambertian", "albedo", (0.0, 1.0), id="lambertian"),
    pytest.param(LommelSeeligerModel, "lommel_seeliger", "w", (0.0, 1.0), id="lommel-seeliger"),
]


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


def _geometry_batch_strategy():
    scalar = st.floats(
        min_value=0.0,
        max_value=float(np.pi / 2.0),
        allow_nan=False,
        allow_infinity=False,
        width=64,
    )
    return st.lists(st.tuples(scalar, scalar, scalar), min_size=1, max_size=64).map(
        lambda rows: _geometry_batch_from_rows(np.asarray(rows, dtype=np.float64))
    )


def _geometry_batch_from_rows(data: np.ndarray) -> GeometryBatch:
    incidence = data[:, 0]
    emission = data[:, 1]
    phase = data[:, 2]
    return _geometry_batch(incidence, emission, phase)


def _make_model(model_cls: type[LambertianModel | LommelSeeligerModel]):
    model = model_cls()
    model.set_backend(Backend.NUMPY)
    return model


@pytest.mark.parametrize("model_cls, model_name, parameter_name, bounds", MODEL_CASES)
def test_model_registration_and_bounds(model_cls, model_name, parameter_name, bounds) -> None:
    registry = ModelRegistry.get_registered_models()

    assert model_name in registry
    assert registry[model_name] is model_cls

    model = model_cls()
    assert model.parameters[parameter_name] == 1.0
    assert model.parameter_bounds()[parameter_name] == bounds
    assert model.parameter_names() == [parameter_name]
    assert model.parameter_priors()[parameter_name].lower_bound == bounds[0]
    assert model.parameter_priors()[parameter_name].upper_bound == bounds[1]


@pytest.mark.parametrize("model_cls, model_name, parameter_name, bounds", MODEL_CASES)
def test_from_dict_roundtrip(model_cls, model_name, parameter_name, bounds) -> None:
    payload = {
        "model_name": model_cls.model_name,
        "parameters": {parameter_name: 0.37},
        "metadata": {"source": "unit-test"},
        "backend": Backend.NUMPY.value,
    }

    model = model_cls.from_dict(payload)

    assert model.model_name == model_cls.model_name
    assert model.parameters[parameter_name] == pytest.approx(0.37)
    assert model.metadata == {"source": "unit-test"}
    assert model.backend == Backend.NUMPY


@pytest.mark.parametrize("model_cls, model_name, parameter_name, bounds", MODEL_CASES)
def test_unit_enforcement(model_cls, model_name, parameter_name, bounds) -> None:
    model = model_cls()
    geometry = _geometry_batch(
        np.array([90.0], dtype=np.float64),
        emission=np.array([0.0], dtype=np.float64),
        phase=np.array([0.0], dtype=np.float64),
    )

    with pytest.raises(UnitError, match="angles appear to be in degrees"):
        model.reflectance(geometry)


@pytest.mark.parametrize("model_cls, model_name, parameter_name, bounds", MODEL_CASES)
def test_known_values_numpy(model_cls, model_name, parameter_name, bounds) -> None:
    model = _make_model(model_cls)

    if parameter_name == "albedo":
        model.parameters[parameter_name] = 1.0
        geometry = _geometry_batch(
            np.array([0.0, np.pi / 3.0], dtype=np.float64),
            emission=np.zeros(2, dtype=np.float64),
            phase=np.zeros(2, dtype=np.float64),
        )
        expected = np.array([1.0 / np.pi, 0.5 / np.pi], dtype=np.float64)
    else:
        model.parameters[parameter_name] = 1.0
        geometry = _geometry_batch(
            np.array([0.0, np.pi / 3.0], dtype=np.float64),
            emission=np.array([0.0, np.pi / 3.0], dtype=np.float64),
            phase=np.zeros(2, dtype=np.float64),
        )
        expected = np.array([1.0 / 8.0, 1.0 / 8.0], dtype=np.float64)

    result = model.reflectance(geometry)

    np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-10)


@settings(deadline=None, max_examples=50)
@given(geometry=_geometry_batch_strategy())
@pytest.mark.parametrize("model_cls, model_name, parameter_name, bounds", MODEL_CASES)
def test_reflectance_is_non_negative(model_cls, model_name, parameter_name, bounds, geometry: GeometryBatch) -> None:
    model = _make_model(model_cls)
    model.parameters[parameter_name] = 0.73

    result = model.reflectance(geometry)

    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)


@settings(deadline=None, max_examples=50)
@given(geometry=_geometry_batch_strategy())
@pytest.mark.parametrize("model_cls", [LambertianModel, LommelSeeligerModel])
def test_lommel_seeliger_reciprocity(model_cls, geometry: GeometryBatch) -> None:
    # I/F models include a mu0 projection factor and do not exhibit pure Helmholtz reciprocity.
    pytest.skip("I/F baselines are not expected to satisfy pure reciprocity.")


def test_lommel_seeliger_limb_behavior() -> None:
    model = _make_model(LommelSeeligerModel)
    model.parameters["w"] = 1.0

    geometry = _geometry_batch(
        np.array([np.pi / 2.0], dtype=np.float64),
        emission=np.array([np.pi / 2.0], dtype=np.float64),
        phase=np.array([0.0], dtype=np.float64),
    )

    result = model.reflectance(geometry)

    np.testing.assert_array_equal(result, np.zeros_like(result))
    assert np.isfinite(result).all()


@settings(deadline=None, max_examples=50)
@given(rows=_valid_geometry_rows_strategy())
@pytest.mark.parametrize("model_cls, model_name, parameter_name, bounds", MODEL_CASES)
def test_dual_backend_equivalence(model_cls, model_name, parameter_name, bounds, rows: list[tuple[float, float, float]]) -> None:
    torch = pytest.importorskip("torch")

    data = np.asarray(rows, dtype=np.float64)
    incidence = data[:, 0]
    emission = data[:, 1]
    phase = data[:, 2]

    numpy_model = model_cls()
    numpy_model.set_backend(Backend.NUMPY)
    numpy_model.parameters[parameter_name] = 0.61
    numpy_result = numpy_model.reflectance(_geometry_batch(incidence, emission, phase))

    torch_model = model_cls()
    torch_model.set_backend(Backend.TORCH)
    torch_model.parameters[parameter_name] = 0.61
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

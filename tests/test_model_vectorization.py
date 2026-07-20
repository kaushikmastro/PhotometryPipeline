from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from photometry.core.types import GeometryBatch  # noqa: E402
from photometry.models import baselines as _baselines  # noqa: F401,E402
from photometry.models.base import ModelRegistry  # noqa: E402


@pytest.fixture(scope="module")
def million_point_geometry() -> GeometryBatch:
    """One-million-point vectorization workload for backend smoke testing."""
    n = 1_000_000
    incidence = np.full(n, np.deg2rad(30.0), dtype=np.float64)
    emission = np.full(n, np.deg2rad(20.0), dtype=np.float64)
    phase = np.full(n, np.deg2rad(40.0), dtype=np.float64)
    return GeometryBatch(incidence=incidence, emission=emission, phase=phase)


@pytest.fixture(scope="module")
def registered_models() -> list:
    """Instantiate every registered model with default parameters."""
    return [model_cls() for model_cls in ModelRegistry.get_registered_models().values()]


def test_registered_models_vectorization_contract(registered_models, million_point_geometry):
    """
    Benchmark every registered model on a one-million-element geometry batch.

    Contract:
    - reflectance must complete in under 5 seconds on a single CPU core
    - output must preserve the vectorized shape
    """
    assert registered_models, "No registered photometry models were found."

    for model in registered_models:
        t0 = time.perf_counter()
        result = model.reflectance(million_point_geometry)
        elapsed = time.perf_counter() - t0

        assert elapsed < 5.0, f"{model.model_name} took {elapsed:.3f}s for 1,000,000 points"
        assert result.shape == (1_000_000,)

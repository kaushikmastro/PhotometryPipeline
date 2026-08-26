from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from photometry.core.types import GeometryBatch  # noqa: E402
from photometry.fitting.least_sq import LeastSquaresFitter  # noqa: E402
from photometry.models.hapke import HapkeModel  # noqa: E402

# Real-data regression target: committed Case 1 result (CLAUDE.md), full-data,
# illuminated regime (iof>0.01), Config A of scripts/utils/run_prelim_physfilter.py.
COMMITTED_CASE1 = {"w": 0.46993, "g": -0.33688, "theta_bar": 8.2662}
CASE1_FIXED_PARAMETERS = {"B0": 1.03, "h": 0.04}
CASE1_PARAMETER_BOUNDS = {"w": (0.3, 0.7), "g": (-0.6, 0.0), "theta_bar": (1.0, 50.0)}


def _multi_start_fit(
    geometry: GeometryBatch,
    observed: np.ndarray,
    weights: np.ndarray | None,
    parameter_bounds: dict[str, tuple[float, float]],
    n_starts: int,
    seed: int,
):
    """Mirror the multi-start pattern Hapke.ipynb / run_baseline_fit.py actually use:
    LeastSquaresFitter does one deterministic TRF call per start; multi-start is a
    plain loop around it, keeping the lowest-cost successful result."""
    fitter = LeastSquaresFitter()
    rng = np.random.default_rng(seed)

    best_result = None
    best_cost = np.inf
    for _ in range(n_starts):
        model = HapkeModel(
            enable_shoe=True, enable_roughness=True, fixed_parameters=CASE1_FIXED_PARAMETERS
        )
        guess = {p: float(rng.uniform(lo, hi)) for p, (lo, hi) in parameter_bounds.items()}
        model.parameters.update(guess)

        # Tighten bounds for this fit, same override pattern as run_baseline_fit.py.
        orig_bounds = model.parameter_bounds

        def _bounded(orig=orig_bounds, pb=parameter_bounds):
            b = orig()
            b.update(pb)
            return b

        model.parameter_bounds = _bounded

        result = fitter.fit(
            model=model, geometry=geometry, observed_reflectance=observed, weights=weights
        )
        if result.metadata["success"] and result.objective_value < best_cost:
            best_cost = result.objective_value
            best_result = result

    return best_result


def test_hapke_case1_fit_recovery_synthetic() -> None:
    """Fast smoke test: small multi-start fit on synthetic data recovers known parameters."""
    rng = np.random.default_rng(7)
    true_params = {"w": 0.45, "g": -0.30, "theta_bar": 15.0}

    n = 300
    incidence = rng.uniform(np.deg2rad(5.0), np.deg2rad(45.0), size=n)
    emission = rng.uniform(np.deg2rad(5.0), np.deg2rad(45.0), size=n)
    phase = np.array(
        [rng.uniform(abs(i - e) + 1e-3, i + e) for i, e in zip(incidence, emission, strict=True)]
    )
    geometry = GeometryBatch(incidence=incidence, emission=emission, phase=phase)

    truth_model = HapkeModel(
        enable_shoe=True,
        enable_roughness=True,
        fixed_parameters=CASE1_FIXED_PARAMETERS,
        parameters=dict(true_params),
    )
    clean = np.asarray(truth_model._reflectance_numpy(geometry))
    observed = clean + rng.normal(scale=0.005, size=clean.shape)

    best_result = _multi_start_fit(
        geometry=geometry,
        observed=observed,
        weights=None,
        parameter_bounds={"w": (0.3, 0.6), "g": (-0.4, -0.2), "theta_bar": (1.0, 30.0)},
        n_starts=8,
        seed=7,
    )

    assert best_result is not None
    assert best_result.fitted_parameters["w"] == pytest.approx(true_params["w"], abs=0.02)
    assert best_result.fitted_parameters["g"] == pytest.approx(true_params["g"], abs=0.02)
    assert best_result.fitted_parameters["theta_bar"] == pytest.approx(
        true_params["theta_bar"], abs=1.5
    )


@pytest.mark.slow
def test_hapke_case1_real_data_regression() -> None:
    """Full 100-start Case 1 fit on the real committed dataset. Reproduces the
    committed headline result to ~5 significant figures when it passes; runtime is
    reported via -s (measured ~5s locally, on ~950 pre-binned rows)."""
    data_path = PROJECT_ROOT / "data" / "silver" / "dsk256" / "binned_prelim_iof001.parquet"
    if not data_path.exists():
        pytest.skip(f"real committed-fit dataset not present: {data_path}")

    df = pd.read_parquet(data_path)
    mean_iof = df["mean_iof"].to_numpy()
    std_iof = df["std_iof"].to_numpy()
    std_iof = np.where(std_iof == 0, mean_iof * 0.01, std_iof)
    weights = 1.0 / std_iof

    geometry = GeometryBatch(
        incidence=np.deg2rad(df["mean_incidence"].to_numpy()),
        emission=np.deg2rad(df["mean_emission"].to_numpy()),
        phase=np.deg2rad(df["mean_phase"].to_numpy()),
    )

    t0 = time.perf_counter()
    best_result = _multi_start_fit(
        geometry=geometry,
        observed=mean_iof,
        weights=weights,
        parameter_bounds=CASE1_PARAMETER_BOUNDS,
        n_starts=100,
        seed=42,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n[test_hapke_case1_real_data_regression] 100-start fit took {elapsed:.2f}s")

    assert best_result is not None
    assert best_result.fitted_parameters["w"] == pytest.approx(COMMITTED_CASE1["w"], abs=1e-3)
    assert best_result.fitted_parameters["g"] == pytest.approx(COMMITTED_CASE1["g"], abs=1e-3)
    assert best_result.fitted_parameters["theta_bar"] == pytest.approx(
        COMMITTED_CASE1["theta_bar"], abs=1e-3
    )

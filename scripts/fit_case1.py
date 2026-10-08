"""Reconstruction of the committed Case 1 Hapke fit, runnable end-to-end from the
already-binned golden parquet (no DuckDB / no >100M-pixel raw table touched here --
that stage is `scripts/utils/run_prelim_physfilter.py`'s `prebin()`, not this script).

WHY THIS FILE EXISTS: `scripts/run_baseline_fit.py`, which `run_prelim_physfilter.py`
imports via `importlib.util.spec_from_file_location(...)` to actually run Case 1, does
not exist anywhere in the current working tree or git history (confirmed: repo-wide
`find`/`grep`, zero hits both for the file and for `def run_case`). Circumstantial
evidence (`.gitignore`'s `scripts/adhoc/` rule + its "preserved on disk, not tracked"
comment, plus the reflog commit "reorganize scripts/ into tracked CORE set + gitignored
adhoc/ diagnostics" postdating the June 8, 2026 job that produced the committed numbers)
points to it having been swept into that gitignored bucket and later deleted from disk
-- but this is inference, not something read from a diff. There is no commit to recover
it from.

WHAT IS READ, NOT INFERRED, below: `tests/test_hapke_fit_recovery.py` already contains
a working real-data regression test (`test_hapke_case1_real_data_regression`, marked
`@pytest.mark.slow`) against this exact golden parquet, with a helper (`_multi_start_fit`)
whose own docstring states it "mirror[s] the multi-start pattern Hapke.ipynb /
run_baseline_fit.py actually use." `CASE1_FIXED_PARAMETERS`, `CASE1_PARAMETER_BOUNDS`,
`COMMITTED_CASE1`, the multi-start loop structure (`np.random.default_rng(seed)`, one
deterministic TRF call per start via `LeastSquaresFitter`, keep lowest-cost successful
result), and critically the **weighting formula** (`weights = 1.0 / std_iof`, zero-guarded
by substituting `mean_iof * 0.01` where `std_iof == 0`) are all copied verbatim from that
file, not re-derived. `CASE1_FIXED_PARAMETERS` and `CASE1_PARAMETER_BOUNDS` are also
independently confirmed against `scripts/utils/run_prelim_physfilter.py:112-114`
(`fixed_parameters={"B0": 1.03, "h": 0.04}`, `parameter_bounds={"w": (0.3, 0.7), "g":
(-0.6, 0.0), "theta_bar": (1.0, 50.0)}`, `n_starts=100`) -- identical in both sources.

WHAT IS **NOT** SETTLED: `weighting.py`'s own module docstring (`Weighting.robust_binned`,
`src/photometry/fitting/weighting.py:71-78`) separately claims Case 1 used
`sqrt(n_pixels)/iof_iqr` (the legacy `{"n_pixels","iof_iqr"}` dict path), which is a
**different formula** from `test_hapke_fit_recovery.py`'s `1.0/std_iof` -- these are not
algebraically equivalent (one has a sqrt(n_pixels) factor keyed to a per-bin pixel count
of order 1e5-1e6; the other doesn't). Both claims exist in the tracked codebase right now
and contradict each other. This script exists partly to settle that empirically: run it
once per `--weighting` choice and compare against `COMMITTED_CASE1`.

`--weighting robust_binned` cannot actually be evaluated from this golden parquet: its
formula needs a per-bin IQR, and the parquet's schema (confirmed from
`run_prelim_physfilter.py`'s `prebin()` SQL, `run_prelim_physfilter.py:68-74`) only has
`mean_incidence, mean_emission, mean_phase, mean_iof, std_iof, n_pixels` -- no IQR column
anywhere. Passing `--weighting robust_binned` raises `SystemExit` explaining this rather
than silently substituting an approximation (e.g. IQR ~= 1.349*std for a normal
distribution) that nobody asked for.

Usage:
    python scripts/fit_case1.py --weighting legacy_std
    python scripts/fit_case1.py --weighting statistical
    python scripts/fit_case1.py --weighting uniform
    python scripts/fit_case1.py --weighting robust_binned   # exits with an explanation
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from photometry.core.types import GeometryBatch  # noqa: E402
from photometry.fitting.least_sq import LeastSquaresFitter  # noqa: E402
from photometry.fitting.weighting import Weighting  # noqa: E402
from photometry.models.hapke import HapkeModel  # noqa: E402

DEFAULT_DATA_PATH = ROOT / "data" / "silver" / "dsk256" / "binned_prelim_iof001.parquet"

# Read verbatim from tests/test_hapke_fit_recovery.py:22-24.
COMMITTED_CASE1 = {"w": 0.46993, "g": -0.33688, "theta_bar": 8.2662}
CASE1_FIXED_PARAMETERS = {"B0": 1.03, "h": 0.04}
CASE1_PARAMETER_BOUNDS = {"w": (0.3, 0.7), "g": (-0.6, 0.0), "theta_bar": (1.0, 50.0)}
N_STARTS = 100
SEED = 42

WEIGHTING_CHOICES = ("legacy_std", "statistical", "robust_binned", "uniform")


def multi_start_fit(
    geometry: GeometryBatch,
    observed: np.ndarray,
    weights: np.ndarray | None,
    parameter_bounds: dict[str, tuple[float, float]],
    n_starts: int,
    seed: int,
):
    """Copied from tests/test_hapke_fit_recovery.py::_multi_start_fit verbatim (only
    renamed, underscore dropped so it's importable). Not modified -- see that file for
    the original. Kept here as a duplicate rather than an import from tests/ so this
    script has no test-suite dependency; see the module docstring for why the test
    itself wasn't refactored to import from here instead (not done without being able
    to run pytest to verify the refactor)."""
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


def build_weights(scheme: str, df: pd.DataFrame) -> np.ndarray | Weighting:
    n_pixels = df["n_pixels"].to_numpy()
    mean_iof = df["mean_iof"].to_numpy()
    std_iof = df["std_iof"].to_numpy()

    if scheme == "legacy_std":
        # Read verbatim from tests/test_hapke_fit_recovery.py:120-122. Bypasses the
        # Weighting class entirely -- a raw array, exactly as that test builds it.
        std_safe = np.where(std_iof == 0, mean_iof * 0.01, std_iof)
        return 1.0 / std_safe

    if scheme == "statistical":
        return Weighting.statistical(n_pixels=n_pixels, std=std_iof)

    if scheme == "uniform":
        return Weighting.uniform(n_obs=len(df))

    if scheme == "robust_binned":
        raise SystemExit(
            "--weighting robust_binned cannot be evaluated from this golden parquet: "
            "Weighting.robust_binned(n_pixels, iqr) needs a per-bin interquartile-range "
            "column, and this parquet's schema (mean_incidence, mean_emission, "
            "mean_phase, mean_iof, std_iof, n_pixels -- confirmed from "
            "run_prelim_physfilter.py's prebin() SQL) has no such column. Not "
            "substituting an IQR approximation (e.g. 1.349*std) since nobody asked for "
            "one -- that would be guessing, not reading."
        )

    raise ValueError(f"unknown weighting scheme: {scheme!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--weighting", choices=WEIGHTING_CHOICES, default="legacy_std")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"golden parquet not found: {args.data}")

    df = pd.read_parquet(args.data)
    print(f"Loaded {len(df)} bins from {args.data}")
    print(f"Weighting scheme: {args.weighting}")
    print(f"Fixed parameters: {CASE1_FIXED_PARAMETERS}")
    print(f"Parameter bounds: {CASE1_PARAMETER_BOUNDS}")
    print(f"n_starts={N_STARTS}  seed={SEED}")

    weights_obj = build_weights(args.weighting, df)
    weights = weights_obj.values if isinstance(weights_obj, Weighting) else weights_obj

    geometry = GeometryBatch(
        incidence=np.deg2rad(df["mean_incidence"].to_numpy()),
        emission=np.deg2rad(df["mean_emission"].to_numpy()),
        phase=np.deg2rad(df["mean_phase"].to_numpy()),
    )
    observed = df["mean_iof"].to_numpy()

    t0 = time.perf_counter()
    result = multi_start_fit(
        geometry=geometry,
        observed=observed,
        weights=weights,
        parameter_bounds=CASE1_PARAMETER_BOUNDS,
        n_starts=N_STARTS,
        seed=SEED,
    )
    elapsed = time.perf_counter() - t0

    if result is None:
        raise SystemExit("Optimization collapsed: no multi-start run converged.")

    print(f"\nElapsed: {elapsed:.2f}s")
    print(f"Fitted parameters: {result.fitted_parameters}")
    print(f"reduced_chi_square: {result.metadata.get('reduced_chi_square')}")

    print("\nComparison to committed Case 1 (5 decimals):")
    all_match = True
    for name, committed_value in COMMITTED_CASE1.items():
        fitted_value = result.fitted_parameters[name]
        match = round(fitted_value, 5) == round(committed_value, 5)
        all_match &= match
        print(
            f"  {name:10s} fitted={fitted_value:.5f}  committed={committed_value:.5f}  "
            f"match_to_5dp={match}"
        )
    print(f"\nAll three parameters match to 5 decimals: {all_match}")


if __name__ == "__main__":
    main()

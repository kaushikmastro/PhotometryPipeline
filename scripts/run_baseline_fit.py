"""
Li et al. (2013) Table 2 baseline fit — Survey-only, low-i/low-e subset.

Replicates the three cases from Li et al. 2013 Table 2 (F1 clear filter,
disk-resolved) on Survey-only cubes with i<50° and e<50°:

  Case 1: fix B0=1.03, h=0.04; fit w, g, theta_bar     (3 free)
  Case 2: fix B0=1.03;         fit w, g, theta_bar, h  (4 free)
  Case 3: all five free        fit w, g, theta_bar, B0, h (5 free;
          B0/h will be unconstrainable — zero bins below 5° phase)

Data: Survey mission phase only, i<50°, e<50°, ≥10 px/bin.
Weights: 1/std_iof (inverse within-bin scatter).
100 random multi-starts per case; best solution by lowest cost retained.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from photometry.core.types import GeometryBatch
from photometry.fitting.least_sq import LeastSquaresFitter
from photometry.models.hapke import HapkeModel


DATA_PATH = (
    ROOT / "data" / "06_silver_layer_dsk256"
    / "combined_rc_survey_sample_corrected_dsk256.parquet"
)
GRID_SIZE = 5.0
INC_MAX = 50.0
EMI_MAX = 50.0
MIN_PIXELS = 10


def build_survey_loie() -> pl.DataFrame:
    """Survey-only cubes, i<50°, e<50°, ≥MIN_PIXELS per bin."""
    df = (
        pl.scan_parquet(DATA_PATH)
        .filter(pl.col("mission_phase") == "SURVEY")
        .select(["phase", "incidence", "emission", "iof"])
        .with_columns([
            ((pl.col("phase")     / GRID_SIZE).round() * GRID_SIZE).alias("alpha_grid"),
            ((pl.col("incidence") / GRID_SIZE).round() * GRID_SIZE).alias("i_grid"),
            ((pl.col("emission")  / GRID_SIZE).round() * GRID_SIZE).alias("e_grid"),
        ])
        .group_by(["alpha_grid", "i_grid", "e_grid"])
        .agg([
            pl.col("incidence").mean().alias("mean_incidence"),
            pl.col("emission").mean().alias("mean_emission"),
            pl.col("phase").mean().alias("mean_phase"),
            pl.col("iof").mean().alias("mean_iof"),
            pl.col("iof").std().alias("std_iof"),
            pl.col("iof").count().alias("n_pixels"),
        ])
        .filter(
            (pl.col("n_pixels") >= MIN_PIXELS)
            & (pl.col("mean_incidence") < INC_MAX)
            & (pl.col("mean_emission") < EMI_MAX)
        )
        .drop_nulls(subset=["std_iof"])
        .sort(["alpha_grid", "i_grid", "e_grid"])
        .collect()
    )
    # Guard against zero-std bins (single-pixel clusters that survived n_pixels filter)
    return df.with_columns(
        pl.when(pl.col("std_iof") == 0)
        .then(pl.col("mean_iof") * 0.01)
        .otherwise(pl.col("std_iof"))
        .alias("std_iof")
    )


def fractional_rms(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean(((observed - predicted) / np.maximum(observed, 1e-9)) ** 2))
    )


def run_case(
    *,
    name: str,
    fixed_parameters: dict[str, float],
    parameter_bounds: dict[str, tuple[float, float]],
    n_starts: int = 100,
) -> dict[str, object]:
    """Fit one Li et al. case with multi-start TRF least-squares.

    parameter_bounds covers only the FREE parameters (those not in
    fixed_parameters). It is used for both the random starting-point
    draw and as the actual optimization bounds (overrides model defaults).
    """
    df = build_survey_loie()
    geometry = GeometryBatch(
        incidence=np.deg2rad(df["mean_incidence"].to_numpy()),
        emission=np.deg2rad(df["mean_emission"].to_numpy()),
        phase=np.deg2rad(df["mean_phase"].to_numpy()),
    )
    observed = df["mean_iof"].to_numpy()
    weights  = 1.0 / df["std_iof"].to_numpy()

    print(f"\n[{name}] {df.height} bins  "
          f"phase {df['mean_phase'].min():.1f}°–{df['mean_phase'].max():.1f}°  "
          f"fixed={list(fixed_parameters)}  free={list(parameter_bounds)}")

    fitter = LeastSquaresFitter()
    rng = np.random.default_rng(42)

    best_result = None
    best_cost = np.inf
    all_results: list[dict] = []

    for start_idx in range(n_starts):
        model = HapkeModel(
            enable_shoe=True,
            enable_roughness=True,
            fixed_parameters=fixed_parameters,
        )

        # Set random starting point within physical bounds
        guess = {
            pname: float(rng.uniform(lo, hi))
            for pname, (lo, hi) in parameter_bounds.items()
        }
        model.parameters.update(guess)

        # Override model.parameter_bounds() so TRF uses our tighter bounds,
        # not the mathematical maxima from HapkeModel defaults.
        _orig_bounds = model.parameter_bounds

        def _bounded(orig=_orig_bounds, pb=parameter_bounds):
            b = orig()
            b.update(pb)
            return b

        model.parameter_bounds = _bounded

        result = fitter.fit(
            model=model,
            geometry=geometry,
            observed_reflectance=observed,
            weights=weights,
        )
        all_results.append({
            "cost": float(result.objective_value),
            "params": dict(result.fitted_parameters),
        })
        if result.objective_value < best_cost:
            best_result = result
            best_cost = result.objective_value

        if (start_idx + 1) % 25 == 0:
            print(f"  [{name}] {start_idx+1}/{n_starts} starts  "
                  f"best cost so far: {best_cost:.6f}")

    if best_result is None:
        raise RuntimeError(f"{name}: no start converged")

    # Evaluate best solution
    best_model = HapkeModel(
        enable_shoe=True,
        enable_roughness=True,
        fixed_parameters=fixed_parameters,
    )
    best_model.parameters.update(best_result.fitted_parameters)
    predicted = np.asarray(best_model.reflectance(geometry)).reshape(-1)
    frms = fractional_rms(observed, predicted)

    bh = best_result.metadata.get("boundary_hits", {})
    print(f"  [{name}] BEST  cost={best_cost:.6f}  fRMS={frms*100:.3f}%  "
          f"params={best_result.fitted_parameters}  boundary={bh}")

    # Multi-start spread across all 100 runs
    free_params = list(parameter_bounds.keys())
    spread: dict[str, dict] = {}
    for pname in free_params:
        vals = np.array([r["params"][pname] for r in all_results])
        spread[pname] = {
            "mean": float(vals.mean()),
            "std":  float(vals.std()),
            "min":  float(vals.min()),
            "max":  float(vals.max()),
        }

    # Print spread for immediate inspection
    print(f"  [{name}] Multi-start spread ({n_starts} runs):")
    for pname, s in spread.items():
        print(f"    {pname:>10}: mean={s['mean']:.5f}  std={s['std']:.5f}  "
              f"min={s['min']:.5f}  max={s['max']:.5f}")

    return {
        "case": name,
        "n_bins": int(df.height),
        "n_starts": n_starts,
        "fixed_parameters": fixed_parameters,
        "fitted_parameters": best_result.fitted_parameters,
        "cost": float(best_result.objective_value),
        "fractional_rms_pct": frms * 100.0,
        "boundary_hits": bh,
        "success": bool(best_result.metadata.get("success", False)),
        "multistart_spread": spread,
        "metadata": {
            "parameter_errors": best_result.metadata.get("parameter_errors"),
            "reduced_chi_square": best_result.metadata.get("reduced_chi_square"),
            "nfev": best_result.metadata.get("nfev"),
            "message": best_result.metadata.get("message"),
        },
    }


def main() -> None:
    # Case 1: B0=1.03, h=0.04 fixed (Helfenstein & Veverka 1989 defaults)
    # Free: w, g, theta_bar — directly comparable to Li et al. 2013 Table 2 Case 1
    case1 = run_case(
        name="case1_B0h_fixed",
        fixed_parameters={"B0": 1.03, "h": 0.04},
        parameter_bounds={
            "w":         (0.3, 0.7),
            "g":         (-0.6, 0.0),
            "theta_bar": (1.0, 50.0),
        },
    )

    # Case 2: B0=1.03 fixed, h free
    # Free: w, g, theta_bar, h — Li et al. 2013 Table 2 Case 2
    case2 = run_case(
        name="case2_B0_fixed",
        fixed_parameters={"B0": 1.03},
        parameter_bounds={
            "w":         (0.3, 0.7),
            "g":         (-0.6, 0.0),
            "theta_bar": (1.0, 50.0),
            "h":         (0.001, 1.0),
        },
    )

    # Case 3: all five free — Li et al. 2013 Table 2 Case 3
    # Expected: B0 and h will be unconstrainable (zero bins below 5° phase).
    # B0 will likely pin at its upper bound and h will be unconstrained.
    # Report to document the unconstrainability, matching Li et al. methodology.
    case3 = run_case(
        name="case3_all_free",
        fixed_parameters={},
        parameter_bounds={
            "w":         (0.3, 0.7),
            "g":         (-0.6, 0.0),
            "theta_bar": (1.0, 50.0),
            "B0":        (0.0, 2.0),
            "h":         (0.001, 1.0),
        },
    )

    payload = {"case1": case1, "case2": case2, "case3": case3}
    print("\n" + "=" * 72)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

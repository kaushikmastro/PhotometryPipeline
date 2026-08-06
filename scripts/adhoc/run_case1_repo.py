from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from photometry.core.types import GeometryBatch
from photometry.fitting.least_sq import LeastSquaresFitter
from photometry.models.hapke import HapkeModel


ROOT = Path(__file__).resolve().parents[1]
PARQUET_PATH_DSK256 = ROOT / "data" / "silver/dsk256" / "combined_rc_survey_sample_corrected_dsk256.parquet"
GRID_SIZE = 5.0


def build_df_binned() -> pl.DataFrame:
    df_approach = (
        pl.scan_parquet(PARQUET_PATH_DSK256)
        .with_columns([
            ((pl.col("phase") / GRID_SIZE).round() * GRID_SIZE).alias("alpha_grid"),
            ((pl.col("incidence") / GRID_SIZE).round() * GRID_SIZE).alias("i_grid"),
            ((pl.col("emission") / GRID_SIZE).round() * GRID_SIZE).alias("e_grid"),
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
        .filter(pl.col("n_pixels") >= 10)
        .drop_nulls(subset=["std_iof"])
        .sort(["alpha_grid", "i_grid", "e_grid"])
        .collect()
    )

    # Match the notebook's downstream cleaning.
    df_approach = df_approach.with_columns(
        pl.when(pl.col("std_iof") == 0)
        .then(pl.col("mean_iof") * 0.01)
        .otherwise(pl.col("std_iof"))
        .alias("std_iof")
    )
    return df_approach


def main() -> None:
    df_binned = build_df_binned()
    print(f"Built df_binned with {df_binned.height:,} rows")
    print(f"Columns: {df_binned.columns}")

    geom_clean_dsk = GeometryBatch(
        incidence=np.deg2rad(df_binned["mean_incidence"].to_numpy()),
        emission=np.deg2rad(df_binned["mean_emission"].to_numpy()),
        phase=np.deg2rad(df_binned["mean_phase"].to_numpy()),
    )
    measured_iof_dsk = df_binned["mean_iof"].to_numpy()
    weights_inv_sigma_dsk = 1.0 / df_binned["std_iof"].to_numpy()

    model = HapkeModel(
        enable_shoe=True,
        enable_roughness=True,
        fixed_parameters={"B0": 1.03, "h": 0.04},
    )
    fitter = LeastSquaresFitter()

    print(f"Optimizing: {model.parameter_names()}")

    n_starts = 100
    np.random.seed(42)
    initial_guesses = [
        {
            "w": np.random.uniform(0.3, 0.7),
            "g": np.random.uniform(-0.5, -0.1),
            "theta_bar": np.random.uniform(1.0, 30.0),
        }
        for _ in range(n_starts)
    ]

    best_cost = np.inf
    best_result = None

    for guess in initial_guesses:
        model.parameters.update(guess)

        result = fitter.fit(
            model=model,
            geometry=geom_clean_dsk,
            observed_reflectance=measured_iof_dsk,
            weights=weights_inv_sigma_dsk,
        )

        if result.metadata["success"] and result.objective_value < best_cost:
            best_cost = result.objective_value
            best_result = result

    if best_result is None:
        raise RuntimeError("Optimization collapsed. No multi-start vectors converged.")

    print("Parameter fitting complete.")
    print(f"Cost: {best_result.objective_value:.4f}")
    for param_name, value in best_result.fitted_parameters.items():
        print(f"{param_name}: {value:.6f}")

    print("Parameter uncertainties (1-Sigma)")
    parameter_errors = best_result.metadata.get("parameter_errors") or {}
    for param_name in best_result.fitted_parameters:
        error = parameter_errors.get(param_name)
        if error is None:
            print(f"{param_name}: +/- nan")
        else:
            print(f"{param_name}: +/- {error:.6f}")

    print(json.dumps({"result": best_result.fitted_parameters, "metadata": best_result.metadata}, indent=2))


if __name__ == "__main__":
    main()
"""
Three-case Li et al. fit on 110825-DSK geometry.

Reads from binned_survey_110825.parquet (~949 rows, pre-aggregated by
prebin_survey_110825.py via DuckDB COPY). Never touches the 678M-row
combined parquet — that would OOM the Polars scanner.

Identical methodology to run_baseline_fit.py:
  - Survey-only, i<50°, e<50°, n>=10 px/bin, 5°×5°×5° bins (already applied)
  - 100 multi-start TRF least-squares, weights = 1/std_iof
  - Cases 1, 2, 3 as per Li et al. 2013 Table 2

Source data: data/06_silver_layer_dsk256/binned_survey_110825.parquet
             (~949 rows from Gaskell SPC Q=256, provided to NAIF 2011-08-25)
"""
from __future__ import annotations
import importlib.util, sys, json
import numpy as np
import polars as pl
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BINNED = ROOT / "data" / "06_silver_layer_dsk256" / "binned_survey_110825.parquet"
if not BINNED.exists():
    print(f"ERROR: {BINNED} not found. Run prebin_survey_110825.py first.", file=sys.stderr)
    sys.exit(1)

# Load run_baseline_fit helpers (run_case, fractional_rms, HapkeModel, etc.)
spec = importlib.util.spec_from_file_location(
    "run_baseline_fit", ROOT / "scripts" / "run_baseline_fit.py")
rbf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rbf)

# Override build_survey_loie to read the pre-binned parquet directly.
# The schema matches: mean_incidence, mean_emission, mean_phase, mean_iof, std_iof, n_pixels.
# Zero-std guard applied here to match run_baseline_fit.py lines 74-80.
def _load_prebin() -> pl.DataFrame:
    df = pl.read_parquet(str(BINNED))
    df = df.with_columns(
        pl.when(pl.col("std_iof") == 0)
        .then(pl.col("mean_iof") * 0.01)
        .otherwise(pl.col("std_iof"))
        .alias("std_iof")
    )
    return df

rbf.build_survey_loie = _load_prebin
rbf.DATA_PATH = BINNED  # for logging inside run_case

print("=" * 72)
print("THREE-CASE LI ET AL. FIT — Gaskell SPC 110825 (mission-science model)")
print(f"Pre-binned input: {BINNED}")
df_check = pl.read_parquet(str(BINNED))
print(f"Bins: {len(df_check)}  phase {df_check['mean_phase'].min():.1f}–{df_check['mean_phase'].max():.1f}°")
print("=" * 72)

case1 = rbf.run_case(
    name="case1_110825",
    fixed_parameters={"B0": 1.03, "h": 0.04},
    parameter_bounds={"w": (0.3, 0.7), "g": (-0.6, 0.0), "theta_bar": (1.0, 50.0)},
)
case2 = rbf.run_case(
    name="case2_110825",
    fixed_parameters={"B0": 1.03},
    parameter_bounds={"w": (0.3, 0.7), "g": (-0.6, 0.0), "theta_bar": (1.0, 50.0), "h": (0.001, 1.0)},
)
case3 = rbf.run_case(
    name="case3_110825",
    fixed_parameters={},
    parameter_bounds={"w": (0.3, 0.7), "g": (-0.6, 0.0), "theta_bar": (1.0, 50.0), "B0": (0.0, 2.0), "h": (0.001, 1.0)},
)

payload = {"dsk_version": "vesta_gaskell_256_110825.bds",
           "dsk_provenance": "Gaskell SPC Q=256, provided to NAIF 2011-08-25",
           "case1": case1, "case2": case2, "case3": case3}

print("\n" + "=" * 72)
print(json.dumps(payload, indent=2, sort_keys=True))

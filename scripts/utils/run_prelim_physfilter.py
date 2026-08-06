"""
Case 1 fit on preliminary DSK with TWO filter configurations.

Config A (headline): iof>0.01 brightness cut + i_mean<50, e_mean<50, n>=10
  Validated as a clean shadow filter (physical floor 0.045 >> 0.01 cut).
  This is the production fRMS number.

Config B (honest):   incidence<90 pixel-level + i_mean<50, e_mean<50, n>=10, NO iof cut
  Includes shadow-contaminated bins. Used to confirm 1% result (~26%).

Input:  data/04_geometry_tables_fast/survey/*F1B*.parquet  (~682M pixels)
Outputs:
  data/silver/dsk256/binned_prelim_iof001.parquet    (Config A)
  data/silver/dsk256/binned_prelim_physfilter.parquet (Config B)
"""
from __future__ import annotations
import json, sys, socket
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[2]
SILVER = ROOT / "data" / "silver/dsk256"
TMP    = Path("/scratch/kaushim07/duckdb_tmp_prelim_phys")

hostname = socket.getfqdn().lower()
if "login" in hostname:
    print(f"ERROR: run on compute node ({hostname})", file=sys.stderr)
    sys.exit(1)

INPUT_GLOB = str(ROOT / "data" / "04_geometry_tables_fast" / "survey" / "*F1B*.parquet")
OUT_A = SILVER / "binned_prelim_iof001.parquet"       # headline filter
OUT_B = SILVER / "binned_prelim_physfilter.parquet"   # no iof cut

TMP.mkdir(parents=True, exist_ok=True)
SILVER.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
import duckdb
import numpy as np
import polars as pl
import importlib.util

def bankers(col, d=5.0):
    return f"""
      CASE WHEN ({col}/{d}) - FLOOR({col}/{d}) = 0.5
           THEN (CASE WHEN CAST(FLOOR({col}/{d}) AS BIGINT) % 2 = 0
                      THEN FLOOR({col}/{d}) ELSE CEIL({col}/{d}) END) * {d}
           ELSE ROUND({col}/{d}) * {d}
      END"""

def prebin(con, input_glob, output, iof_cut=None, label=""):
    where_extra = "AND iof > 0.01" if iof_cut else ""
    print(f"\n[{label}] Prebinning → {output.name}  (iof_cut={iof_cut}) …")
    con.execute(f"""
    COPY (
        WITH rounded AS (
            SELECT iof, incidence, emission, phase,
                   {bankers('phase')}     AS alpha_grid,
                   {bankers('incidence')} AS i_grid,
                   {bankers('emission')}  AS e_grid
            FROM read_parquet('{input_glob}')
            WHERE iof       IS NOT NULL
              AND incidence IS NOT NULL
              AND emission  IS NOT NULL
              AND phase     IS NOT NULL
              AND incidence < 90.0
              {where_extra}
        )
        SELECT alpha_grid, i_grid, e_grid,
               AVG(incidence)   AS mean_incidence,
               AVG(emission)    AS mean_emission,
               AVG(phase)       AS mean_phase,
               AVG(iof)         AS mean_iof,
               STDDEV_SAMP(iof) AS std_iof,
               COUNT(*)         AS n_pixels
        FROM rounded
        GROUP BY 1, 2, 3
        HAVING COUNT(*)      >= 10
           AND AVG(incidence) < 50.0
           AND AVG(emission)  < 50.0
        ORDER BY 1, 2, 3
    ) TO '{output}' (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)
    n = duckdb.execute(f"SELECT COUNT(*) FROM read_parquet('{output}')").fetchone()[0]
    print(f"  Bins: {n}")
    return n

def load_bins(path):
    df = pl.read_parquet(str(path))
    return df.with_columns(
        pl.when(pl.col("std_iof") == 0)
        .then(pl.col("mean_iof") * 0.01)
        .otherwise(pl.col("std_iof"))
        .alias("std_iof")
    )

def fit_case1(df, label):
    spec = importlib.util.spec_from_file_location("rbf", ROOT / "scripts" / "run_baseline_fit.py")
    rbf  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rbf)
    rbf.build_survey_loie = lambda: df

    iof = df["mean_iof"].to_numpy()
    std = df["std_iof"].to_numpy()
    cv  = std / np.maximum(iof, 1e-9)
    n   = df["n_pixels"].to_numpy()

    print(f"\n  [{label}] bins={len(df)}  mean_iof={iof.mean():.5f}  "
          f"mean_std={std.mean():.5f}  CV>1: {(cv>1.0).sum()}")

    result = rbf.run_case(
        name=f"case1_{label}",
        fixed_parameters={"B0": 1.03, "h": 0.04},
        parameter_bounds={"w": (0.3, 0.7), "g": (-0.6, 0.0), "theta_bar": (1.0, 50.0)},
        n_starts=100,
    )
    return result

# ── Step 1: prebin both configs ───────────────────────────────────────────────
con = duckdb.connect()
con.execute(f"SET memory_limit='32GB'")
con.execute(f"SET temp_directory='{TMP}'")
con.execute("SET threads=8")

n_raw = con.execute(f"SELECT COUNT(*) FROM read_parquet('{INPUT_GLOB}')").fetchone()[0]
print(f"Input: {INPUT_GLOB}")
print(f"Raw pixels: {n_raw:,}")

n_a = prebin(con, INPUT_GLOB, OUT_A, iof_cut=True,  label="Config A: iof>0.01")
n_b = prebin(con, INPUT_GLOB, OUT_B, iof_cut=False, label="Config B: no iof cut")
con.close()

# ── Step 2: fit both configs ──────────────────────────────────────────────────
df_a = load_bins(OUT_A)
df_b = load_bins(OUT_B)

print("\n" + "=" * 70)
print("FITTING Config A (iof>0.01, headline filter)")
print("=" * 70)
res_a = fit_case1(df_a, "iof001")

print("\n" + "=" * 70)
print("FITTING Config B (no iof cut, physical filter)")
print("=" * 70)
res_b = fit_case1(df_b, "niocut")

# ── Step 3: side-by-side report ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("FULL-DATA COMPARISON — Case 1, preliminary DSK")
print("=" * 70)
fp_a = res_a["fitted_parameters"]; md_a = res_a["metadata"]
fp_b = res_b["fitted_parameters"]; md_b = res_b["metadata"]

print(f"\n  {'config':36} {'fRMS':>7} {'theta_bar':>10} {'w':>7} {'g':>8} {'chi2':>7} {'n_bins':>7}")
print(f"  {'A: iof>0.01 (headline, validated shadow cut)':36} "
      f"{res_a['fractional_rms_pct']:>7.3f} {fp_a['theta_bar']:>10.4f} "
      f"{fp_a['w']:>7.4f} {fp_a['g']:>8.4f} "
      f"{md_a.get('reduced_chi_square',float('nan')):>7.4f} {n_a:>7}")
print(f"  {'B: no iof cut (shadow-retained, reference)':36} "
      f"{res_b['fractional_rms_pct']:>7.3f} {fp_b['theta_bar']:>10.4f} "
      f"{fp_b['w']:>7.4f} {fp_b['g']:>8.4f} "
      f"{md_b.get('reduced_chi_square',float('nan')):>7.4f} {n_b:>7}")
print(f"  {'1% sample, iof>0.01 (archived baseline)':36} "
      f"{'10.135':>7} {'5.496':>10} {'0.4626':>7} {'-0.3323':>8} {'0.1857':>7} {'949':>7}")
print(f"  {'1% sample, no iof cut':36} "
      f"{'26.442':>7} {'3.314':>10} {'0.3813':>7} {'-0.4003':>8} {'0.1188':>7} {'934':>7}")

payload = {
    "config_A_iof001": res_a,
    "config_B_niocut": res_b,
    "note": "Full-data (682M pixels) prelim DSK. Config A = headline. Config B = reference.",
}
print("\n" + "=" * 70)
print(json.dumps(payload, indent=2, sort_keys=True))

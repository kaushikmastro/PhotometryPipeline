"""
Cross-check: DuckDB prebin vs Polars build_survey_loie on IDENTICAL input.

Uses the ORIGINAL preliminary-model geometry parquet
(combined_rc_survey_sample_corrected_dsk256.parquet, 7.5M rows)
which has already been validated by the established fit.

Two methods, same filters, same 5x5x5 grid:
  (a) Polars path — exact build_survey_loie() code from run_baseline_fit.py
  (b) DuckDB path — ROUND()-based GROUP BY as used in prebin_survey_110825.py

Reports:
  - Whether any pixels fall at exact half-degree-multiple boundaries
    (the only case where the rounding conventions diverge)
  - Bin-by-bin max absolute difference on alpha_grid, i_grid, e_grid,
    mean_iof, std_iof, n_pixels for matched bins
  - Bins present in one method but not the other (assignment divergence)
  - VERDICT: IDENTICAL or DIVERGE
"""
from __future__ import annotations
import importlib.util, os, sys
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import polars as pl
import duckdb

DATA = ROOT / "data" / "06_silver_layer_dsk256" / \
       "combined_rc_survey_sample_corrected_dsk256.parquet"

INC_MAX     = 50.0
EMI_MAX     = 50.0
MIN_PIXELS  = 10
GRID        = 5.0
TOL         = 1e-8       # required agreement on floating-point columns

print(f"Input: {DATA}")
print(f"Filters: Survey-only, i<{INC_MAX}, e<{EMI_MAX}, n>={MIN_PIXELS}, grid={GRID}°")

# ── METHOD A: Polars (build_survey_loie verbatim) ────────────────────────────
print("\n── Method A: Polars (exact build_survey_loie path) ──────────────────")

spec = importlib.util.spec_from_file_location(
    "run_baseline_fit", ROOT / "scripts" / "run_baseline_fit.py")
rbf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rbf)
rbf.DATA_PATH = DATA

df_pol = rbf.build_survey_loie()
print(f"  Bins: {len(df_pol)}")
print(f"  phase range: {df_pol['mean_phase'].min():.2f}–{df_pol['mean_phase'].max():.2f}°")
print(f"  mean_iof: {df_pol['mean_iof'].mean():.6f}")

# ── METHOD B: DuckDB (same logic as prebin_survey_110825.py) ─────────────────
print("\n── Method B: DuckDB ROUND()-based GROUP BY ───────────────────────────")

con = duckdb.connect()
df_duck = con.execute(f"""
    SELECT
        ROUND(phase     / {GRID}) * {GRID}  AS alpha_grid,
        ROUND(incidence / {GRID}) * {GRID}  AS i_grid,
        ROUND(emission  / {GRID}) * {GRID}  AS e_grid,
        AVG(incidence)   AS mean_incidence,
        AVG(emission)    AS mean_emission,
        AVG(phase)       AS mean_phase,
        AVG(iof)         AS mean_iof,
        STDDEV_SAMP(iof) AS std_iof,
        COUNT(*)         AS n_pixels
    FROM read_parquet('{DATA}')
    WHERE mission_phase = 'SURVEY'
      AND iof       IS NOT NULL
      AND incidence IS NOT NULL
      AND emission  IS NOT NULL
      AND phase     IS NOT NULL
    GROUP BY 1, 2, 3
    HAVING COUNT(*)      >= {MIN_PIXELS}
       AND AVG(incidence) < {INC_MAX}
       AND AVG(emission)  < {EMI_MAX}
    ORDER BY 1, 2, 3
""").pl()   # returns Polars DataFrame
con.close()

# Apply the same zero-std guard as build_survey_loie
df_duck = df_duck.with_columns(
    pl.when(pl.col("std_iof") == 0)
    .then(pl.col("mean_iof") * 0.01)
    .otherwise(pl.col("std_iof"))
    .alias("std_iof")
)

print(f"  Bins: {len(df_duck)}")
print(f"  phase range: {df_duck['mean_phase'].min():.2f}–{df_duck['mean_phase'].max():.2f}°")
print(f"  mean_iof: {df_duck['mean_iof'].mean():.6f}")

# ── HALF-BOUNDARY CHECK ──────────────────────────────────────────────────────
print("\n── Half-boundary pixel count (the only source of rounding divergence) ─")
# A pixel can only cause divergence if phase/5, incidence/5, or emission/5
# is EXACTLY 0.5 mod 1 in float32. Check the raw data.
raw = (
    pl.scan_parquet(str(DATA))
    .filter(pl.col("mission_phase") == "SURVEY")
    .select(["phase", "incidence", "emission"])
    .collect()
)
for col in ["phase", "incidence", "emission"]:
    vals = raw[col].to_numpy()
    scaled = vals / GRID
    # float32 exact half-integer: fractional part == 0.5 exactly
    frac = scaled - np.floor(scaled)
    n_half = int((frac == 0.5).sum())
    print(f"  {col}: {n_half} pixels with x/{GRID} fractional part == 0.5 exactly")

# ── BIN SET COMPARISON ───────────────────────────────────────────────────────
print("\n── Bin assignment comparison ────────────────────────────────────────────")

key_pol  = set(zip(df_pol ["alpha_grid"].to_list(),
                   df_pol ["i_grid"].to_list(),
                   df_pol ["e_grid"].to_list()))
key_duck = set(zip(df_duck["alpha_grid"].to_list(),
                   df_duck["i_grid"].to_list(),
                   df_duck["e_grid"].to_list()))

only_pol  = key_pol  - key_duck
only_duck = key_duck - key_pol
common    = key_pol  & key_duck

print(f"  Bins in Polars only:  {len(only_pol)}")
print(f"  Bins in DuckDB only:  {len(only_duck)}")
print(f"  Bins in both:         {len(common)}")

if only_pol:
    print(f"  Polars-only bins: {sorted(only_pol)[:10]}")
if only_duck:
    print(f"  DuckDB-only bins: {sorted(only_duck)[:10]}")

# ── VALUE COMPARISON ON MATCHED BINS ─────────────────────────────────────────
print("\n── Value comparison on matched bins ─────────────────────────────────────")

df_pol_s  = df_pol .sort(["alpha_grid", "i_grid", "e_grid"])
df_duck_s = df_duck.sort(["alpha_grid", "i_grid", "e_grid"])

# Join on grid keys
joined = df_pol_s.join(
    df_duck_s,
    on=["alpha_grid", "i_grid", "e_grid"],
    how="inner",
    suffix="_duck"
)
print(f"  Matched bins for value comparison: {len(joined)}")

cols_to_check = {
    "mean_incidence": ("mean_incidence", "mean_incidence_duck"),
    "mean_emission":  ("mean_emission",  "mean_emission_duck"),
    "mean_phase":     ("mean_phase",     "mean_phase_duck"),
    "mean_iof":       ("mean_iof",       "mean_iof_duck"),
    "std_iof":        ("std_iof",        "std_iof_duck"),
    "n_pixels":       ("n_pixels",       "n_pixels_duck"),
}

all_match = True
print(f"\n  {'column':>16}  {'max |diff|':>12}  {'mean |diff|':>12}  {'pass (< {:.0e})'.format(TOL):>18}")
for label, (c_pol, c_duck) in cols_to_check.items():
    a = joined[c_pol].to_numpy().astype(float)
    b = joined[c_duck].to_numpy().astype(float)
    # n_pixels: integer comparison
    if label == "n_pixels":
        diff = np.abs(a - b)
        max_d = float(diff.max())
        mean_d = float(diff.mean())
        ok = max_d == 0
        print(f"  {label:>16}  {max_d:>12.0f}  {mean_d:>12.6f}  {'PASS' if ok else 'FAIL':>18}")
    else:
        diff = np.abs(a - b)
        max_d = float(diff.max())
        mean_d = float(diff.mean())
        ok = max_d < TOL
        print(f"  {label:>16}  {max_d:>12.2e}  {mean_d:>12.2e}  {'PASS' if ok else f'FAIL (>{TOL:.0e})':>18}")
    if not ok:
        all_match = False
        # Show worst offending bins
        worst_idx = int(np.argmax(diff))
        row = joined[worst_idx]
        print(f"    Worst bin: α={row['alpha_grid'].item():.0f} i={row['i_grid'].item():.0f} "
              f"e={row['e_grid'].item():.0f} | Polars={a[worst_idx]:.10f} DuckDB={b[worst_idx]:.10f}")

# ── VERDICT ──────────────────────────────────────────────────────────────────
print(f"\n{'='*68}")
print("VERDICT")
print("=" * 68)

bin_mismatch = len(only_pol) + len(only_duck)
if all_match and bin_mismatch == 0:
    print(f"\n  IDENTICAL: DuckDB prebin and Polars build_survey_loie produce")
    print(f"  the same {len(common)} bins with max |diff| < {TOL:.0e} on all columns.")
    print(f"  The DuckDB binning path is confirmed neutral.")
    print(f"  The theta_bar difference between 110825 and preliminary fits")
    print(f"  is attributable to the shape model change alone.")
elif bin_mismatch > 0 and all_match:
    print(f"\n  PARTIAL MISMATCH: {bin_mismatch} bins differ in assignment.")
    print(f"  Values agree on shared bins, but bin set differs.")
    print(f"  Fix: align DuckDB ROUND() to Polars banker's rounding before")
    print(f"  accepting any 110825 fit result.")
else:
    print(f"\n  DIVERGE: {bin_mismatch} bin assignment differences AND/OR")
    print(f"  value differences > {TOL:.0e}. DuckDB path is NOT neutral.")
    print(f"  Do not interpret 110825 fit until binning is reconciled.")

"""
Controlled shape-model comparison: pre-Dawn preliminary DSK vs 110825 mission-science DSK.

CONTROLLED means: identical treatment on both shape models —
  - Same image set: Survey F1B (845 images, the common set in both geometry tables)
  - Same sampling: 1% Bernoulli, seed=42
  - Same binning: DuckDB with banker's rounding (round-half-to-even, matching Polars)
  - Same filters: Survey-only, i<50°, e<50°, n>=10, 5°×5°×5° grid
  - ONLY variable: which .bds file was loaded when the geometry was computed

Sources:
  Preliminary DSK: data/04_geometry_tables_fast/survey/*F1B*.parquet  (f_solar=892)
  110825 DSK:      data/geometry/dsk256/survey/*.parquet  (f_solar=892)

Outputs:
  data/silver/dsk256/binned_prelim_1pct.parquet
  data/silver/dsk256/binned_110825_1pct.parquet

Also produces the Step 3 characterisation: per-bin local-incidence spread
for the FULL 110825 dataset (all 285M pixels), to quantify the within-bin
scatter finding without applying a filter.

Step 3 output:
  data/silver/dsk256/scatter_characterisation_110825.parquet
"""
import socket, sys, os
from pathlib import Path
import duckdb

ROOT   = Path(__file__).resolve().parents[2]
SILVER = ROOT / "data" / "silver/dsk256"
SILVER.mkdir(parents=True, exist_ok=True)
TMP    = Path("/scratch/kaushim07/duckdb_tmp_ctrl")
TMP.mkdir(parents=True, exist_ok=True)

hostname = socket.getfqdn().lower()
if "login" in hostname:
    print(f"ERROR: refusing to run on login node ({hostname}).", file=sys.stderr)
    sys.exit(1)

PRELIM_GLOB = str(ROOT / "data" / "04_geometry_tables_fast" / "survey" / "*F1B*.parquet")
D110_GLOB   = str(ROOT / "data" / "geometry/dsk256" / "survey" / "*.parquet")

PRELIM_OUT  = SILVER / "binned_prelim_1pct.parquet"
D110_OUT    = SILVER / "binned_110825_1pct.parquet"
SCATTER_OUT = SILVER / "scatter_characterisation_110825.parquet"

SAMPLE_SEED = 42
SAMPLE_PCT  = 1   # Bernoulli 1%

# Banker's-rounding macro for DuckDB (matches Polars .round() round-half-to-even)
def bankers_round_expr(col: str, divisor: float = 5.0) -> str:
    return f"""
        CASE
            WHEN ({col} / {divisor}) - FLOOR({col} / {divisor}) = 0.5
                THEN (CASE WHEN CAST(FLOOR({col} / {divisor}) AS BIGINT) % 2 = 0
                           THEN FLOOR({col} / {divisor})
                           ELSE CEIL({col} / {divisor})
                      END) * {divisor}
            ELSE ROUND({col} / {divisor}) * {divisor}
        END"""

BANKERS_ALPHA = bankers_round_expr("phase")
BANKERS_I     = bankers_round_expr("incidence")
BANKERS_E     = bankers_round_expr("emission")

PREBIN_SQL = """
    WITH sampled AS (
        SELECT phase, incidence, emission, iof
        FROM read_parquet('{glob}')
        WHERE iof IS NOT NULL AND incidence IS NOT NULL
          AND emission IS NOT NULL AND phase IS NOT NULL
        USING SAMPLE {pct}% (bernoulli, {seed})
    ),
    rounded AS (
        SELECT iof, incidence, emission, phase,
               {alpha} AS alpha_grid,
               {i_gr}  AS i_grid,
               {e_gr}  AS e_grid
        FROM sampled
    )
    SELECT
        alpha_grid, i_grid, e_grid,
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
""".format(
    glob="{glob}", pct=SAMPLE_PCT, seed=SAMPLE_SEED,
    alpha=BANKERS_ALPHA, i_gr=BANKERS_I, e_gr=BANKERS_E
)

# Step 3 scatter characterisation — full 285M pixels, adds STD(incidence) per bin
SCATTER_SQL = f"""
    WITH rounded AS (
        SELECT iof, incidence, emission, phase,
               {BANKERS_ALPHA} AS alpha_grid,
               {BANKERS_I}     AS i_grid,
               {BANKERS_E}     AS e_grid
        FROM read_parquet('{D110_GLOB}')
        WHERE iof IS NOT NULL AND incidence IS NOT NULL
          AND emission IS NOT NULL AND phase IS NOT NULL
    )
    SELECT
        alpha_grid, i_grid, e_grid,
        AVG(incidence)          AS mean_incidence,
        STDDEV_SAMP(incidence)  AS std_incidence,   -- within-bin local-slope scatter
        AVG(emission)           AS mean_emission,
        STDDEV_SAMP(emission)   AS std_emission,
        AVG(phase)              AS mean_phase,
        AVG(iof)                AS mean_iof,
        STDDEV_SAMP(iof)        AS std_iof,
        COUNT(*)                AS n_pixels
    FROM rounded
    GROUP BY 1, 2, 3
    HAVING COUNT(*)      >= 10
       AND AVG(incidence) < 50.0
       AND AVG(emission)  < 50.0
    ORDER BY 1, 2, 3
"""

con = duckdb.connect()
con.execute(f"SET memory_limit='16GB'")
con.execute(f"SET temp_directory='{TMP}'")
con.execute("SET threads=4")

print("=" * 68)
print("CONTROLLED SHAPE-MODEL COMPARISON")
print(f"  Sampling: {SAMPLE_PCT}% Bernoulli, seed={SAMPLE_SEED}")
print(f"  Images:   Survey F1B (both DSK sources)")
print(f"  Binning:  banker's rounding, DuckDB COPY")
print(f"  Filters:  i<50°, e<50°, n>=10")
print(f"  Variable: ONLY the .bds file used during geometry")
print("=" * 68)

# ── Pre-Dawn preliminary DSK (fast tables, F1B only) ─────────────────────────
print(f"\nStep 1a: Prebinning preliminary DSK (1% sample from fast F1B)...")
n_prelim = con.execute(f"SELECT COUNT(*) FROM read_parquet('{PRELIM_GLOB}')").fetchone()[0]
print(f"  Total pixels available: {n_prelim:,}")

prelim_sql = PREBIN_SQL.format(glob=PRELIM_GLOB)
con.execute(f"COPY ({prelim_sql}) TO '{PRELIM_OUT}' (FORMAT PARQUET, COMPRESSION SNAPPY)")
n_bins_prelim = con.execute(f"SELECT COUNT(*) FROM read_parquet('{PRELIM_OUT}')").fetchone()[0]
print(f"  Output bins: {n_bins_prelim}  → {PRELIM_OUT.name}")

# ── 110825 mission-science DSK ────────────────────────────────────────────────
print(f"\nStep 1b: Prebinning 110825 DSK (1% sample from dsk256_110825)...")
n_110 = con.execute(f"SELECT COUNT(*) FROM read_parquet('{D110_GLOB}')").fetchone()[0]
print(f"  Total pixels available: {n_110:,}")

d110_sql = PREBIN_SQL.format(glob=D110_GLOB)
con.execute(f"COPY ({d110_sql}) TO '{D110_OUT}' (FORMAT PARQUET, COMPRESSION SNAPPY)")
n_bins_110 = con.execute(f"SELECT COUNT(*) FROM read_parquet('{D110_OUT}')").fetchone()[0]
print(f"  Output bins: {n_bins_110}  → {D110_OUT.name}")

# ── Step 3: Scatter characterisation (full 110825 data) ──────────────────────
print(f"\nStep 3: Scatter characterisation on full 110825 dataset (285M pixels)...")
con.execute(f"COPY ({SCATTER_SQL}) TO '{SCATTER_OUT}' (FORMAT PARQUET, COMPRESSION SNAPPY)")
n_scatter = con.execute(f"SELECT COUNT(*) FROM read_parquet('{SCATTER_OUT}')").fetchone()[0]
print(f"  Output bins: {n_scatter}  → {SCATTER_OUT.name}")

con.close()
print("\nAll prebin outputs written. Run run_fit_controlled.py next.")
print(f"  {PRELIM_OUT}")
print(f"  {D110_OUT}")
print(f"  {SCATTER_OUT}")

"""
Pre-aggregate 678M-row Survey F1B geometry into ~949 binned cubes.

Reads directly from the 845 per-image geometry parquets via DuckDB
(avoids loading the 18GB combined parquet through Polars).

Output: data/silver/dsk256/binned_survey_110825.parquet
        ~949 rows matching the schema of build_survey_loie() output
        (mean_incidence, mean_emission, mean_phase, mean_iof, std_iof, n_pixels,
         alpha_grid, i_grid, e_grid)

Fit scripts load this tiny parquet directly — no Polars scan of raw pixels.
"""
import socket, sys, os
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[2]
INPUT_GLOB = str(ROOT / "data" / "geometry/dsk256" / "survey" / "*.parquet")
OUTPUT    = ROOT / "data" / "silver/dsk256" / "binned_survey_110825.parquet"
TMP_DIR   = Path("/scratch/kaushim07/duckdb_tmp_110825")

hostname = socket.getfqdn().lower()
if "login" in hostname:
    print(f"ERROR: refusing to run on login node ({hostname}). Use sbatch.", file=sys.stderr)
    sys.exit(1)

TMP_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

print(f"Input:  {INPUT_GLOB}")
print(f"Output: {OUTPUT}")
print(f"Tmp:    {TMP_DIR}")

con = duckdb.connect()
# Enable disk spill so DuckDB can handle 678M rows safely
con.execute(f"SET memory_limit='16GB'")
con.execute(f"SET temp_directory='{TMP_DIR}'")
con.execute("SET threads=4")

# Count input rows first (lightweight scan)
n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{INPUT_GLOB}')").fetchone()[0]
print(f"Input rows: {n:,}")

# Aggregate: identical to build_survey_loie() in run_baseline_fit.py
# Grid: ROUND(x/5)*5 matches polars ((col/5).round()*5) behaviour
print("Aggregating into 5°×5°×5° bins …")
con.execute(f"""
    COPY (
        -- Banker's rounding macro: matches Polars .round() (round-half-to-even).
        -- DuckDB ROUND() is round-half-away-from-zero; at x/5 = N.5 the two
        -- conventions assign different bins. The CASE expression below replicates
        -- IEEE 754 round-half-to-even so bin assignments are byte-identical to
        -- build_survey_loie() in run_baseline_fit.py.
        --
        -- Formula: if fractional part == 0.5, round to nearest EVEN integer;
        -- otherwise standard round. Applied separately to each angle column.
        WITH rounded AS (
            SELECT
                iof, incidence, emission, phase,
                -- alpha_grid
                CASE
                    WHEN (phase / 5.0) - FLOOR(phase / 5.0) = 0.5
                        THEN (CASE WHEN CAST(FLOOR(phase / 5.0) AS BIGINT) % 2 = 0
                                   THEN FLOOR(phase / 5.0)
                                   ELSE CEIL(phase / 5.0)
                              END) * 5.0
                    ELSE ROUND(phase / 5.0) * 5.0
                END AS alpha_grid,
                -- i_grid
                CASE
                    WHEN (incidence / 5.0) - FLOOR(incidence / 5.0) = 0.5
                        THEN (CASE WHEN CAST(FLOOR(incidence / 5.0) AS BIGINT) % 2 = 0
                                   THEN FLOOR(incidence / 5.0)
                                   ELSE CEIL(incidence / 5.0)
                              END) * 5.0
                    ELSE ROUND(incidence / 5.0) * 5.0
                END AS i_grid,
                -- e_grid
                CASE
                    WHEN (emission / 5.0) - FLOOR(emission / 5.0) = 0.5
                        THEN (CASE WHEN CAST(FLOOR(emission / 5.0) AS BIGINT) % 2 = 0
                                   THEN FLOOR(emission / 5.0)
                                   ELSE CEIL(emission / 5.0)
                              END) * 5.0
                    ELSE ROUND(emission / 5.0) * 5.0
                END AS e_grid
            FROM read_parquet('{INPUT_GLOB}')
            WHERE iof       IS NOT NULL
              AND incidence IS NOT NULL
              AND emission  IS NOT NULL
              AND phase     IS NOT NULL
        )
        SELECT
            alpha_grid,
            i_grid,
            e_grid,
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
    ) TO '{OUTPUT}' (FORMAT PARQUET, COMPRESSION SNAPPY)
""")
con.close()

n_bins = duckdb.execute(f"SELECT COUNT(*) FROM read_parquet('{OUTPUT}')").fetchone()[0]
size   = os.path.getsize(OUTPUT)
print(f"Output bins: {n_bins}  (expected ~949)")
print(f"Output size: {size:,} bytes")
print("Done.")

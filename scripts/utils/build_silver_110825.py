"""
Build combined_survey_sample_110825.parquet from the 110825-DSK geometry tables.

Reads all Survey F1B geometry parquets from
  data/geometry/dsk256/survey/
and combines them (no sampling — full dataset) into
  data/silver/dsk256/combined_survey_sample_110825.parquet

Adds mission_phase = 'SURVEY' column to match the schema expected by
run_baseline_fit.py / build_survey_loie().
"""
import socket, sys
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[2]
INPUT_GLOB = str(ROOT / "data" / "geometry/dsk256" / "survey" / "*.parquet")
OUTPUT    = ROOT / "data" / "silver/dsk256" / "combined_survey_sample_110825.parquet"

hostname = socket.getfqdn().lower()
if "login" in hostname:
    print(f"ERROR: refusing to run on login node ({hostname}). Use srun/sbatch.", file=sys.stderr)
    sys.exit(1)

print(f"Reading Survey F1B geometry from: {INPUT_GLOB}")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()

# Row count and sanity check without materialising
n_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{INPUT_GLOB}')").fetchone()[0]
iof_range = con.execute(f"""
    SELECT MIN(iof), MAX(iof)
    FROM read_parquet('{INPUT_GLOB}')
    WHERE iof IS NOT NULL
""").fetchone()
print(f"  Rows: {n_rows:,}")
print(f"  iof range: {iof_range[0]:.4f} – {iof_range[1]:.4f}")
print(f"  Saving to: {OUTPUT}")

# Write directly via DuckDB native COPY — never materialises in Python RAM
con.execute(f"""
    COPY (
        SELECT
            image_id,
            'SURVEY'::VARCHAR AS mission_phase,
            pixel_x,
            pixel_y,
            CAST(iof       AS FLOAT)  AS iof,
            CAST(incidence AS FLOAT)  AS incidence,
            CAST(emission  AS FLOAT)  AS emission,
            CAST(phase     AS FLOAT)  AS phase,
            CAST(latitude  AS FLOAT)  AS latitude,
            CAST(longitude AS FLOAT)  AS longitude,
            CAST(phase     AS DOUBLE) AS mean_phase
        FROM read_parquet('{INPUT_GLOB}')
        WHERE iof       IS NOT NULL
          AND incidence IS NOT NULL
          AND emission  IS NOT NULL
          AND phase     IS NOT NULL
    ) TO '{OUTPUT}' (FORMAT PARQUET, COMPRESSION SNAPPY)
""")
con.close()

import os; print(f"  Written: {os.path.getsize(OUTPUT):,} bytes")
print("Done.")

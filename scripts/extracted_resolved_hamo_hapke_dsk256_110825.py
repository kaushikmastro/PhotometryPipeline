"""
Usage:
    python scripts/extract_resolved_hamo_hapke_dsk256_110825.py
Requirements:
    - duckdb
    - pyarrow
Input parquet patterns:
    data/04_geometry_tables_dsk256_110825/hamo/*.parquet  ← f_solar=892 (CORRECT)
Output parquet:
    data/06_silver_layer_dsk256/hamo_dsk256_110825.parquet
"""
from pathlib import Path
import socket
import sys
import time
import duckdb

INPUT_GLOBS = ['data/04_geometry_tables_dsk256_110825/hamo/*.parquet']
OUTPUT_DIR = Path('data/06_silver_layer_dsk256')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / 'hamo_dsk256_110825.parquet'

def guard_against_login_node() -> None:
    hostname = socket.getfqdn().lower()
    if 'login' in hostname or hostname == 'login.curta.zedat.fu-berlin.de':
        print('============================================================', file=sys.stderr)
        print(f'WARNING: refusing to run on shared login node: {hostname}', file=sys.stderr)
        print('Submit via SLURM/srun to a compute node instead.', file=sys.stderr)
        print('============================================================', file=sys.stderr)
        sys.exit(1)

def main() -> None:
    guard_against_login_node()

    con = duckdb.connect()
    con.execute("SET memory_limit='24GB'")
    con.execute("SET threads=8")
    con.execute("SET temp_directory='/scratch/kaushim07/duckdb_tmp'")

    sql = """
    SELECT
            image_id,
            pixel_x,
            pixel_y,
            iof,
            incidence,
            emission,
            phase,
            latitude,
            longitude,
            'hamo' AS mission_phase
        FROM read_parquet('data/04_geometry_tables_dsk256_110825/hamo/*.parquet')
        WHERE incidence < 80.0
          AND emission < 80.0
          AND iof > 0.0156 * COS(RADIANS(incidence))
          AND image_id LIKE '%F1B%'
    """
    print('Running DuckDB query to extract sample...')
    start_time = time.time()
    print(f'Start time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))}')
    try:
        copy_sql = f"COPY ({sql}) TO '{OUTPUT_PATH.as_posix()}' (FORMAT PARQUET, COMPRESSION 'ZSTD')"
        con.execute(copy_sql)
    except Exception as exc:
        print(f'\nDuckDB extraction failed: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    print(f'End time:   {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))}')
    print(f'Total elapsed: {int(minutes)}m {seconds:.2f}s')
    print(f'Success! Wrote sample to: {OUTPUT_PATH}')

    print('\nVerification: calculating summary stats out-of-core...')
    verify_sql = f"""
    SELECT 
        mission_phase,
        COUNT(*) AS row_count,
        AVG(phase) AS average_phase
    FROM read_parquet('{OUTPUT_PATH.as_posix()}')
    GROUP BY mission_phase
    """
    summary_df = con.execute(verify_sql).df()
    if summary_df.empty or summary_df['row_count'].iloc[0] == 0:
        print('Warning: output file is empty. No summary to report.')
        return
    print('\nSummary by mission_phase:')
    print(summary_df.to_string(index=False, formatters={'average_phase': '{:.6f}'.format}))

if __name__ == '__main__':
    main()
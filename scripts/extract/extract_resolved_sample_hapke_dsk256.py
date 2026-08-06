"""
Extract a 1% random sample of disk-resolved photometric data from RC and Survey
parquet files for Hapke fitting.

Usage:
	python scripts/extract_resolved_sample_hapke_dsk256.py

Requirements:
	- duckdb
	- pandas
	- pyarrow

Input parquet patterns:
	data/04_geometry_tables_fast/rc/*.parquet      ← f_solar=892 (CORRECT)
	data/04_geometry_tables_fast/survey/*.parquet  ← f_solar=892 (CORRECT)

Output parquet:
	data/silver/dsk256/combined_rc_survey_sample_dsk256.parquet
"""

from pathlib import Path
import socket
import sys
import time

import duckdb
import pandas as pd

INPUT_GLOBS = [
	'data/04_geometry_tables_fast/rc/*.parquet',
	'data/04_geometry_tables_fast/survey/*.parquet',
]
OUTPUT_DIR = Path('data/silver/dsk256')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / 'combined_rc_survey_sample_corrected_dsk256.parquet'


def guard_against_login_node() -> None:
	hostname = socket.getfqdn().lower()
	if 'login' in hostname or hostname == 'login.curta.zedat.fu-berlin.de':
		print(
			'============================================================',
			file=sys.stderr,
		)
		print(
			f'WARNING: refusing to run on shared login node: {hostname}',
			file=sys.stderr,
		)
		print(
			'Submit this script "$ srun --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G --partition=main --qos=standard --time=04:00:00 --pty bash" to a compute node or batch job instead.',
			file=sys.stderr,
		)
		print(
			'============================================================',
			file=sys.stderr,
		)
		sys.exit(1)


def main() -> None:
	guard_against_login_node()

	sql = """
        WITH filtered_rc AS (
        -- Select ONLY required columns to trigger Parquet projection pushdown
        SELECT 
            image_id, pixel_x, pixel_y, iof, 
            incidence, emission, phase, latitude, longitude 
        FROM read_parquet('data/04_geometry_tables_fast/rc/*.parquet')
        WHERE incidence < 80.0 AND emission < 80.0 AND iof > 0.01
        USING SAMPLE 1% (bernoulli)
    ),
    filtered_survey AS (
        SELECT 
            image_id, pixel_x, pixel_y, iof, 
            incidence, emission, phase, latitude, longitude 
        FROM read_parquet('data/04_geometry_tables_fast/survey/*.parquet')
        WHERE incidence < 80.0 AND emission < 80.0 AND iof > 0.01
        USING SAMPLE 1% (bernoulli)
    ),
    combined_source AS (
        SELECT *, 'RC' AS mission_phase FROM filtered_rc
        UNION ALL
        SELECT *, 'SURVEY' AS mission_phase FROM filtered_survey
    )
    SELECT
        image_id,
        mission_phase,
        pixel_x,
        pixel_y,
        iof,
        incidence,
        emission,
        phase,
        latitude,
        longitude,
        AVG(phase) OVER (PARTITION BY image_id) AS mean_phase
    FROM combined_source
	
	"""

	print('Running DuckDB query to extract sample...')
	con = duckdb.connect(database=':memory:')
	con.execute('PRAGMA enable_progress_bar;')

	start_time = time.time()
	print(f'Start time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))}')

	try:
		copy_sql = f"COPY ({sql}) TO '{OUTPUT_PATH.as_posix()}' (FORMAT PARQUET)"
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

	print('\nVerification: loading saved sample for summary stats...')
	df = pd.read_parquet(OUTPUT_PATH)

	row_count = len(df)
	print(f'Row count: {row_count:,}')

	if row_count == 0:
		print('Warning: output file is empty. No summary to report.')
		return

	summary = (
		df.groupby('mission_phase', dropna=False)
		.agg(row_count=('image_id', 'size'), average_phase=('phase', 'mean'))
		.reset_index()
		.sort_values('mission_phase')
	)

	print('\nSummary by mission_phase:')
	print(summary.to_string(index=False, formatters={'average_phase': '{:.6f}'.format}))


if __name__ == '__main__':
	main()

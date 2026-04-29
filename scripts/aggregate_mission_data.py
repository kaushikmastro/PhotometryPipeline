#!/usr/bin/env python3
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import duckdb

PHASES = ["survey", "rc", "hamo", "lamo"]

PHASE_CONFIG = {
    "rc": {"emission_cut": 75, "bin_width": 0.5},
    "survey": {"emission_cut": 60, "bin_width": 1.0},
    "hamo": {"emission_cut": 60, "bin_width": 1.0},
    "lamo": {"emission_cut": 60, "bin_width": 1.0},
}


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "05_aggregated"
    output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA enable_progress_bar;")

    overall_start_time = time.time()

    for phase in PHASES:
        config = PHASE_CONFIG[phase]
        phase_dir = project_root / "data" / "04_geometry_tables" / phase
        parquet_files = sorted(phase_dir.glob("*.parquet")) if phase_dir.exists() else []

        if not phase_dir.exists():
            print(f"[SKIP] Phase '{phase}': directory not found at {phase_dir}")
            continue

        if not parquet_files:
            print(f"[SKIP] Phase '{phase}': no parquet files found in {phase_dir}")
            continue

        start_time = time.time()
        phase_glob = str(phase_dir / "*.parquet")
        output_path = output_dir / f"{phase}_phase_curve.csv"

        print(
            f"[START] Aggregating phase '{phase}' from {len(parquet_files):,} files: {phase_glob}"
        )

        current_time = datetime.now().strftime("%H:%M:%S")
        print(
            f"[START] {current_time} | Aggregating {phase.upper()} from {len(parquet_files):,} files..."
        )
        query = f"""
        COPY (
            SELECT
                CASE
                    WHEN phase < 15 THEN ROUND(phase / {config['bin_width']}) * {config['bin_width']}
                    ELSE ROUND(phase)
                END AS phase_bin_deg,
                COUNT(*)::BIGINT AS n_pixels,
                APPROX_QUANTILE(iof, 0.50)::DOUBLE AS median_iof,
                APPROX_QUANTILE(iof, 0.25)::DOUBLE AS iof_q25,
                APPROX_QUANTILE(iof, 0.75)::DOUBLE AS iof_q75,
                AVG(incidence)::DOUBLE AS mean_incidence,
                AVG(emission)::DOUBLE AS mean_emission,
                AVG(phase)::DOUBLE AS mean_phase,
                STDDEV_SAMP(phase)::DOUBLE AS stddev_phase
            FROM read_parquet('{phase_glob}')
            WHERE incidence < 90
              AND emission < {config['emission_cut']}
              AND iof > 0.0
            GROUP BY 1
            ORDER BY 1
        ) TO '{output_path}' (HEADER, DELIMITER ',');
        """
        con.execute(query)
        con = duckdb.connect(database=":memory:")  # Reconnect to ensure we read the latest file system state
        con.execute("PRAGMA enable_progress_bar;")

        if output_path.exists():
            rows_written = con.execute(
                f"""
            SELECT COUNT(*)
            FROM read_csv_auto('{output_path}')"""
            ).fetchone()[0]

            elapsed = time.time() - start_time
            print(
                f"[DONE] Phase '{phase}': wrote {rows_written:,} bins to {output_path} in {elapsed:.1f} seconds"
            )
        else:
            print(
                f"[ERROR] Phase '{phase}': query completed but no output file was created. Check if the filter conditions are too strict."
            )
    con.close()
    total_elapsed = time.time() - overall_start_time
    print(f"Total processing time: {total_elapsed:.1f} seconds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import duckdb

PHASES = ["survey", "rc", "hamo", "lamo"]


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "05_aggregated"
    output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")

    for phase in PHASES:
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

        query = f"""
        COPY (
            SELECT
                ROUND(phase)::INTEGER AS phase_bin_deg,
                COUNT(*)::BIGINT AS pixel_count,
                AVG(iof)::DOUBLE AS mean_iof,
                AVG(incidence)::DOUBLE AS mean_incidence,
                AVG(emission)::DOUBLE AS mean_emission
            FROM read_parquet('{phase_glob}')
            WHERE incidence < 90
              AND emission < 60
              AND iof > 0.0
            GROUP BY 1
            ORDER BY 1
        ) TO '{output_path}' (HEADER, DELIMITER ',');
        """
        current_time = datetime.now().strftime("%H:%M:%S")
        print(
            f"[START] {current_time} | Aggregating {phase.upper()} from {len(parquet_files):,} files..."
        )
        con.execute(query)
        con = duckdb.connect(database=":memory:")  # Reconnect to ensure we read the latest file system state
        con.execute("SET show_progress=true;")  # Enable progress bar for the next query

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

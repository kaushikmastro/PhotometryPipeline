#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _fmt_pct(value: float) -> str:
    return f"{value:,.4f}%"


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parquet_glob = project_root / "data" / "04_geometry_tables" / "*" / "*.parquet"
    parquet_files = sorted(project_root.glob("data/04_geometry_tables/*/*.parquet"))

    print("Phase 2 preflight: corrupted parquet sweep")
    print(f"Parquet source: {parquet_glob}")

    if not parquet_files:
        print("No parquet files found. Nothing to audit.")
        return 0

    corrupted_count = 0
    for parquet_path in parquet_files:
        try:
            _ = pq.ParquetFile(parquet_path).schema
        except Exception as exc:
            corrupted_count += 1
            print(f"WARNING: Removing corrupted parquet: {parquet_path} ({exc})")
            os.remove(parquet_path)

    print(f"Corrupted files removed: {_fmt_int(corrupted_count)}")

    # Enforce conservative thread counts on shared nodes during the aggregate query.
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    con = duckdb.connect(database=":memory:")
    query = """
        WITH valid AS (
            SELECT phase
            FROM read_parquet(?, union_by_name=true)
            WHERE incidence < 90
              AND emission < 60
              AND iof > 0.0
        )
        SELECT
            COUNT(*)::BIGINT AS total_valid,
            SUM(CASE WHEN phase < 15.0 THEN 1 ELSE 0 END)::BIGINT AS lt_15,
            SUM(CASE WHEN phase < 10.0 THEN 1 ELSE 0 END)::BIGINT AS lt_10,
            SUM(CASE WHEN phase < 5.0 THEN 1 ELSE 0 END)::BIGINT AS lt_5
        FROM valid
    """

    result = con.execute(query, [str(parquet_glob)]).fetchone()
    con.close()

    if result is None:
        raise RuntimeError("DuckDB query returned no rows.")

    total_valid, lt_15, lt_10, lt_5 = map(int, result)

    def pct(count: int) -> float:
        return (count / total_valid * 100.0) if total_valid > 0 else 0.0

    rows = [
        ("Phase < 15.0 deg", lt_15, pct(lt_15)),
        ("Phase < 10.0 deg", lt_10, pct(lt_10)),
        ("Phase < 5.0 deg", lt_5, pct(lt_5)),
    ]

    print()
    print("DuckDB Phase Coverage Audit")
    print("QA filters: incidence < 90, emission < 60, iof > 0.0")
    print(f"Total valid pixels: {_fmt_int(total_valid)}")
    print()
    print("+-------------------+----------------------+----------------+")
    print("| Phase regime      | Valid pixel count    | Percent of all |")
    print("+-------------------+----------------------+----------------+")
    for label, count, percentage in rows:
        print(f"| {label:<17} | {_fmt_int(count):>20} | {_fmt_pct(percentage):>14} |")
    print("+-------------------+----------------------+----------------+")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

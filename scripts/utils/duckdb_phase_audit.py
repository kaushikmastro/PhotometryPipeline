#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import duckdb


def format_int(value: int) -> str:
    return f"{value:,}"


def format_pct(value: float) -> str:
    return f"{value:,.4f}%"


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    parquet_glob = project_root / "data" / "04_geometry_tables" / "*" / "*.parquet"

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
    if result is None:
        raise RuntimeError("DuckDB query returned no rows.")

    total_valid, lt_15, lt_10, lt_5 = map(int, result)

    def pct(count: int) -> float:
        if total_valid == 0:
            return 0.0
        return (count / total_valid) * 100.0

    rows = [
        ("Phase < 15.0 deg", lt_15, pct(lt_15)),
        ("Phase < 10.0 deg", lt_10, pct(lt_10)),
        ("Phase < 5.0 deg", lt_5, pct(lt_5)),
    ]

    print("DuckDB Phase Coverage Audit")
    print("Parquet source:", parquet_glob)
    print("QA filters: incidence < 90, emission < 60, iof > 0.0")
    print()
    print(f"Total valid pixels: {format_int(total_valid)}")
    print()
    print("+-------------------+----------------------+----------------+")
    print("| Phase regime      | Valid pixel count    | Percent of all |")
    print("+-------------------+----------------------+----------------+")
    for label, count, percentage in rows:
        print(
            f"| {label:<17} | {format_int(count):>20} | {format_pct(percentage):>14} |"
        )
    print("+-------------------+----------------------+----------------+")

    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

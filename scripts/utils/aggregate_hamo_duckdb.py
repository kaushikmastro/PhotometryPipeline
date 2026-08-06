#!/usr/bin/env python3
from pathlib import Path

import duckdb


def main():
    project_root = Path(__file__).resolve().parents[2]
    hamo_glob = str(project_root / "data" / "04_geometry_tables" / "hamo" / "*.parquet")
    output_path = str(project_root / "data" / "05_aggregated" / "hamo_phase_curve.csv")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")
    
    # DIAGNOSTIC: Let's see what the data actually looks like
    print(f"Checking first file in: {hamo_glob}")
    try:
        sample = con.execute(f"SELECT * FROM read_parquet('{hamo_glob}') LIMIT 1").fetchone()
        columns = [desc[0] for desc in con.description]
        print(f"Found columns: {columns}")
    except Exception as e:
        print(f"ERROR: Could not read Parquet files. Check path: {e}")
        return 1

    # THE NUCLEAR QUERY: Using F-Strings and explicit casting
    query = f"""
    COPY (
        SELECT 
            ROUND(phase) AS phase_bin_deg,
            AVG(iof) AS mean_iof,
            AVG(incidence) AS mean_incidence,
            AVG(emission) AS mean_emission
        FROM read_parquet('{hamo_glob}')
        WHERE incidence < 90 
          AND emission < 60
          AND iof > 0
        GROUP BY 1
        ORDER BY 1
    ) TO '{output_path}' (HEADER, DELIMITER ',');
    """
    
    print("Running aggregation...")
    con.execute(query)
    
    if Path(output_path).exists():
        count = con.execute(f"SELECT COUNT(*) FROM read_csv_auto('{output_path}')").fetchone()[0]
        print(f"SUCCESS: Wrote {count} bins to {output_path}")
    else:
        print("!! ERROR: Query finished but NO CSV was created. The filter likely returned 0 rows.")
    
    return 0

if __name__ == "__main__":
    main()
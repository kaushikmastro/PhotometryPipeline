"""Disk-resolved golden-layer binning, extracted from Hapke.ipynb (this is the code lost
in the Aug-12 SLURM crash and reconstructed from IPython history -- see the session's
recovery record; it belongs in version control, not a notebook kernel).

Reproduces the existing 5-degree floor-binning EXACTLY, unchanged: banker's rounding
(round-half-to-even) with a float-safe 1e-9 epsilon on phase/incidence/emission, the
triangle-law geometric-validity filter, and the >=10-pixel / non-null-stddev statistical
floor. Only the max_incidence_emission_deg domain cutoff (50 vs 80 in the two committed
variants) and the input/output paths are parameters -- the binning logic itself is not
configurable, by design, since the point is exact reproduction, not a general binner.

RULE 1 (CLAUDE.md): the real input parquet here is O(10GB) / hundreds of millions of
rows -- running this for real MUST go through srun/sbatch, never directly on the login
node. See scripts/submit/ for the sbatch pattern used elsewhere in this project.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BIN_SIZE_DEG = 5.0


def _banker_round_grid_expr(column: str, bin_size_deg: float) -> str:
    """Banker's rounding (round-half-to-even) to the nearest bin_size_deg, with a
    1e-9 float-safe epsilon around the exact .5 boundary. Verbatim from Hapke.ipynb's
    binned_dsk256_lt50/lt80 queries -- do not simplify or reformat, this is byte-for-byte
    what produced the committed golden parquets."""
    return f"""(CASE
        WHEN ABS(({column}/{bin_size_deg}) - FLOOR({column}/{bin_size_deg}) - 0.5) < 1e-9
        THEN (CASE WHEN CAST(FLOOR({column}/{bin_size_deg}) AS BIGINT) % 2 = 0
              THEN FLOOR({column}/{bin_size_deg}) ELSE CEIL({column}/{bin_size_deg}) END)
        ELSE ROUND({column}/{bin_size_deg})
    END) * {bin_size_deg}"""


def build_golden_query(
    input_parquet_path: str,
    max_incidence_emission_deg: float,
    *,
    bin_size_deg: float = BIN_SIZE_DEG,
    image_id_like: str = "%F1B%",
    min_pixels_per_bin: int = 10,
) -> str:
    """Pure string construction, no DB access -- the query itself is unit-testable
    without touching the (huge) real data. Structure matches Hapke.ipynb's
    binned_dsk256_lt50_query / binned_dsk256_lt80_query exactly, generalized only over
    the incidence/emission threshold, bin size, image_id filter, and min-pixel floor.
    """
    alpha_expr = _banker_round_grid_expr("phase", bin_size_deg)
    i_expr = _banker_round_grid_expr("incidence", bin_size_deg)
    e_expr = _banker_round_grid_expr("emission", bin_size_deg)

    return f"""

SELECT
    -- 1. Banker's Rounding with Float-Safe Epsilon (< 1e-9)

    {alpha_expr} AS alpha_grid,

    {i_expr} AS i_grid,

    {e_expr} AS e_grid,

    -- 2. Target Aggregations

    AVG(incidence) AS mean_incidence,
    AVG(emission) AS mean_emission,
    AVG(phase) AS mean_phase,
    AVG(iof) AS mean_iof,
    STDDEV_SAMP(iof) AS std_iof,
    COUNT(*) AS n_pixels

FROM read_parquet('{input_parquet_path}')

-- 3. The Physical Boundary Restored (Triangle Law)

WHERE incidence < {max_incidence_emission_deg}
  AND emission < {max_incidence_emission_deg}
  -- AND iof > 0.01
  AND image_id LIKE '{image_id_like}'
  AND phase >= ABS(incidence - emission)
  AND phase <= (incidence + emission) + 0.1

GROUP BY 1, 2, 3

-- 4. Statistical Validity Check

HAVING COUNT(*) >= {min_pixels_per_bin}
   AND STDDEV_SAMP(iof) IS NOT NULL

ORDER BY 1, 2, 3

"""


def build_disk_resolved_golden(
    input_parquet_path: str,
    output_path: str,
    max_incidence_emission_deg: float,
    *,
    threads: int = 8,
    memory_limit: str = "28GB",
) -> pd.DataFrame:
    """Run the binning query and write the result to output_path. Mirrors
    Hapke.ipynb's cell-level pattern (PRAGMA threads/memory_limit, then the query, then
    to_parquet) exactly."""
    duckdb.sql(f"PRAGMA threads={threads};")
    duckdb.sql(f"PRAGMA memory_limit='{memory_limit}';")

    query = build_golden_query(input_parquet_path, max_incidence_emission_deg)
    df = duckdb.sql(query).df()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logging.info("Wrote %d bins to %s", len(df), output_path)
    return df


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    input_path = (
        project_root / "data" / "silver" / "gaskell_dsk256_110825"
        / "DR_survey_gaskell_dsk256_110825.parquet"
    )

    for max_deg, suffix in [(50.0, "range50"), (80.0, "range80")]:
        output_path = project_root / "data" / "golden" / f"survey_binned_dsk256_110825_{suffix}.parquet"
        df = build_disk_resolved_golden(str(input_path), str(output_path), max_deg)
        logging.info("%s: %d bins", suffix, len(df))


if __name__ == "__main__":
    main()

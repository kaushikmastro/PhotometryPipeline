
"""Bin a disk-resolved silver-layer parquet into a 5-degree (i, e, alpha) geometry cube.
 
Produces the golden layer consumed by the photometric fitting notebooks. Works for any
mission phase (survey / hamo / lamo / rc / approach) and any incidence/emission domain.
 
The binning logic here is deliberately identical to the original Hapke.ipynb
implementation -- round-to-nearest with banker's tie-breaking, the spherical triangle
filter, and the >=10-pixel statistical floor. Do not change it casually: the committed
Case 1 result (w=0.46993, g=-0.33688, theta_bar=8.2662 deg) depends on this exact
definition, and a bit-for-bit reproduction test guards it.
 
Usage
-----
Survey, i,e < 50 (the committed Case 1 domain):
 
    python scripts/extract/bin_geometry_cube.py \
        --input data/silver/gaskell_dsk256_110825/DR_survey_gaskell_dsk256_110825.parquet \
        --incidence 50 --emission 50
 
HAMO, wider domain:
 
    python scripts/extract/bin_geometry_cube.py \
        --input data/silver/gaskell_dsk256_110825/DR_hamo_gaskell_dsk256_110825.parquet \
        --incidence 80 --emission 80
 
LAMO, explicit output path:
 
    python scripts/extract/bin_geometry_cube.py \
        --input data/silver/gaskell_dsk256_110825/DR_lamo_gaskell_dsk256_110825.parquet \
        --incidence 80 --emission 80 \
        --output data/golden/lamo_binned_gaskell_dsk256_110825_i80_e80.parquet
 
Run on a compute node, not the login node:
 
    srun --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=28G \
         --partition=main --qos=standard --time=02:00:00 --pty bash
 
Requirements: duckdb, pyarrow
"""
 
from __future__ import annotations
 
import argparse
import os
import socket
import sys
import time
from pathlib import Path
 
import duckdb
 
# Default bin width in degrees. Changing this breaks comparability with every
# committed result; it is exposed as a flag only for deliberate sensitivity tests.
DEFAULT_BIN_DEG = 5.0
 
# Minimum pixels per bin. Below this the per-bin standard deviation (used as the
# fitting weight) is not meaningful.
DEFAULT_MIN_COUNT = 10
 
# Tolerance on the upper triangle-inequality bound, in degrees, to absorb
# floating-point noise in the SPICE-derived angles.
TRIANGLE_TOL_DEG = 0.1
 
 
def guard_against_login_node() -> None:
    hostname = socket.getfqdn().lower()
    if "login" in hostname:
        print("=" * 60, file=sys.stderr)
        print(f"Refusing to run on shared login node: {hostname}", file=sys.stderr)
        print("Use srun/sbatch to get a compute node (see module docstring).", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        sys.exit(1)
 
 
def detect_resources(args) -> tuple[int, str]:
    """Take threads/memory from SLURM when available, else the CLI defaults."""
    threads = args.threads
    if threads is None:
        threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
 
    memory = args.memory
    if memory is None:
        mem_mb = os.environ.get("SLURM_MEM_PER_NODE")
        if mem_mb:
            # Leave ~15% headroom so DuckDB doesn't get OOM-killed by SLURM.
            memory = f"{int(int(mem_mb) * 0.85 / 1024)}GB"
        else:
            memory = "28GB"
    return threads, memory
 
 
def parse_input_name(stem: str) -> tuple[str, str]:
    """Split 'DR_survey_gaskell_dsk256_110825' -> ('survey', 'gaskell_dsk256_110825').
 
    Falls back gracefully if the convention changes; callers can always override
    with --mission-phase / --shape-model.
    """
    parts = stem.split("_")
    mission_phase = parts[1] if len(parts) > 1 else "unknown"
    shape_model = "_".join(parts[2:]) if len(parts) > 2 else "unknown"
    return mission_phase, shape_model
 
 
def bankers_round_expr(column: str, bin_deg: float) -> str:
    """SQL for round-half-to-even binning, returning the bin CENTRE in degrees.
 
    Note this is round-to-nearest, not floor: alpha_grid=10.0 covers [7.5, 12.5).
    Downstream code must not add half a bin width to these labels.
    """
    q = f"({column}/{bin_deg})"
    return f"""(CASE
            WHEN ABS({q} - FLOOR({q}) - 0.5) < 1e-9
            THEN (CASE WHEN CAST(FLOOR({q}) AS BIGINT) % 2 = 0
                  THEN FLOOR({q}) ELSE CEIL({q}) END)
            ELSE ROUND({q})
        END) * {bin_deg}"""
 
 
def build_sql(input_path: Path, args) -> str:
    bin_deg = args.bin_size
    return f"""
    SELECT
        {bankers_round_expr('phase', bin_deg)} AS alpha_grid,
        {bankers_round_expr('incidence', bin_deg)} AS i_grid,
        {bankers_round_expr('emission', bin_deg)} AS e_grid,
 
        AVG(incidence)    AS mean_incidence,
        AVG(emission)     AS mean_emission,
        AVG(phase)        AS mean_phase,
        AVG(iof)          AS mean_iof,
        QUANTILE_CONT(iof, 0.75) - QUANTILE_CONT(iof,0.25) AS iqr_iof,
        STDDEV_SAMP(iof)  AS std_iof,
        COUNT(*)          AS n_pixels
 
    FROM read_parquet('{input_path.as_posix()}')
 
    WHERE incidence < {args.incidence}
      AND emission  < {args.emission}
      -- Spherical triangle inequality: the phase angle must lie between the
      -- difference and the sum of the incidence and emission angles.
      AND phase >= ABS(incidence - emission)
      AND phase <= (incidence + emission) + {TRIANGLE_TOL_DEG}
 
    GROUP BY 1, 2, 3
 
    HAVING COUNT(*) >= {args.min_count}
       AND STDDEV_SAMP(iof) IS NOT NULL
 
    ORDER BY 1, 2, 3
    """
 
 
def parse_args():
    p = argparse.ArgumentParser(
        description="Bin a disk-resolved silver parquet into an (i, e, alpha) geometry cube.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path,
                   help="Silver-layer parquet, e.g. data/silver/<model>/DR_<phase>_<model>.parquet")
    p.add_argument("--incidence", type=float, default=50.0,
                   help="Incidence angle upper bound, exclusive (degrees)")
    p.add_argument("--emission", type=float, default=50.0,
                   help="Emission angle upper bound, exclusive (degrees)")
 
    p.add_argument("--output", type=Path, default=None,
                   help="Explicit output parquet path. Overrides --output-dir and auto-naming.")
    p.add_argument("--output-dir", type=Path, default=Path("data/golden"),
                   help="Directory for the auto-named output (relative to --project-root)")
    p.add_argument("--project-root", type=Path, default=None,
                   help="Repo root for resolving relative paths. Default: inferred from this file.")
 
    p.add_argument("--mission-phase", default=None,
                   help="Override the phase parsed from the input filename (survey/hamo/lamo/rc/approach)")
    p.add_argument("--shape-model", default=None,
                   help="Override the shape-model tag parsed from the input filename")
 
    p.add_argument("--bin-size", type=float, default=DEFAULT_BIN_DEG,
                   help="Bin width in degrees. Changing this breaks comparability with committed results.")
    p.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT,
                   help="Minimum pixels per bin for the per-bin std to be meaningful")
 
    p.add_argument("--threads", type=int, default=None,
                   help="DuckDB threads. Default: $SLURM_CPUS_PER_TASK, else 8.")
    p.add_argument("--memory", default=None,
                   help="DuckDB memory limit, e.g. '28GB'. Default: 85%% of $SLURM_MEM_PER_NODE, else 28GB.")
 
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite the output if it already exists")
    p.add_argument("--allow-login-node", action="store_true",
                   help="Bypass the login-node guard (do not use for real runs)")
    return p.parse_args()
 
 
def main() -> None:
    args = parse_args()
    if not args.allow_login_node:
        guard_against_login_node()
 
    project_root = args.project_root or Path(__file__).resolve().parents[2]
 
    input_path = args.input if args.input.is_absolute() else (project_root / args.input)
    input_path = input_path.resolve()
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)
 
    phase_parsed, model_parsed = parse_input_name(input_path.stem)
    mission_phase = args.mission_phase or phase_parsed
    shape_model = args.shape_model or model_parsed
 
    if args.output is not None:
        output_path = args.output if args.output.is_absolute() else (project_root / args.output)
    else:
        out_dir = args.output_dir if args.output_dir.is_absolute() else (project_root / args.output_dir)
        name = (f"{mission_phase}_binned_{shape_model}"
                f"_i{int(args.incidence)}_e{int(args.emission)}.parquet")
        output_path = out_dir / name
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
 
    if output_path.exists() and not args.overwrite:
        print(f"Output already exists (use --overwrite): {output_path}", file=sys.stderr)
        sys.exit(1)
 
    threads, memory = detect_resources(args)
 
    if abs(args.bin_size - DEFAULT_BIN_DEG) > 1e-9:
        print(f"WARNING: bin size {args.bin_size} deg differs from the {DEFAULT_BIN_DEG} deg "
              f"standard. Results will not be comparable to committed fits.", file=sys.stderr)
 
    print("-" * 60)
    print(f"Input        : {input_path}")
    print(f"Output       : {output_path}")
    print(f"Phase / model: {mission_phase} / {shape_model}")
    print(f"Domain       : incidence < {args.incidence}, emission < {args.emission}")
    print(f"Binning      : {args.bin_size} deg, min {args.min_count} px/bin")
    print(f"DuckDB       : {threads} threads, {memory} limit")
    print("-" * 60)
 
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA enable_progress_bar;")
    con.execute(f"PRAGMA threads={threads};")
    con.execute(f"PRAGMA memory_limit='{memory}';")
 
    sql = build_sql(input_path, args)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
 
    start = time.time()
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}")
    try:
        con.execute(
            f"COPY ({sql}) TO '{tmp_path.as_posix()}' (FORMAT PARQUET, COMPRESSION 'ZSTD')"
        )
        tmp_path.replace(output_path)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"\nBinning failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
 
    elapsed = time.time() - start
    print(f"Elapsed: {int(elapsed // 60)}m {elapsed % 60:.1f}s")
 
    summary = con.execute(f"""
        SELECT
            COUNT(*)        AS total_bins,
            SUM(n_pixels)   AS total_pixels,
            AVG(n_pixels)   AS avg_px_per_bin,
            MIN(alpha_grid) AS alpha_min,
            MAX(alpha_grid) AS alpha_max,
            MIN(i_grid)     AS i_min,
            MAX(i_grid)     AS i_max,
            MIN(e_grid)     AS e_min,
            MAX(e_grid)     AS e_max
        FROM read_parquet('{output_path.as_posix()}')
    """).df()
 
    if summary.empty or summary["total_bins"].iloc[0] == 0:
        print("\nWARNING: output is empty. Check the domain filters.", file=sys.stderr)
        sys.exit(1)
 
    row = summary.iloc[0]
    print("\nSummary")
    print(f"  bins           : {int(row['total_bins']):,}")
    print(f"  pixels binned  : {int(row['total_pixels']):,}")
    print(f"  mean px/bin    : {row['avg_px_per_bin']:.1f}")
    print(f"  alpha range    : {row['alpha_min']:.1f} - {row['alpha_max']:.1f} deg")
    print(f"  incidence range: {row['i_min']:.1f} - {row['i_max']:.1f} deg")
    print(f"  emission range : {row['e_min']:.1f} - {row['e_max']:.1f} deg")
    print(f"\nWrote {output_path}")
 
 
if __name__ == "__main__":
    main()
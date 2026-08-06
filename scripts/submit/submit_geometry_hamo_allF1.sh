#!/usr/bin/env bash
#SBATCH --job-name=hamo_geom_allF1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=50:00:00
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --output=logs/geom_hamo_allF1_%j.out
#SBATCH --error=logs/geom_hamo_allF1_%j.err

# HAMO all-F1-version-letter geometry grind with mission-science 110825 DSK.
# Input:  calibrated_raw_images/hamo/ (5547 F1-any images: 1089 F1B + 4458 new)
# Output: geometry/dsk256/hamo_allF1/  (separate from hamo/)
#
# Rate: measured 36.56 sec/image wall-clock at 8 workers on the prior F1B-only
# run (job 25893871: 1089 images in 11h03m). Idempotent skip logic means this
# job only computes the ~4458 non-B images even though it globs all 5547;
# expected new-image wall time ~= 4458/8 * 292.5s/image-cpu-time ~= 45.2h.
# --time=50:00:00 gives ~10% buffer. QOS standard MaxWall=14-00:00:00, so no
# limit conflict.

set -euo pipefail
cd /home/kaushim07/photometry_mcmc_env

mkdir -p logs

source /home/kaushim07/miniforge3/etc/profile.d/conda.sh
conda activate photomc_env

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export POLARS_MAX_THREADS=1

SCRATCH=/scratch/kaushim07/vesta_data
SPICE_DIR="${SCRATCH}/spice_kernels"
IMG_DIR="${SCRATCH}/calibrated_raw_images/hamo"
OUT_DIR="${SCRATCH}/geometry/dsk256/hamo_allF1"
F1B_DIR="${SCRATCH}/geometry/dsk256/hamo"

echo "========================================================"
echo "HAMO all-F1 geometry grind — DSK 110825, f_solar=892"
echo "Input: ${IMG_DIR}"
echo "Output: ${OUT_DIR}  (F1B-only tables at ${F1B_DIR} untouched)"
echo "Start: $(date)"
echo "Workers: ${SLURM_CPUS_PER_TASK:-8}"
echo "========================================================"

# ── Provenance checks — abort on any failure ───────────────────────────────
echo "DSK in metakernel:"
grep "110825" "${SPICE_DIR}/dawn_dynamic.tm" || \
    { echo "ERROR: vesta_gaskell_256_110825.bds not found in metakernel"; exit 1; }

echo "DSK file present:"
stat -c '%s %n' "${SPICE_DIR}/vesta_gaskell_256_110825.bds" || \
    { echo "ERROR: 110825 DSK file not found on scratch"; exit 1; }

echo "HAMO all-F1 image count on disk:"
find "${IMG_DIR}" -name "*F1[A-Z].IMG" | wc -l

echo "Sanity: existing validated F1B-only output untouched (should still be 1089):"
find "${F1B_DIR}" -name "*.parquet" 2>/dev/null | wc -l

# ── Geometry grind ─────────────────────────────────────────────────────────
python scripts/geometry/run_geometry_hamo_allF1.py

echo "========================================================"
echo "Geometry complete: $(date)"
echo "========================================================"

# ── Post-job validation (runs on same compute node, not login node) ────────
echo "Running post-job validation..."
mkdir -p /scratch/kaushim07/duckdb_tmp

python3 - << 'PYEOF'
import duckdb, pathlib, re, sys

out_dir = "/scratch/kaushim07/vesta_data/geometry/dsk256/hamo_allF1"
f1b_dir = "/scratch/kaushim07/vesta_data/geometry/dsk256/hamo"
parquets = list(pathlib.Path(out_dir).glob("*.parquet"))
if not parquets:
    print(f"VALIDATION FAILED: no parquet files in {out_dir}")
    sys.exit(1)

con = duckdb.connect()
con.execute("SET memory_limit='28GB'; SET temp_directory='/scratch/kaushim07/duckdb_tmp'; SET threads=8;")

r = con.execute(f"""
    SELECT COUNT(*)                 AS n_pixels,
           COUNT(DISTINCT image_id) AS n_images,
           AVG(iof)                 AS mean_iof,
           MAX(iof)                 AS max_iof
    FROM read_parquet('{out_dir}/*.parquet')
""").fetchone()
n_pixels, n_images, mean_iof, max_iof = r

# Filter-contamination check: every image_id must end in F1<single A-Z letter>.
# Guards against accidentally sweeping in a different FILTER_NUMBER.
bad = con.execute(f"""
    SELECT COUNT(DISTINCT image_id)
    FROM read_parquet('{out_dir}/*.parquet')
    WHERE NOT regexp_matches(image_id, 'F1[A-Z]$')
""").fetchone()[0]

# Letter breakdown for the report.
letters = con.execute(f"""
    SELECT regexp_extract(image_id, 'F1([A-Z])$', 1) AS letter,
           COUNT(DISTINCT image_id) AS n
    FROM read_parquet('{out_dir}/*.parquet')
    GROUP BY 1 ORDER BY 1
""").fetchall()

size_bytes = sum(p.stat().st_size for p in parquets)

print(f"n_parquets       : {len(parquets)}")
print(f"n_pixels          : {n_pixels:,}")
print(f"n_images          : {n_images}")
print(f"mean_iof          : {mean_iof:.6f}")
print(f"max_iof           : {max_iof:.6f}")
print(f"filter_contam_rows: {bad}  (must be 0)")
print(f"total_size_bytes  : {size_bytes:,}")
print("letter breakdown  :", ", ".join(f"F1{l}={n}" for l, n in letters))

ok = True
if n_images != 5547:
    print(f"FAIL n_images: expected 5547, got {n_images}")
    ok = False
if bad != 0:
    print(f"FAIL filter contamination: {bad} image_id(s) not matching F1[A-Z]$")
    ok = False
if not (0.07 <= mean_iof <= 0.20):
    print(f"FAIL mean_iof={mean_iof:.4f}: expected [0.07, 0.20]")
    ok = False

if not ok:
    sys.exit(1)

print("PASS: n_images=5547, zero filter contamination, mean_iof in [0.07, 0.20]")
sentinel = pathlib.Path("logs/hamo_allF1_geometry_complete.sentinel")
sentinel.write_text(
    f"n_parquets={len(parquets)}\nn_pixels={n_pixels}\nn_images={n_images}\n"
    f"mean_iof={mean_iof:.6f}\nmax_iof={max_iof:.6f}\ntotal_size_bytes={size_bytes}\n"
    f"dsk=vesta_gaskell_256_110825.bds\nf_solar=892\n"
    f"letter_breakdown={dict(letters)}\n"
)
print(f"Sentinel written: {sentinel}")
PYEOF

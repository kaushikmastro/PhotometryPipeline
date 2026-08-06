#!/usr/bin/env bash
#SBATCH --job-name=lamo_geom_110825
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --output=logs/geom_lamo_110825_%j.out
#SBATCH --error=logs/geom_lamo_110825_%j.err

# LAMO F1B geometry grind with mission-science 110825 DSK.
# Input:  calibrated_raw_images/lamo/ (4349 F1B images, CYCLE15-20 + Transfer-to-LAMO)
# Output: geometry/dsk256/lamo/
# Rate:   ~33 sec/image effective throughput at 8 workers -> ~40h; 48h wall for margin

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
IMG_DIR="${SCRATCH}/calibrated_raw_images/lamo"
OUT_DIR="${SCRATCH}/geometry/dsk256/lamo"

echo "========================================================"
echo "LAMO geometry grind — DSK 110825, f_solar=892"
echo "Input: ${IMG_DIR}"
echo "Output: ${OUT_DIR}"
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

echo "LAMO F1B image count:"
find "${IMG_DIR}" -name "*F1B*.IMG" | wc -l

# ── Geometry grind ─────────────────────────────────────────────────────────
python scripts/geometry/run_geometry_lamo_110825.py

echo "========================================================"
echo "Geometry complete: $(date)"
echo "========================================================"

# ── Post-job validation (runs on same compute node, not login node) ────────
echo "Running post-job validation..."
mkdir -p /scratch/kaushim07/duckdb_tmp

python3 - << 'PYEOF'
import duckdb, pathlib, sys

out_dir = "/scratch/kaushim07/vesta_data/geometry/dsk256/lamo"
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
print(f"n_parquets  : {len(parquets)}")
print(f"n_pixels    : {n_pixels:,}")
print(f"n_images    : {n_images}")
print(f"mean_iof    : {mean_iof:.6f}")
print(f"max_iof     : {max_iof:.6f}")

ok = True
if n_images != 4349:
    print(f"FAIL n_images: expected 4349, got {n_images}")
    ok = False
if not (0.06 <= mean_iof <= 0.20):
    print(f"FAIL mean_iof={mean_iof:.4f}: expected [0.06, 0.20]  (f_solar=1473 floor ~0.060)")
    ok = False

if not ok:
    sys.exit(1)

print("PASS: n_images=4349, mean_iof in [0.06, 0.20]")
sentinel = pathlib.Path("logs/lamo_110825_geometry_complete.sentinel")
sentinel.write_text(
    f"n_parquets={len(parquets)}\nn_pixels={n_pixels}\nn_images={n_images}\n"
    f"mean_iof={mean_iof:.6f}\nmax_iof={max_iof:.6f}\n"
    f"dsk=vesta_gaskell_256_110825.bds\nf_solar=892\n"
)
print(f"Sentinel written: {sentinel}")
PYEOF
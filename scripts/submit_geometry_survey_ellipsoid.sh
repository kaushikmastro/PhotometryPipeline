#!/usr/bin/env bash
#SBATCH --job-name=survey_geom_ellipsoid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=13:00:00
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --output=logs/geom_survey_ellipsoid_%j.out
#SBATCH --error=logs/geom_survey_ellipsoid_%j.err

# Survey F1B geometry grind — ELLIPSOID shape model (analytical, no DSK).
# Input:  01_calibrated_images/survey/ (845 F1B *IMG* files on disk)
# Output: 04_geometry_tables_ellipsoid/survey/
# Radii:  BODY2000004_RADII = (289, 280, 229) km from pck00010.tpc (DAWN-derived)
# NOTE:   1153 = total Survey parquet count including all filters; 845 = F1B only.
# Ellipsoid intersections are analytically fast; actual runtime expected well under 13h.

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
SPICE_DIR="${SCRATCH}/02_spice_kernels"
IMG_DIR="${SCRATCH}/01_calibrated_images/survey"
OUT_DIR="${SCRATCH}/04_geometry_tables_ellipsoid/survey"

echo "========================================================"
echo "Survey geometry grind — ELLIPSOID, f_solar=892"
echo "Radii: (289, 280, 229) km from pck00010.tpc"
echo "Input: ${IMG_DIR}"
echo "Output: ${OUT_DIR}"
echo "Start: $(date)"
echo "Workers: ${SLURM_CPUS_PER_TASK:-8}"
echo "========================================================"

# ── Provenance checks — abort on any failure ───────────────────────────────
echo "PCK kernel present:"
stat -c '%s %n' "${SPICE_DIR}/pck00010.tpc" || \
    { echo "ERROR: pck00010.tpc not found"; exit 1; }

echo "Vesta radii in PCK:"
grep "BODY2000004_RADII" "${SPICE_DIR}/pck00010.tpc" || \
    { echo "ERROR: BODY2000004_RADII not found in PCK"; exit 1; }

echo "Survey F1B image count:"
find "${IMG_DIR}" -name "*F1B*.IMG" | wc -l

# ── Geometry grind ─────────────────────────────────────────────────────────
python scripts/run_geometry_survey_ellipsoid.py

echo "========================================================"
echo "Geometry complete: $(date)"
echo "========================================================"

# ── Post-job validation (runs on same compute node, not login node) ────────
# Validates on daylight pixels (iof>0.01 AND incidence<80) to avoid
# night-side pixel contamination from ellipsoid full-frame coverage.
echo "Running post-job validation..."
mkdir -p /scratch/kaushim07/duckdb_tmp

python3 << 'PYEOF'
import duckdb, os
os.makedirs("/scratch/kaushim07/duckdb_tmp", exist_ok=True)
con = duckdb.connect()
con.execute("""
  SET memory_limit='12GB';
  SET threads=4;
  SET temp_directory='/scratch/kaushim07/duckdb_tmp';
""")
ELLIPSOID = "/scratch/kaushim07/vesta_data/04_geometry_tables_ellipsoid/survey/*.parquet"
r = con.execute(f"""
  SELECT
    COUNT(*)                    AS n_total,
    COUNT(DISTINCT image_id)    AS n_images,
    AVG(iof)                    AS mean_iof_all,
    AVG(CASE WHEN iof > 0.01 AND incidence < 80
             THEN iof END)      AS mean_iof_daylight,
    SUM(CASE WHEN iof > 0.01 AND incidence < 80
             THEN 1 ELSE 0 END) AS n_daylight_pixels
  FROM read_parquet('{ELLIPSOID}')
""").fetchone()
print(f"n_total           : {r[0]:,}")
print(f"n_images          : {r[1]}")
print(f"mean_iof_all      : {r[2]:.5f}")
print(f"mean_iof_daylight : {r[3]:.5f}")
print(f"n_daylight_pixels : {r[4]:,}")
assert r[1] == 816, \
  f"FAIL: expected 816 images, got {r[1]}  (845 F1B files - 29 approach DOY 11123 = 816 genuine Survey)"
assert r[3] is not None and 0.08 <= r[3] <= 0.25, \
  f"FAIL: mean_iof_daylight={r[3]:.5f} outside [0.08,0.25]"
print("VALIDATION PASSED")
open("logs/survey_ellipsoid_geometry_complete.sentinel",
     "w").write("OK")
PYEOF
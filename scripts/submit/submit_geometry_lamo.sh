#!/usr/bin/env bash
#SBATCH --job-name=lamo_geometry
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --output=logs/geometry_lamo_%j.out
#SBATCH --error=logs/geometry_lamo_%j.err

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

echo "========================================================"
echo "LAMO geometry grind — DSK/UNPRIORITIZED, f_solar=892"
echo "Epoch: DOY 11346–12167 (Dec 2011 – Jun 2012)"
echo "Start: $(date)"
echo "Workers: ${SLURM_CPUS_PER_TASK:-8}"
echo "========================================================"

# Provenance checks — abort if these fail
echo "DSK in metakernel:"
grep -i "vesta_gaskell" /scratch/kaushim07/vesta_data/spice_kernels/dawn_dynamic.tm || \
    { echo "ERROR: no DSK found in metakernel"; exit 1; }

echo "DSK file size:"
stat -c '%s %n' /scratch/kaushim07/vesta_data/spice_kernels/vesta_gaskell_256*.bds || \
    { echo "ERROR: DSK file not found on scratch"; exit 1; }

echo "LAMO F1B image count:"
find /scratch/kaushim07/vesta_data/calibrated_raw_images/lamo -name "*F1B*.IMG" | wc -l

python scripts/geometry/run_geometry_lamo.py

echo "========================================================"
echo "Geometry complete: $(date)"
echo "========================================================"

# ── Post-job validation ───────────────────────────────────────────────────────
echo "Running post-job validation ..."
python -c "
import duckdb, pathlib, sys

lamo_dir = '/scratch/kaushim07/vesta_data/04_geometry_tables_fast/lamo'
parquets = list(pathlib.Path(lamo_dir).glob('*.parquet'))
if not parquets:
    print('VALIDATION FAILED: no parquet files written to', lamo_dir)
    sys.exit(1)

r = duckdb.sql('''
    SELECT COUNT(*)              AS n,
           AVG(iof)             AS mean_iof,
           MAX(iof)             AS max_iof,
           COUNT(DISTINCT image_id) AS n_images
    FROM read_parquet(\"''' + lamo_dir + '''/*.parquet\")
''').fetchone()

print(f'n_pixels={r[0]:,}  mean_iof={r[1]:.4f}  '
      f'max_iof={r[2]:.4f}  n_images={r[3]}')

assert 0.10 < r[1] < 0.22, f'VALIDATION FAILED: mean_iof={r[1]:.4f} out of range [0.10, 0.22]'

sentinel = pathlib.Path('logs/lamo_geometry_complete.sentinel')
sentinel.write_text('OK')
print('LAMO GEOMETRY VALIDATED — sentinel written to', sentinel)
"
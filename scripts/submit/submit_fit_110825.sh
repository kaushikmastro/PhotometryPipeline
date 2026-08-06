#!/usr/bin/env bash
# Pre-bin 678M rows → ~949 cubes (DuckDB COPY, disk-spill enabled) then fit.
# Memory sizing from resource ledger:
#   - DuckDB GROUP BY 678M rows: ~16GB peak (disk-spill at 16GB limit)
#   - Three-case Hapke fit on ~949 bins: ~1.3GB
#   - Request 20G with 4 cores for DuckDB parallelism
#SBATCH --job-name=fit_110825
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=20G
#SBATCH --partition=main --qos=standard --time=01:00:00
#SBATCH --output=logs/fit_110825_%j.out --error=logs/fit_110825_%j.err

set -euo pipefail
cd /home/kaushim07/photometry_mcmc_env

source /home/kaushim07/miniforge3/etc/profile.d/conda.sh
conda activate photomc_env

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export RAYON_NUM_THREADS=1 POLARS_MAX_THREADS=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "=== Step 1: Pre-aggregate 678M rows → ~949 bins via DuckDB COPY ==="
python scripts/utils/prebin_survey_110825.py

echo "=== Step 2: Three-case fit on pre-binned data (~949 rows, ~1.3GB) ==="
python scripts/utils/run_fit_110825.py

#!/usr/bin/env bash
# Controlled shape-model comparison.
# Step 1: prebins BOTH DSK geometry tables with identical treatment (1% sample,
#         banker's rounding, same filters) + Step 3 scatter characterisation.
# Step 2: fits both and prints side-by-side table.
# Memory: DuckDB prebins 285M rows (full 110825) for Step 3 → ~16GB peak.
# Ledger: prelim prebin (~8.45M sampled px): trivial; 110825 prebin same; scatter=285M.
#SBATCH --job-name=ctrl_comparison
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=20G
#SBATCH --partition=main --qos=standard --time=00:30:00
#SBATCH --output=logs/ctrl_comparison_%j.out --error=logs/ctrl_comparison_%j.err

set -euo pipefail
cd /home/kaushim07/photometry_mcmc_env
source /home/kaushim07/miniforge3/etc/profile.d/conda.sh
conda activate photomc_env

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export RAYON_NUM_THREADS=1 POLARS_MAX_THREADS=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "=== Step 1 + 3: Prebinning both DSK sets and scatter characterisation ==="
python scripts/utils/run_controlled_comparison.py

echo "=== Step 2: Fitting both pre-binned sets and reporting comparison ==="
python scripts/run_fit_controlled.py

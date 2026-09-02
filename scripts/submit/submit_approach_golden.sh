#!/usr/bin/env bash
#SBATCH --job-name=approach_golden
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G
#SBATCH --partition=main --qos=standard --time=01:00:00
#SBATCH --output=logs/approach_golden_%j.out --error=logs/approach_golden_%j.err

set -euo pipefail
cd /home/kaushim07/photometry_mcmc_env

source /home/kaushim07/miniforge3/etc/profile.d/conda.sh
conda activate photomc_env

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "Node: $(hostname)"
python scripts/golden/build_approach_disk_integrated_golden.py

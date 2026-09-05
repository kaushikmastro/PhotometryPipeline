#!/usr/bin/env bash
#SBATCH --job-name=approach_phase_coverage
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G
#SBATCH --partition=main --qos=standard --time=00:15:00
#SBATCH --output=logs/approach_phase_coverage_%j.out --error=logs/approach_phase_coverage_%j.err

set -euo pipefail
cd /home/kaushim07/photometry_mcmc_env

source /home/kaushim07/miniforge3/etc/profile.d/conda.sh
conda activate photomc_env

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "Node: $(hostname)"
python .tmp/approach_phase_coverage.py

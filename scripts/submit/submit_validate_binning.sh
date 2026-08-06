#!/usr/bin/env bash
#SBATCH --job-name=validate_binning
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G
#SBATCH --partition=main --qos=standard --time=00:10:00
#SBATCH --output=logs/validate_binning_%j.out --error=logs/validate_binning_%j.err

set -euo pipefail
cd /home/kaushim07/photometry_mcmc_env
source /home/kaushim07/miniforge3/etc/profile.d/conda.sh
conda activate photomc_env

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export RAYON_NUM_THREADS=1 POLARS_MAX_THREADS=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python scripts/utils/validate_binning_agreement.py

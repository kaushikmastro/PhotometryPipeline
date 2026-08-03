#!/usr/bin/env bash
#SBATCH --job-name=hapke_fit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=main
#SBATCH --qos=standard
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /home/kaushim07/photometry_mcmc_env
source /home/kaushim07/miniforge3/etc/profile.d/conda.sh
conda activate photomc_env

export RAYON_NUM_THREADS=1
export POLARS_MAX_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/run_baseline_fit.py
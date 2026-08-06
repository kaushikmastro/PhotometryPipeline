#!/usr/bin/env bash
#SBATCH --job-name=hapke_albedo_het
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=32G
#SBATCH --partition=main --qos=standard --time=00:30:00
#SBATCH --output=logs/%x_%j.out --error=logs/%x_%j.err

set -euo pipefail
cd /home/kaushim07/photometry_mcmc_env
source /home/kaushim07/miniforge3/etc/profile.d/conda.sh
conda activate photomc_env

export RAYON_NUM_THREADS=1 POLARS_MAX_THREADS=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python scripts/diagnostics/diag_albedo_heterogeneity.py

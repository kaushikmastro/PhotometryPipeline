#!/bin/bash
#SBATCH --job-name=prelim_physfilter
#SBATCH --output=/home/kaushim07/photometry_mcmc_env/logs/prelim_physfilter_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=main
#SBATCH --qos=standard

set -euo pipefail

source /home/kaushim07/miniforge3/etc/profile.d/conda.sh
conda activate photomc_env

cd /home/kaushim07/photometry_mcmc_env
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Job: prelim DSK, physical illumination filter, Case 1 fit"
echo "Start: $(date)"

python scripts/utils/run_prelim_physfilter.py

echo "Done: $(date)"

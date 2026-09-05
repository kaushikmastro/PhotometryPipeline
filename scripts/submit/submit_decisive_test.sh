#!/usr/bin/env bash
#SBATCH --job-name=decisive_aperture_test
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G
#SBATCH --partition=main --qos=standard --time=00:30:00
#SBATCH --output=logs/decisive_aperture_test_%j.out --error=logs/decisive_aperture_test_%j.err

set -euo pipefail
cd /home/kaushim07/photometry_mcmc_env

source /home/kaushim07/miniforge3/etc/profile.d/conda.sh
conda activate photomc_env

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "Node: $(hostname)"
python .tmp/decisive_aperture_vs_pixel_test.py

#!/bin/bash
#SBATCH --job-name=xmirror_classify
#SBATCH --output=/home/kaushim07/photometry_mcmc_env/logs/xmirror_classify_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=main
#SBATCH --qos=standard

set -euo pipefail

source /home/kaushim07/miniforge3/etc/profile.d/conda.sh
conda activate photomc_env

cd /home/kaushim07/photometry_mcmc_env
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

echo "Job: x-mirror deterministic classification"
echo "Start: $(date)"

python scripts/diagnostics/xmirror_classify.py

echo "Done: $(date)"

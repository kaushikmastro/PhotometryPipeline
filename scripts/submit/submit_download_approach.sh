#!/usr/bin/env bash
#SBATCH --job-name=download_approach
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G
#SBATCH --partition=main --qos=standard --time=02:00:00
#SBATCH --output=logs/download_approach_%j.out --error=logs/download_approach_%j.err

set -euo pipefail
cd /home/kaushim07/photometry_mcmc_env

source /home/kaushim07/miniforge3/etc/profile.d/conda.sh
conda activate photomc_env

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

echo "Node: $(hostname)"
python .tmp/download_approach.py

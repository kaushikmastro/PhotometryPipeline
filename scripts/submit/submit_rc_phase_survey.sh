#!/usr/bin/env bash
#SBATCH --job-name=rc_phase_survey
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=8G
#SBATCH --partition=main --qos=standard --time=00:10:00
#SBATCH --output=logs/rc_phase_survey_%j.out --error=logs/rc_phase_survey_%j.err

set -euo pipefail
cd /home/kaushim07/photometry_mcmc_env

source /home/kaushim07/miniforge3/etc/profile.d/conda.sh
conda activate photomc_env

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "Node: $(hostname)"
python .tmp/rc_phase_survey.py

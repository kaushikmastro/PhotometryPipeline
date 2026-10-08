#!/usr/bin/env bash
#SBATCH --job-name=talk_slide_pixel_counts
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=28G
#SBATCH --partition=main --qos=standard --time=00:30:00
#SBATCH --output=logs/talk_slide_pixel_counts_%j.out --error=logs/talk_slide_pixel_counts_%j.err

set -euo pipefail
cd /home/kaushim07/photometry_mcmc_env

source /home/kaushim07/miniforge3/etc/profile.d/conda.sh
conda activate photomc_env

echo "Node: $(hostname)"
python .tmp/talk_slide_pixel_counts.py

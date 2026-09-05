#!/bin/bash
#SBATCH --job-name=vesta_download
#SBATCH --output=/home/kaushim07/photometry_mcmc_env/logs/download_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --qos=standard

set -euo pipefail

mkdir -p /home/kaushim07/photometry_mcmc_env/logs

if source /home/kaushim07/miniforge3/bin/activate photomc_env 2>/dev/null; then
	echo "Activated env: photomc_env"
else
	echo "ERROR: Could not activate photomc_env" >&2
	exit 10
fi

cd /home/kaushim07/photometry_mcmc_env/

python scripts/utils/run_ingestion.py
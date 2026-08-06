#!/bin/bash
#SBATCH --job-name=vesta_raytrace
#SBATCH --output=/home/kaushim07/photometry_mcmc_env/logs/compute_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --qos=standard

set -euo pipefail

mkdir -p /home/kaushim07/photometry_mcmc_env/logs

if source /home/kaushim07/miniforge3/bin/activate photometry_mcmc_env 2>/dev/null; then
    echo "Activated env: photometry_mcmc_env"
elif source /home/kaushim07/miniforge3/bin/activate photomc_env 2>/dev/null; then
    echo "Activated env: photomc_env"
else
    echo "ERROR: Could not activate photometry_mcmc_env or photomc_env" >&2
    exit 10
fi

cd /home/kaushim07/photometry_mcmc_env
METAKERNEL_PATH="/home/kaushim07/photometry_mcmc_env/data/spice_kernels/dawn_dynamic.tm"
python scripts/geometry/run_geometry.py --workers "${SLURM_CPUS_PER_TASK:-8}" --metakernel "${METAKERNEL_PATH}"

#!/bin/bash
#SBATCH --job-name=hamo_pilot
#SBATCH --output=/home/kaushim07/photometry_mcmc_env/logs/hamo_pilot_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=85G
#SBATCH --time=01:00:00
#SBATCH --partition=main
#SBATCH --qos=standard

set -euo pipefail

mkdir -p /home/kaushim07/photometry_mcmc_env/logs

source /home/kaushim07/miniforge3/etc/profile.d/conda.sh
conda activate photomc_env

cd /home/kaushim07/photometry_mcmc_env
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

SLICE_FILE="$1"

echo "========================================================"
echo "HAMO pilot job: slice=${SLICE_FILE} workers=14"
echo "========================================================"

python scripts/geometry/run_geometry.py \
    --data-root /scratch/kaushim07/vesta_data \
    --metakernel /scratch/kaushim07/vesta_data/spice_kernels/dawn_dynamic.tm \
    --mode DSK256 \
    --output-subdir geometry/dsk256 \
    --workers 14 \
    --image-list "${SLICE_FILE}"

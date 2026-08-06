#!/bin/bash
#SBATCH --job-name=vesta_fast
#SBATCH --output=/home/kaushim07/photometry_mcmc_env/logs/fast_track_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --qos=standard

set -e

source /home/kaushim07/miniforge3/bin/activate photomc_env
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src:$(pwd)/scripts"

python scripts/run_geometry_fast.py \
    --data-root /scratch/kaushim07/vesta_data \
    --metakernel data/spice_kernels/dawn_dynamic.tm \
    --workers 32 \
    --mode DSK256
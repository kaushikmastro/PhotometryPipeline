#!/bin/bash
#SBATCH --job-name=whole_disk_approach_grind
#SBATCH --output=/home/kaushim07/photometry_mcmc_env/logs/whole_disk_approach_grind_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=main
#SBATCH --qos=standard

set -euo pipefail

mkdir -p /home/kaushim07/photometry_mcmc_env/logs

source /home/kaushim07/miniforge3/etc/profile.d/conda.sh
conda activate photomc_env

cd /home/kaushim07/photometry_mcmc_env
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

SLICE_FILE="$1"

echo "========================================================"
echo "Whole-disk approach grind (2011198_OPNAV_017 + 2011199_OPNAV_018,"
echo "margin_px>=2 filtered, 65 frames): slice=${SLICE_FILE} workers=8"
echo "Node: $(hostname)"
echo "========================================================"

python scripts/geometry/run_geometry.py \
    --data-root /scratch/kaushim07/vesta_data \
    --metakernel /scratch/kaushim07/vesta_data/spice_kernels/dawn_dynamic.tm \
    --mode DSK256 \
    --output-subdir geometry/gaskell_dsk256_110825 \
    --workers 8 \
    --require-area-columns \
    --image-list "${SLICE_FILE}"

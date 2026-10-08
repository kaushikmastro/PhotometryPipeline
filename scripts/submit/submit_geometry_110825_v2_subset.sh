#!/usr/bin/env bash
#SBATCH --job-name=geom_110825_v2_subset
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --partition=main --qos=standard --time=00:30:00
#SBATCH --output=logs/geom_110825_v2_subset_%j.out --error=logs/geom_110825_v2_subset_%j.err

set -euo pipefail
cd /home/kaushim07/photometry_mcmc_env

source /home/kaushim07/miniforge3/etc/profile.d/conda.sh
conda activate photomc_env

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "Node: $(hostname)"
echo "DSK in metakernel: $(grep vesta_gaskell_256 data/spice_kernels/dawn_dynamic.tm)"

python scripts/geometry/run_geometry_110825_v2_subset.py

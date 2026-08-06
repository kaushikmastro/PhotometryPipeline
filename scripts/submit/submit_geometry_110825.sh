#!/usr/bin/env bash
#SBATCH --job-name=geom_110825
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH --partition=main --qos=standard --time=06:00:00
#SBATCH --output=logs/geom_110825_%j.out --error=logs/geom_110825_%j.err

set -euo pipefail
cd /home/kaushim07/photometry_mcmc_env

source /home/kaushim07/miniforge3/etc/profile.d/conda.sh
conda activate photomc_env

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export RAYON_NUM_THREADS=1 POLARS_MAX_THREADS=4
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "DSK in metakernel: $(grep vesta_gaskell_256 data/spice_kernels/dawn_dynamic.tm)"
echo "DSK size on disk:  $(stat -c '%s %n' data/spice_kernels/vesta_gaskell_256_110825.bds)"

python scripts/geometry/run_geometry_110825.py

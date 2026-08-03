#!/bin/bash
#SBATCH --job-name=decomp_fulldata
#SBATCH --output=/home/kaushim07/photometry_mcmc_env/logs/decomp_fulldata_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=main
#SBATCH --qos=standard

set -euo pipefail

source /home/kaushim07/miniforge3/etc/profile.d/conda.sh
conda activate photomc_env

cd /home/kaushim07/photometry_mcmc_env
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== Residual decomposition — full-data committed parameters ==="
echo "Parameters: w=0.46993  g=-0.33688  theta_bar=8.2662  (Hapke-2002 H)"
echo "Source: job 25799987, Config A, 682M pixels, iof>0.01, 950 bins"
echo "Start: $(date)"

echo ""
echo "--- diag_decomp_testB.py ---"
python scripts/diag_decomp_testB.py

echo ""
echo "--- diag_albedo_heterogeneity.py ---"
python scripts/diag_albedo_heterogeneity.py

echo ""
echo "--- diag_albedo_geometry.py ---"
python scripts/diag_albedo_geometry.py

echo ""
echo "--- diag_resolution_diskfn.py ---"
python scripts/diag_resolution_diskfn.py

echo ""
echo "Done: $(date)"

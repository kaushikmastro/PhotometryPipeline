#!/bin/bash
#SBATCH --job-name=vesta_geometry
#SBATCH --output=/home/kaushim07/photometry_mcmc_env/logs/geometry_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32   
#SBATCH --mem=64G            
#SBATCH --time=72:00:00     
#SBATCH --qos=standard

set -euo pipefail

# Guarantee the logging directory is generated on the host file system
mkdir -p /home/kaushim07/photometry_mcmc_env/logs

if source /home/kaushim07/miniforge3/bin/activate photomc_env 2>/dev/null; then
    echo "Activated env: photomc_env"
else
    echo "ERROR: Could not activate photomc_env" >&2
    exit 10
fi

cd /home/kaushim07/photometry_mcmc_env

# SYSTEM PACKAGING INTEGRATION
# Registers your local package structures so all background multiprocessing cores
# resolve namespace structures natively without fallback lookups.
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

echo "========================================================"
echo "Pipeline configuration validation complete."
echo "Running geometry extraction engine via: ${SLURM_CPUS_PER_TASK:-32} cores."
echo "========================================================"

# Execute the hardened multi-processing wrapper with specific model assignments
python scripts/geometry/run_geometry.py \
    --data-root /scratch/kaushim07/vesta_data \
    --metakernel data/spice_kernels/dawn_dynamic.tm \
    --workers "${SLURM_CPUS_PER_TASK:-32}" \
    --mode DSK256
#!/bin/bash
# Re-run geometry computation with updated SPICE kernels
# This script will:
# 1. Cancel any previous jobs
# 2. Clean up failed outputs
# 3. Submit new SLURM job with August 2011 SPK coverage

set -e

WD="/home/kaushim07/photometry_mcmc_env"
DATA_ROOT="$WD/data"
LOGS_DIR="$WD/logs"

echo "=========================================================================="
echo "Geometry Re-Run: SPICE Kernels Updated with August 2011 SPK Coverage"
echo "=========================================================================="
echo ""

# Step 1: Cancel any previous job
echo "[1/4] Checking for running geometry jobs..."
if squeue -u kaushim07 -h | grep -q "vesta_raytrace"; then
    OLD_JOB=$(squeue -u kaushim07 -h -o "%.18i %.9P %.20j" | grep "vesta_raytrace" | awk '{print $1}')
    echo "  Found running job: $OLD_JOB (CANCELLED)"
    scancel "$OLD_JOB" || true
    sleep 2
else
    echo "  No running jobs found"
fi
echo ""

# Step 2: Verify dynamic metakernel
echo "[2/4] Verifying SPICE kernel setup..."
if [[ ! -f "$DATA_ROOT/02_spice_kernels/dawn_dynamic.tm" ]]; then
    echo "  ERROR: Dynamic metakernel not found!"
    exit 1
fi

# Check August 2011 SPK coverage
if ! grep -q "110802-110831" "$DATA_ROOT/02_spice_kernels/dawn_dynamic.tm"; then
    echo "  WARNING: August 2011 SPK kernel may be missing"
else
    echo "  ✓ August 2011 SPK coverage confirmed"
fi

# Count kernels
NK=$(grep -c "\.bsp\|\.tsc\|\.bc\|\.tf\|\.ti\|\.tpc\|\.tls" "$DATA_ROOT/02_spice_kernels/dawn_dynamic.tm" || true)
echo "  ✓ Dynamic metakernel ready with ~$NK SPICE kernels"
echo ""

# Step 3: Prepare for re-run
echo "[3/4] Preparing geometry computation..."
echo "  Data root: $DATA_ROOT"
echo "  Metakernel: dawn_dynamic.tm"
echo "  Target images: 3 (FC21B0003931, FC21B0003932, FC21B0003933)"
echo "  Expected output: 3 parquet files in 04_geometry_tables/"
echo ""

# Step 4: Submit SLURM job
echo "[4/4] Submitting SLURM batch job..."
cd "$WD"
JOB_ID=$(sbatch --parsable scripts/submit_compute.sh)
echo "  Job submitted: $JOB_ID"
echo "  Follow progress with:"
echo "    squeue -j $JOB_ID"
echo "    tail -f $LOGS_DIR/compute_${JOB_ID}.log"
echo ""

echo "=========================================================================="
echo "Re-run initiated! Monitor with squeue or tail -f the log file."
echo "=========================================================================="

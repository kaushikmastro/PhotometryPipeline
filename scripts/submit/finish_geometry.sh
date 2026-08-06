#!/bin/bash
#SBATCH --job-name=vesta_ra_finish
#SBATCH --output=/home/kaushim07/photometry_mcmc_env/logs/finish_geometry_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --qos=standard

set -euo pipefail

WORKDIR="/home/kaushim07/photometry_mcmc_env"
DATA_ROOT="$WORKDIR/data"
IMAGE_ROOT="$DATA_ROOT/calibrated_raw_images"
OUTPUT_ROOT="$DATA_ROOT/04_geometry_tables"
LOG_DIR="$WORKDIR/logs"
METAKERNEL_PATH="$DATA_ROOT/spice_kernels/dawn_dynamic.tm"
TMP_MANIFEST="$LOG_DIR/finish_geometry_manifest_${SLURM_JOB_ID:-manual}.csv"

mkdir -p "$LOG_DIR"

if source /home/kaushim07/miniforge3/bin/activate photometry_mcmc_env 2>/dev/null; then
    echo "Activated env: photometry_mcmc_env"
elif source /home/kaushim07/miniforge3/bin/activate photomc_env 2>/dev/null; then
    echo "Activated env: photomc_env"
else
    echo "ERROR: Could not activate photometry_mcmc_env or photomc_env" >&2
    exit 10
fi

cd "$WORKDIR"

echo "image_filename,phase_subdir" > "$TMP_MANIFEST"

queued=0
skipped=0
missing_dirs=0
survey_total=0
survey_pending=0
rc_total=0
rc_pending=0

for phase in survey rc; do
    phase_img_dir="$IMAGE_ROOT/$phase"

    if [[ ! -d "$phase_img_dir" ]]; then
        echo "WARNING: Missing input directory: $phase_img_dir"
        missing_dirs=$((missing_dirs + 1))
        continue
    fi

    while IFS= read -r -d '' img_path; do
        image_filename="$(basename "$img_path")"
        image_stem="${image_filename%.IMG}"
        image_stem_upper="${image_stem^^}"
        output_path="$OUTPUT_ROOT/$phase/${image_stem_upper}_geometry.parquet"

        if [[ "$phase" == "survey" ]]; then
            survey_total=$((survey_total + 1))
        elif [[ "$phase" == "rc" ]]; then
            rc_total=$((rc_total + 1))
        fi

        # Resume logic: skip if geometry parquet already exists.
        if [[ -f "$output_path" ]]; then
            skipped=$((skipped + 1))
            continue
        fi

        echo "$image_filename,$phase" >> "$TMP_MANIFEST"
        queued=$((queued + 1))
        if [[ "$phase" == "survey" ]]; then
            survey_pending=$((survey_pending + 1))
        elif [[ "$phase" == "rc" ]]; then
            rc_pending=$((rc_pending + 1))
        fi
    done < <(find "$phase_img_dir" -maxdepth 1 -type f -name "*.IMG" -print0 | sort -z)
done

hamo_ignored="yes"
lamo_ignored="yes"
hamo_count=0
lamo_count=0
if [[ -d "$IMAGE_ROOT/hamo" ]]; then
    hamo_count=$(find "$IMAGE_ROOT/hamo" -maxdepth 1 -type f -name "*.IMG" | wc -l | tr -d ' ')
fi
if [[ -d "$IMAGE_ROOT/lamo" ]]; then
    lamo_count=$(find "$IMAGE_ROOT/lamo" -maxdepth 1 -type f -name "*.IMG" | wc -l | tr -d ' ')
fi

echo ""
echo "==================== Geometry Resume Preflight ===================="
echo "Survey images: total=${survey_total} pending=${survey_pending}"
echo "RC images:     total=${rc_total} pending=${rc_pending}"
echo "HAMO ignored:  ${hamo_ignored} (discovered ${hamo_count} IMG files; excluded by design)"
echo "LAMO ignored:  ${lamo_ignored} (discovered ${lamo_count} IMG files; excluded by design)"
echo "==================================================================="
echo ""

echo "Resume manifest: $TMP_MANIFEST"
echo "Queued images: $queued"
echo "Skipped existing parquet outputs: $skipped"

if [[ "$queued" -eq 0 ]]; then
    echo "No pending Survey/RC images found. Nothing to process."
    exit 0
fi

if [[ ! -f "$METAKERNEL_PATH" ]]; then
    echo "ERROR: Metakernel not found: $METAKERNEL_PATH" >&2
    exit 11
fi

python scripts/geometry/run_geometry.py \
    --data-root "$DATA_ROOT" \
    --manifest "$TMP_MANIFEST" \
    --metakernel "$METAKERNEL_PATH" \
    --workers "${SLURM_CPUS_PER_TASK:-8}"

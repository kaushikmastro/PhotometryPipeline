#!/bin/bash
# Move-don't-delete quarantine for orphaned committed parquets (same pattern
# used for the survey/ contamination fix tonight). Never deletes anything.
#
# Usage:
#   scripts/quarantine_contaminated.sh <phase> <stem1> [stem2] [stem3] ...
#
# Example:
#   scripts/quarantine_contaminated.sh lamo FC21B0012345_11223344556F1C FC21B0099999_99887766554F1D

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "ERROR: missing arguments. Usage: $0 <phase> <stem1> [stem2] ..." >&2
  exit 1
fi

PHASE="$1"
shift

BASE_DIR="/scratch/kaushim07/vesta_data/geometry/dsk256"
SRC_DIR="${BASE_DIR}/${PHASE}"
DEST_DIR="${BASE_DIR}/${PHASE}_user_roi_contamination"

if [ ! -d "$SRC_DIR" ]; then
  echo "ERROR: source directory does not exist: $SRC_DIR" >&2
  exit 1
fi

mkdir -p "$DEST_DIR"

moved=0
missing=0
for stem in "$@"; do
  src="${SRC_DIR}/${stem}_geometry.parquet"
  if [ -f "$src" ]; then
    mv -v "$src" "$DEST_DIR/"
    moved=$((moved + 1))
  else
    echo "WARNING: expected file not found, skipping: $src" >&2
    missing=$((missing + 1))
  fi
done

echo "Quarantine complete for phase '${PHASE}': moved=${moved} missing=${missing}"
echo "Destination: ${DEST_DIR}"

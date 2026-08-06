#!/usr/bin/env bash
# Download true LAMO calibrated images from PDS SBN.
# Fetches the cumulative index, filters for F1B filter and DOY 12340-13243
# (Dec 5 2012 – Aug 31 2013), then downloads .IMG + .LBL pairs.
# Safe to re-run (skips existing files).
#
#SBATCH --job-name=lamo_img_dl
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2
#SBATCH --mem=8G --time=48:00:00
#SBATCH --partition=main --qos=standard
#SBATCH --output=logs/lamo_img_download_%j.out

set -euo pipefail
cd /home/kaushim07/photometry_mcmc_env
mkdir -p logs

PDS_BASE="https://sbn.psi.edu/holdings/dawn-a-fc2-3-rcal-v1.0"
OUT_DIR="/scratch/kaushim07/vesta_data/calibrated_raw_images/lamo_true"
LOG="logs/lamo_img_download_${SLURM_JOB_ID:-local}.log"

mkdir -p "$OUT_DIR"
echo "[$(date '+%F %T')] LAMO image download starting" | tee "$LOG"
echo "[$(date '+%F %T')] Source: $PDS_BASE" | tee -a "$LOG"
echo "[$(date '+%F %T')] Target: $OUT_DIR" | tee -a "$LOG"
echo "[$(date '+%F %T')] Filter: F1B, DOY 12340-13243 (Dec 2012 – Aug 2013)" | tee -a "$LOG"

source /home/kaushim07/miniforge3/etc/profile.d/conda.sh
conda activate photomc_env

# -- Step 1: download the PDS cumulative index --
IDX_DIR=$(mktemp -d /tmp/pds_idx.XXXXXX)
IDX_TAB="${IDX_DIR}/cumindex.tab"
IDX_FOUND=""

echo "" | tee -a "$LOG"
echo "[$(date '+%F %T')] Fetching PDS cumulative index..." | tee -a "$LOG"

for path in "index/cumindex.tab" "index/CUMINDEX.TAB" "index/index.tab" "index/INDEX.TAB"; do
    url="${PDS_BASE}/${path}"
    if wget --quiet --tries=3 --timeout=120 -O "${IDX_TAB}.tmp" "${url}" 2>/dev/null \
            && [[ -s "${IDX_TAB}.tmp" ]]; then
        mv "${IDX_TAB}.tmp" "$IDX_TAB"
        IDX_FOUND="$url"
        echo "[$(date '+%F %T')] Index found at: $url" | tee -a "$LOG"
        echo "[$(date '+%F %T')] Index rows: $(wc -l < "$IDX_TAB")" | tee -a "$LOG"
        break
    fi
    rm -f "${IDX_TAB}.tmp"
done

if [[ -z "$IDX_FOUND" ]]; then
    echo "[$(date '+%F %T')] ERROR: could not fetch PDS index from any of:" | tee -a "$LOG"
    echo "  ${PDS_BASE}/index/{cumindex,CUMINDEX,index,INDEX}.tab" | tee -a "$LOG"
    echo "  Check the volume URL at: ${PDS_BASE}/" | tee -a "$LOG"
    exit 1
fi

# -- Step 2: extract LAMO F1B URLs from index using Python --
echo "" | tee -a "$LOG"
echo "[$(date '+%F %T')] Extracting LAMO F1B entries from index..." | tee -a "$LOG"

URL_LIST="/tmp/lamo_f1b_urls_${SLURM_JOB_ID:-local}.txt"

python3 - "$IDX_TAB" "$URL_LIST" <<'PYEOF'
import sys, re

idx_path, out_path = sys.argv[1], sys.argv[2]

# PDS3 index is a fixed-width ASCII table. We use regex to find:
#   - The relative path (FILE_SPECIFICATION_NAME or DATA/.../FILENAME)
#   - F1B filter: filename ends in F1B.IMG
#   - LAMO epoch: YYDDD component of filename in range [12340, 13243]
#
# File naming: FC21B{seq}_{YYDDDHHMMSS}F1B.IMG
#              The 5-digit YYDDD is chars 8-12 of the time-tag field.

found = []
header_printed = False

with open(idx_path, errors="replace") as f:
    for i, raw in enumerate(f):
        line = raw.strip()

        # Skip PDS label block (first lines often start with PDS_VERSION_ID etc.)
        if not line or line.startswith("PDS") or line.startswith("/*"):
            continue

        # Look for F1B image filename pattern in this row
        m = re.search(
            r'["\s]([A-Z0-9/_-]*FC2\w+?_(\d{5})\d{6}F1B\.IMG)["\s,]',
            line, re.IGNORECASE
        )
        if not m:
            continue

        raw_path = m.group(1).strip('/"\'')
        yyddd = int(m.group(2))

        if not (12340 <= yyddd <= 13243):
            continue

        # raw_path may be relative from volume root (e.g. "DATA/DIR/FILE.IMG")
        # or just a filename. Normalise to a relative URL path.
        if not raw_path.upper().startswith("DATA"):
            # If no directory prefix, infer from YYDDD convention
            year  = 2000 + int(str(yyddd)[:2])
            doy   = int(str(yyddd)[2:])
            bname = raw_path.split("/")[-1].upper()
            raw_path = f"data/{year}_{doy:03d}/{bname}"

        found.append(raw_path.upper())

        if not header_printed:
            print(f"First match (row {i+1}): {raw_path}  yyddd={yyddd}")
            header_printed = True

# Deduplicate (index may have duplicate rows) and sort
found = sorted(set(found))
with open(out_path, "w") as out:
    out.write("\n".join(found) + "\n" if found else "")

print(f"LAMO F1B images extracted: {len(found)}")
if not found:
    # Print first 3 non-header lines to help debug column layout
    with open(idx_path, errors="replace") as f:
        count = 0
        for line in f:
            if line.strip() and not line.startswith("PDS"):
                print(f"  Sample row: {line[:120]!r}")
                count += 1
                if count >= 3:
                    break
PYEOF

N_URLS=$(grep -c . "$URL_LIST" 2>/dev/null || echo 0)
echo "[$(date '+%F %T')] LAMO F1B URLs to download: $N_URLS" | tee -a "$LOG"
rm -rf "$IDX_DIR"

if [[ "$N_URLS" -eq 0 ]]; then
    echo "[$(date '+%F %T')] No URLs found — check index format above." | tee -a "$LOG"
    echo "  The index column layout may differ from the expected FC2 naming pattern." | tee -a "$LOG"
    echo "  Inspect: ${IDX_FOUND}" | tee -a "$LOG"
    exit 1
fi

# -- Step 3: download .IMG + .LBL pairs --
echo "" | tee -a "$LOG"
echo "[$(date '+%F %T')] Beginning image download ($N_URLS files)..." | tee -a "$LOG"

N_DL=0; N_SKIP=0; N_FAIL=0

while IFS= read -r rel_path; do
    [[ -z "$rel_path" ]] && continue

    fname=$(basename "$rel_path")
    stem="${fname%.IMG}"
    dest_img="${OUT_DIR}/${fname}"
    dest_lbl="${OUT_DIR}/${stem}.LBL"

    if [[ -f "$dest_img" ]]; then
        ((N_SKIP++)) || true
        continue
    fi

    url_img="${PDS_BASE}/${rel_path}"
    url_lbl="${PDS_BASE}/$(dirname "$rel_path")/${stem}.LBL"

    if wget --quiet --tries=5 --timeout=300 -O "${dest_img}.tmp" "$url_img"; then
        mv "${dest_img}.tmp" "$dest_img"
        wget --quiet --tries=3 --timeout=60 -O "$dest_lbl" "$url_lbl" 2>/dev/null || true
        ((N_DL++)) || true
        if (( N_DL % 500 == 0 )); then
            echo "[$(date '+%F %T')]  $N_DL downloaded so far  (skip=$N_SKIP fail=$N_FAIL)" \
                | tee -a "$LOG"
        fi
    else
        rm -f "${dest_img}.tmp"
        echo "[$(date '+%F %T')]  FAIL: $url_img" | tee -a "$LOG"
        ((N_FAIL++)) || true
    fi
done < "$URL_LIST"

rm -f "$URL_LIST"

echo "" | tee -a "$LOG"
echo "[$(date '+%F %T')] === Download complete ===" | tee -a "$LOG"
echo "  Downloaded : $N_DL" | tee -a "$LOG"
echo "  Skipped    : $N_SKIP (already present)" | tee -a "$LOG"
echo "  Failed     : $N_FAIL" | tee -a "$LOG"
echo "[$(date '+%F %T')] Files in $OUT_DIR: $(ls "$OUT_DIR" | wc -l)" | tee -a "$LOG"
[[ "$N_FAIL" -gt 0 ]] && echo "  WARNING: $N_FAIL failures — re-run to retry" | tee -a "$LOG"
echo "[$(date '+%F %T')] DONE." | tee -a "$LOG"
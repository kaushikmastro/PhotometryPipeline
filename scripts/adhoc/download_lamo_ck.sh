#!/usr/bin/env bash
# Download LAMO-epoch FC2 CK kernels from NAIF and insert them into dawn_dynamic.tm.
# Submits as a SLURM job; safe to re-run (skips existing files, idempotent metakernel update).
#
#SBATCH --job-name=lamo_ck_dl
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --mem=4G --time=04:00:00
#SBATCH --partition=main --qos=standard
#SBATCH --output=logs/lamo_ck_download_%j.out

set -euo pipefail
cd /home/kaushim07/photometry_mcmc_env
mkdir -p logs

SPICE_CK="/scratch/kaushim07/vesta_data/spice_kernels/ck"
NAIF_BASE="https://naif.jpl.nasa.gov/pub/naif/pds/data/dawn-m_a-spice-6-v1.0/dawnsp_1000/data/ck"
METAKERNEL="data/spice_kernels/dawn_dynamic.tm"
LOG="logs/lamo_ck_download_${SLURM_JOB_ID:-local}.log"

echo "[$(date '+%F %T')] LAMO CK download starting" | tee "$LOG"
echo "[$(date '+%F %T')] Source: $NAIF_BASE" | tee -a "$LOG"
echo "[$(date '+%F %T')] Target: $SPICE_CK" | tee -a "$LOG"

# -- Steps 1+2: generate weekly CK URLs directly and download --
# NAIF's directory page is styled HTML that wget cannot parse for hrefs.
# Instead, generate the known weekly filename pattern for the full LAMO epoch.
echo "[$(date '+%F %T')] Generating LAMO weekly CK URLs (Jan 7 – Sep 15 2013)..." | tee -a "$LOG"

python3 - <<PYEOF 2>&1 | tee -a "$LOG"
from datetime import date, timedelta
import subprocess, os

NAIF_BASE = "https://naif.jpl.nasa.gov/pub/naif/pds/data/dawn-m_a-spice-6-v1.0/dawnsp_1000/data/ck"
OUT_DIR = "/scratch/kaushim07/vesta_data/spice_kernels/ck"

# Generate weekly ranges from Jan 7 2013 to Sep 15 2013
start = date(2013, 1, 7)
end   = date(2013, 9, 15)
d = start
urls = []
while d <= end:
    d_end = d + timedelta(days=6)
    # Try both naming variants NAIF uses
    for fmt in [
        f"dawn_sc_{d.strftime('%y%m%d')}_{d_end.strftime('%y%m%d')}.bc",
        f"dawn_sc_{d.strftime('%y%m%d')}_{d_end.strftime('%y%m%d')}_v1.bc",
        f"dawn_ql_{d.strftime('%y%m%d')}_{d_end.strftime('%y%m%d')}.bc",
    ]:
        urls.append((fmt, f"{NAIF_BASE}/{fmt}"))
    d += timedelta(days=7)

print(f"Trying {len(urls)} URL candidates ({len(urls)//3} weeks x 3 naming variants)")

downloaded, skipped, failed = 0, 0, 0
for fname, url in urls:
    out = os.path.join(OUT_DIR, fname)
    if os.path.exists(out):
        print(f"SKIP (exists): {fname}")
        skipped += 1
        continue
    result = subprocess.run(
        ["wget", "-q", "--timeout=30", "--tries=3",
         "-O", out, url],
        capture_output=True
    )
    if result.returncode == 0 and os.path.getsize(out) > 1000:
        print(f"OK: {fname} ({os.path.getsize(out):,} bytes)")
        downloaded += 1
    else:
        if os.path.exists(out):
            os.remove(out)  # remove empty/partial file
        # Not a failure — file may not exist for this week
        print(f"NOT FOUND: {fname}")
        failed += 1

print(f"\nSummary: {downloaded} downloaded, {skipped} skipped, "
      f"{failed} not found (expected -- not all weeks have files)")
PYEOF

# -- Step 3: update dawn_dynamic.tm (idempotent) --
echo "" | tee -a "$LOG"
echo "[$(date '+%F %T')] Updating metakernel: $METAKERNEL" | tee -a "$LOG"

python3 - <<PYEOF
import os, re, sys, shutil, datetime

mk = "data/spice_kernels/dawn_dynamic.tm"
ck_dir = "/scratch/kaushim07/vesta_data/spice_kernels/ck"
log_path = "logs/lamo_ck_download_${SLURM_JOB_ID:-local}.log"

# Collect all dawn_sc_13*.bc now in ck_dir, sorted chronologically
new_files = sorted(
    f for f in os.listdir(ck_dir)
    if re.match(r'^dawn_sc_13\d{6}_\d{6}.*\.bc$', f)
)
print(f"LAMO CK files found in ck_dir: {len(new_files)}")

if not new_files:
    print("No dawn_sc_13*.bc in ck_dir — metakernel unchanged")
    sys.exit(0)

lines = open(mk).readlines()

# Idempotency check — if first file already in metakernel, skip
if any(new_files[0] in l for l in lines):
    print(f"Metakernel already contains {new_files[0]} — no change (idempotent)")
    sys.exit(0)

# Find DSK line (insertion point: new CKs go immediately before it)
dsk_idx = next(
    (i for i, l in enumerate(lines) if "vesta_gaskell_256" in l), None
)
if dsk_idx is None:
    print("ERROR: DSK entry ('vesta_gaskell_256') not found in metakernel",
          file=sys.stderr)
    sys.exit(1)

# Each new CK entry needs a trailing comma (DSK line has none — keep it that way)
new_entries = [f"  '\$SPICE/ck/{f}',\n" for f in new_files]

# Backup before modifying
ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
bak = mk + f".bak.{ts}"
shutil.copy2(mk, bak)
print(f"Backup written: {bak}")

# Splice in new entries before the DSK line
lines = lines[:dsk_idx] + new_entries + lines[dsk_idx:]
open(mk, "w").writelines(lines)

msg = (f"Metakernel updated: inserted {len(new_entries)} LAMO CK entries "
       f"before line {dsk_idx+1}. Backup: {bak}")
print(msg)
with open(log_path, "a") as lf:
    lf.write(msg + "\n")
PYEOF

echo "" | tee -a "$LOG"
echo "[$(date '+%F %T')] Final dawn_sc_13*.bc count in $SPICE_CK:" | tee -a "$LOG"
ls "$SPICE_CK"/dawn_sc_13*.bc 2>/dev/null | wc -l | tee -a "$LOG"
echo "[$(date '+%F %T')] DONE." | tee -a "$LOG"

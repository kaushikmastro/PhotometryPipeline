# Photometry MCMC — Project Context for Claude

## CLUSTER COMPUTE RULES — MANDATORY

This project runs on Curta HPC at FU Berlin.
The login node (login.curta.zedat.fu-berlin.de) is SHARED.
Running heavy compute on the login node is a cluster
policy violation and affects all users.

RULE 1: Any DuckDB query on >100M pixels MUST run via srun:
  srun --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G \
       --partition=main --qos=standard --time=00:30:00 \
       --pty python3 << 'PYEOF'
  [query here]
  PYEOF

RULE 2: Any geometry processing MUST use sbatch scripts.
  Never run geometry_engine directly on login node.

RULE 3: MCMC sampling MUST use sbatch scripts.
  Never run emcee/PyMC3 directly on login node.

RULE 4: Before running ANY python script, check:
  hostname  # must NOT show login.curta.zedat.fu-berlin.de

RULE 5: Simple file checks (ls, wc -l, grep, head) are
  acceptable on the login node.
  DuckDB on parquet files > 1GB: always srun.

VIOLATION of these rules causes login node saturation
for all cluster users. This has happened multiple times
in this project. It must not happen again.

## Project
Hapke photometric model fitting on Dawn/FC Vesta Survey F1B disk-resolved data.
Shape model: preliminary Gaskell DSK256 (f_solar=892, `04_geometry_tables_fast/`).
HPC: Curta, conda env `photomc_env`.

## Committed Case 1 Result (preliminary DSK, illuminated regime — FULL DATA)

| parameter | value | source |
|---|---|---|
| w | 0.46993 | `prelim_physfilter_25799987.out`, Config A |
| g | −0.33688 | |
| theta_bar | 8.2662° | |
| fRMS | **10.902%** | illuminated regime only (see filter below) |
| reduced_chi² | 0.200 | |
| n_bins | 950 | |
| H-function | Hapke-2002 (default, `isotropic_h=False`) | `run_prelim_physfilter.py` |
| B0, h | 1.03, 0.04 | fixed |

**These are the paper numbers. Do not change without explicit re-fitting.**

### Superseded 1% Sample Result (archived, do not use downstream)

| parameter | value | source |
|---|---|---|
| w | 0.4626 | `hapke_fit_25749445.out` |
| g | −0.3323 | |
| theta_bar | 5.496° | |
| fRMS | 10.135% | |

**Reason superseded**: theta_bar was sampling-sensitive. 1% sample used corrected DSK256 parquet (~4,500 px/bin at i=45°); full data uses raw geometry tables (~306,000 px/bin). At high incidence (i=45°), 1% sample systematically underestimates I/F by ~0.9%, reducing the apparent roughness signal. Full-data fit is deterministic (100-start spread: θ̄ std=0.00006°). w and g are stable across both fits.

## Validated Shadow Filter: iof > 0.01

The `iof > 0.01` pre-filter applied in `extract_resolved_sample_hapke_dsk256.py` is a **validated shadow exclusion filter**, not a brightness selection bias. Evidence:

- **Physical floor**: darkest known Vesta terrain (albedo ~7%) at i=50° gives iof_min ≈ 0.045, which is 4.5× above the 0.01 cut. No illuminated Vesta surface in the i<50° domain can have iof < 0.01.
- **Distribution gap**: within i<50°, e<50°, 18.62% of pixels have iof<0.01; of those, 14.2% are deep shadow (iof<0.001), 4.4% cast/partial shadow. No continuous population exists between iof=0.01 and the physical minimum.
- **Mean iof of dark pixels**: 0.00076 — consistent with shadow, not dark albedo.
- **Incidence distribution of dark pixels**: mean 55.5° but mean of dark pixels IN the fit domain (i<50°,e<50°) = 37.3° — these are topographic cast shadows inside moderate-incidence bins, not high-incidence penumbra.

**Correct framing**: "Case 1 achieves 10.902% fRMS on the illuminated Survey F1B regime (iof>0.01, i<50°, e<50°, 950 bins). Including shadow-contaminated bins inflates fRMS to ~25% without changing fitted parameters' physical validity."

Without the shadow filter (full data): fRMS=24.975%, theta_bar=3.592°, w=0.3806 (from `prelim_physfilter_25799987.out`, Config B, 949 bins).

## Data Sources

| parquet | rows | iof filter | notes |
|---|---|---|---|
| `binned_prelim_iof001.parquet` | 950 bins | >0.01 | **headline fit input**; full 682M pixels, iof>0.01 (Config A) |
| `binned_prelim_physfilter.parquet` | 949 bins | none | Config B (no iof cut), reference only, 24.975% |
| `combined_rc_survey_sample_corrected_dsk256.parquet` | 6.7M (Survey) | >0.01 | 1% sample — archived, superseded by full-data fit |
| `survey_5pct.parquet` | 33.5M | >0.01 | residual decomposition input |
| `binned_prelim_1pct.parquet` | 934 bins | none | 1% sample, no iof cut, 26.442% — archived |
| `binned_survey_110825.parquet` | 944 bins | none | 110825 DSK full data, 30.979% |

## Residual Decomposition

All four diagnostic scripts use the **full-data committed parameters and Hapke-2002 H-function**:
- `diag_decomp_testB.py` — orthogonal CV decomposition (CV_trend, CV_other)
- `diag_albedo_heterogeneity.py` — N/S albedo signal (survey_5pct.parquet)
- `diag_albedo_geometry.py` — surface-locked vs geometry artifact test
- `diag_resolution_diskfn.py` — 18m vs 7x7 disk-function comparison

**Last confirmed run** (job 25799983, theta_bar=5.496 — STALE, superseded):  
CV_trend=5.13%, CV_other=8.31%, albedo r=0.810.

**Pending re-run** with full-data params (job 25801546, theta_bar=8.2662):  
`submit_decomp_fulldata.sh` submitted. Will update values once complete.

## Completed Results

- Job 25799987: full-data Case 1 fit — **headline result confirmed**. Config A fRMS=10.902%, theta_bar=8.2662°. Config B fRMS=24.975%.
- Job 25799983: decomposition at theta_bar=5.496 — **superseded** by job 25801546.

## Key Scripts

| script | purpose |
|---|---|
| `scripts/run_baseline_fit.py` | headline Case 1/2/3 fit; reads combined parquet |
| `scripts/run_prelim_physfilter.py` | full-data both-filter fit; outputs binned_prelim_iof001.parquet |
| `scripts/submit_prelim_physfilter.sh` | SLURM for above |
| `scripts/submit_decomp_fulldata.sh` | SLURM for decomposition at full-data params (job 25801546) |
| `scripts/run_controlled_comparison.py` | 1% controlled prelim vs 110825 comparison |

## H-function Note

`isotropic_h=True` = Li et al. 2013 IMSA form H(x)=(1+2x)/(1+2γx).  
`isotropic_h=False` (default) = Hapke 2002 approximation.  
The committed Case 1 and all production fits use `isotropic_h=False`.  
The decomposition diagnostics previously used `isotropic_h=True` with stale parameters — this has been corrected.

## Data Provenance

=== DATA PROVENANCE (as of June 21, 2026) ===

GEOMETRY TABLES — 04_geometry_tables_fast/

| Phase   | Parquets | Pixels        | mean_iof | f_solar | Status          |
|---------|----------|---------------|----------|---------|-----------------|
| RC      | 423      | 140,883,994   | 0.125    | 892.0   | VERIFIED        |
| Survey  | 1153     | 982,431,748   | 0.103    | 892.0   | VERIFIED        |
| HAMO    | 1089     | 1,141,899,264 | 0.106    | 892.0   | VERIFIED Jun 15 |
| LAMO    | 4,349    | 4,560,257,024 | 0.082    | 892.0   | VERIFIED Jun 19 |

SUPERSEDED: 04_geometry_tables/ (older pipeline, not used)

DELETED: 04_geometry_tables_dsk256_WRONG_FSOLAR_1473/ — removed Jun 21 2026.
  Built with f_solar=1473.4 (wrong). Controlled comparison result (1.6518×
  scaling factor) committed to CLAUDE.md. No longer needed.

NOTE: LAMO mean_iof=0.082 is lower than Survey/HAMO (0.103/0.106) due to
  mean_inc=61.2° (late-LAMO solar geometry + OpNav images at extreme geometry).
  max_inc=123.6° indicates shadow pixels in raw tables — removed by iof>0.01
  silver layer filter. triangle_violations=0.00203% (shadow pixel geometry
  edge cases, below 0.01% threshold). px/image=1,048,576 = full 1024×1024 FC2
  chip (body fills FOV at 210km altitude, same as HAMO at 680km).

GEOMETRY TABLES — 04_geometry_tables_dsk256_110825/ (MISSION-SCIENCE DSK)

| Phase  | Parquets | Pixels          | mean_iof | f_solar | Status          |
|--------|----------|-----------------|----------|---------|-----------------|
| Survey | 845      | (no sentinel)   | —        | 892.0   | from prior run  |
| HAMO   | 1,089    | 1,141,899,264   | 0.106    | 892.0   | VERIFIED Jun 20 |
| LAMO   | 4,349    | 4,560,257,024   | 0.082    | 892.0   | VERIFIED Jun 21 |

GEOMETRY TABLES — 04_geometry_tables_ellipsoid/ (TRIAXIAL ELLIPSOID)

| Phase  | Parquets | Pixels      | mean_iof_daylight | f_solar | Status          |
|--------|----------|-------------|-------------------|---------|-----------------|
| Survey | 816      | 679,710,522 | 0.137             | 892.0   | VERIFIED Jun 21 |

NOTE: Ellipsoid Survey: mean_iof_all=0.100, mean_iof_daylight=0.137 (iof>0.01, inc<80°).
  816 images = 845 F1B total - 29 approach images (DOY 11123).
  679M px < 816M expected (full-frame) because body doesn't fill all corners of
  FOV for every Survey-phase pointing at 2700 km altitude.

=== CALIBRATED IMAGES — CORRECTED INVENTORY ===

data/01_calibrated_images/hamo/
  DOY 2011-246 to ~2012-196 (HAMO-1 epoch)
  Contains: HAMO-1 imagery + overlap with LAMO epoch

data/01_calibrated_images/lamo/
  DOY 2011-312 to 2012-121 (Nov 2011 – May 2012)
  Contains: REAL VESTA LAMO DATA — COMPLETE (4,349 F1B images)
  NOT mislabelled — this IS the LAMO-phase data

  Breakdown:
    192 F1B  DOY 11312–11345  Transfer-to-LAMO (OpNav observations)
    4,157 F1B DOY 12077–12118  LAMO CYCLE15–CYCLE20 (late LAMO)
    0 F1B    CYCLE1–CYCLE14    Dawn did NOT do F1B imaging early LAMO

  PDS verification (Jun 2026): PDS INDEX.TAB has exactly 4,349 F1B
  LAMO+Transfer-to-LAMO entries; all match on-disk filenames.
  Early LAMO cycles (CYCLE1–CYCLE14) exist in PDS DATA directory
  but contain only non-F1B filter images — no F1B data to download.
  On-disk set is COMPLETE — no additional download needed.

LAMO F1B coverage per PDS: DOY 11312–12118
PDS volume: sbnarchive.psi.edu/pds3/dawn/fc/DWNVFC2_1B/

HAMO geometry tables were built from both hamo/ + lamo/ correctly
(lamo/ holds real LAMO-epoch images, not a mislabelling).

=== SPICE CK COVERAGE ===

Present (in data/02_spice_kernels/ and ck/ subdirectory):
  - dawn_fc2_110723_120725_grv221108_v1.bc: Jul 2011 – Jul 2012
  - 283 quicklook/weekly CK segments: Sep 2007 – Jan 6 2013

CK FULLY COVERS VESTA LAMO (DOY 11346–12167):
  Precision FC2 CK (Jul 2011 – Jul 2012) covers entire LAMO epoch.
  Quicklook CKs through Jan 6 2013 also span this range.
  No additional CK download needed for LAMO geometry grind.

NOTE: 36 dawn_sc_13*.bc files (cruise-to-Ceres, Jan–Sep 2013)
  were downloaded in error Jun 2026 and remain in scratch/ck/.
  They have been removed from dawn_dynamic.tm (backup .bak.20260616*).
  Irrelevant for Vesta pipeline — no further action needed.

KNOWN GAP (Jul 19 2026): 49 HAMO F1D images excluded — unrecoverable CK gap.
  DOY 2012-181 (Jun 29), ~07:00-15:00 TDB region contains a genuine
  spacecraft attitude-telemetry gap. Confirmed via direct SPICE
  ckgp/pxform probing, not just ckcov file-boundary metadata — ckcov
  reports the segment as "covered" but per-epoch pointing lookups fail
  throughout the window, consistent with a real CK Type 3 internal gap
  (segment-level boundary != gap-free interior coverage). Not a metakernel
  loading omission: dawn_sc_120625_120701.bc IS loaded (confirmed present
  in the active kernel pool) and its file-level date range nominally
  covers this window, but the actual telemetry is missing at this specific
  epoch. Checked all 324 CK files on disk (both loaded and unloaded) for
  -203000 (DAWN_SPACECRAFT) coverage of this window: none exist anywhere.
  Unrecoverable with data currently on disk. Excluded from geometry
  tables; HAMO all-letter final: 5,498/5,547 (99.1%).

=== OPEN ITEMS ===

[x] LAMO geometry grind — COMPLETE (Jun 19 2026, job 25886790)
    4,349 parquets, 4,560,257,024 pixels, mean_iof=0.082, f_solar=892 VERIFIED
    Validation job 25892508: all PASS/FAIL checks passed.
    [ ] Build LAMO silver layer (next step):
        Mirror Survey silver build; filter F1B, i<80, e<80, iof>0.01
        Source: 04_geometry_tables_fast/lamo/
        Output: 06_silver_layer_lamo/lamo_dsk256.parquet

[x] RC mean_iof formal verification — CLOSED (0.125 confirmed)

[ ] HAMO silver layer build — READY TO START
    Mirror Survey silver build:
    - Filter: F1B only, i<80, e<80, iof>0.01
    - Source: 04_geometry_tables_fast/hamo/
    - Output: 06_silver_layer_hamo/hamo_dsk256.parquet
    - Note: include both hamo/ and lamo/ geometry parquets
      (lamo/ holds real LAMO-epoch data, both already ground)

=== 110825 DSK + ELLIPSOID GEOMETRY GRINDS — COMPLETE ===

[x] Job 1: HAMO 110825 DSK (F1B only) — COMPLETE (Jun 20 2026, job 25893871)
    1,089 parquets, 1,141,899,264 px, mean_iof=0.106, f_solar=892
    Sentinel: logs/hamo_110825_geometry_complete.sentinel VERIFIED

[x] HAMO all-letter (F1B-F1G) geometry grind — COMPLETE (Jul 19 2026)
    5,498/5,547 parquets (99.1%) in 04_geometry_tables_dsk256_110825/hamo/
    49 F1D images excluded — unrecoverable CK gap, see SPICE CK COVERAGE
    above ("KNOWN GAP" entry). Full verification performed: sacct exit
    codes for all submission jobs, parquet-count reconciliation against
    raw *F1[A-Z].IMG inventory (5,547 on disk), zero duplicates, zero
    misfiled images from other phases, content spot-checks (row count,
    mean_iof, incidence range) across multiple slices/letters.
    Fixed during this grind: run_geometry.py previously hardcoded its
    output directory from --mode, silently diverging from the real
    committed baseline location — now a required --output-subdir CLI
    arg with no default. Also fixed: _phase_subdir_from_image_path()'s
    silent default-to-"survey" fallback had contaminated the committed
    survey/ baseline with 16 unrelated images earlier in this session
    (from ad-hoc .FIT-adapter test runs whose paths didn't contain
    rc/survey/hamo/lamo); root-caused but not yet fully fixed at the
    function level (deferred in favor of the required --output-subdir
    workaround) — see tests/test_phase_subdir_routing.py's documented
    "KNOWN DESIGN GAP" test for the still-open follow-up.

[x] Survey/LAMO all-letter (F1C-F1I) geometry grind — COMPLETE (Jul 29 2026)
    Mirrors the completed HAMO all-letter grind above: F1B baselines for
    Survey (845 images) and LAMO (4,349 images) are already complete, so
    this grinds the remaining non-F1B filter letters (F1C through F1I)
    into the SAME committed directories
    (04_geometry_tables_dsk256_110825/survey/ and .../lamo/), same pattern
    as HAMO's F1B + F1B-F1G coexisting in 04_geometry_tables_dsk256_110825/hamo/.
    Downstream silver-layer builds must keep filtering F1B only (unaffected).

    Pilot batch COMPLETE (Jul 20 2026, jobs 26116492-26116499):
    8 jobs (4 Survey slices + 4 LAMO slices, 50 images/slice, 400 total).
    All COMPLETED, exit 0:0, 0 ERROR/Traceback, 0 terminal task errors.
    Parquet counts: survey 845→1045 (+200), lamo 4349→4549 (+200), both
    matching expected upper bound exactly. Slice-to-parquet cross-reference:
    400/400 present, 0 missing. Content spot-checks (3 Survey + 2 LAMO,
    row count/mean_iof/incidence+emission range): all physically plausible.
    NOTE: pilot job logs/names say "hamo_pilot" — leftover script/template
    name (scripts/submit_hamo_pilot.sh reused for the pilot slices), not a
    phase-routing bug; verified via each log's own slice= header line.

    Throughput finding: cross-phase concurrency (4 Survey + 4 LAMO jobs
    running simultaneously) measured at 491.1 img/hr (400 images / 48:52
    slowest-job elapsed), vs the HAMO same-phase 8-job baseline of
    1,086.8 img/hr — only 45.2% of same-phase throughput. Likely cause:
    concurrent jobs loading different SPICE metakernel time windows
    (Survey vs LAMO epochs) increases kernel-pool/I/O contention vs HAMO's
    8 same-phase jobs sharing one metakernel window.

    Decision: full remaining batch runs as SEQUENTIAL same-phase batches
    (Survey full batch, then LAMO full batch via SLURM --dependency=afterok
    chain), not mixed concurrent, to avoid the cross-phase throughput
    penalty. See scripts/submit_survey_allletter.sh and
    scripts/submit_lamo_allletter.sh.

    Full batch COMPLETE (Jul 29 2026), submitted via a Prefect grind_phase
    flow (scripts/prefect_grind_phase_flow.py) for dashboard visibility,
    running against a persistent worker (work pool "geometry-grind",
    deployment grind-phase/grind-phase) — not tied to any single SSH
    session. Survey: 8 jobs (26161512-26161519), 108 images, 8:02-32:37
    min each, all COMPLETED exit 0:0. LAMO: 8 jobs (26161523-26161530),
    5,894 images, 4:06-5:39 hr each (within the 12:00:00 budget), all
    COMPLETED exit 0:0, each submitted with
    --dependency=afterok:<all 8 Survey job IDs> so LAMO only started
    once Survey fully succeeded (SLURM-side trigger; Prefect layered on
    top for monitoring only, per the cross-phase throughput penalty
    found in the pilot).

    Final reconciliation, both phases:
    - Survey: raw=1,153, committed=1,153, remaining=0, contamination-clean.
      845 (F1B) + 200 (pilot) + 108 (this batch) = 1,153 exact.
    - LAMO: raw=10,443, committed=10,443, remaining=0, contamination-clean.
      4,349 (F1B) + 200 (pilot) + 5,894 (this batch) = 10,443 exact.
    - Slice-to-parquet cross-reference: 108/108 Survey, 5,894/5,894 LAMO,
      0 missing on either phase.
    - All 16 job logs: 0 ERROR/Traceback, 0 terminal task errors, and
      each job's "Successful files" count matches its slice size exactly.
    - Content spot-checks (5 Survey: 2xF1C/3xF1D — F1F already exhausted
      by the pilot; 5 LAMO: F1C/E/F/H/I): all physically plausible, row
      count/mean_iof/incidence+emission range. One LAMO F1E frame showed
      incidence 93-97° (sun below horizon for the whole image, mean_iof
      ~1.5e-5) — an extreme-geometry raw frame, not a defect; consistent
      with the documented iof>0.01 silver-layer shadow filter not yet
      applied at this raw-table stage.

    Unlike HAMO, zero images excluded — no CK gap or equivalent found on
    either phase; both reconcile exactly to zero missing/duplicate/misfiled.

[x] Job 2: LAMO 110825 DSK — COMPLETE (Jun 21 2026, job 25893955)
    4,349 parquets, 4,560,257,024 px, mean_iof=0.082, mean_inc=61.2°, f_solar=892
    non_f1b=0; n_images=4349 PASS; mean_iof in [0.06,0.20] PASS
    Sentinel: logs/lamo_110825_geometry_complete.sentinel VERIFIED

[x] Job 3: Survey ELLIPSOID — COMPLETE (Jun 20 2026, job 25898149)
    816 parquets, 679,710,522 px, mean_iof_daylight=0.137, f_solar=892
    non_f1b=0; n_images=816 PASS; mean_iof_daylight in [0.08,0.25] PASS
    Sentinel: logs/survey_ellipsoid_geometry_complete.sentinel VERIFIED
    NOTE: 1 stray approach-image parquet (FC21B0001898, DOY 11123) removed
    before validation — created by early smoke test via --image flag.

[ ] MCMC submission — pending geometry completion

[ ] DPS abstract — deadline July 1

[x] ACM payment — DONE June 19

=== VALIDATED PIPELINE STATE ===

Production fit source: 04_geometry_tables_fast/survey/
  F1B only, 845 contributing images, f_solar=892
  Committed parameters: w=0.4656, g=-0.3325, θ̄=8.38°
  CV-RMSE=9.223% = √(5.125²+7.669²) exactly
  Reduced chi-square=1.31 (floor-weighted, floor=CV_other·mean_iof)

DSK kernel: vesta_gaskell_256_110825.bds
  SHA-256: b9c3c81a... (48,022,528 bytes)
  Source: NAIF DAWN SPICE archive, mission-science Gaskell SPC

f_solar: 892.0 W/m²
  Derivation: Albedo Anchor from Li et al. 2013 Table 2
  Physical check: 892/1361 = 65.5% of solar constant ✓
  Primary source: Sierks et al. 2011 (F_solar not directly
  tabulated for F1 broadband; derived via spectral integration)

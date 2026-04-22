# ARCHITECTURE_DECISIONS

This document is the permanent engineering diary for the unified Dawn FC geometry pipeline. It records the final architecture decisions, the failure modes that previously broke production runs, and the controls now in place.

## The I/F Reflectance Physics

Root cause and correction:
- A 100x photometry scaling defect existed in geometry generation because calibrated FC image values were being propagated without conversion to dimensionless I/F ratio.
- The geometry engine now enforces percent-to-ratio conversion at source as raw_value divided by 100.0.

Physical guardrails:
- A hard rejection guardrail now treats values as fatal if min is less than -0.01 or max is greater than 1.05 after conversion.
- A tolerance clip path handles near-bound noise by clipping valid-tolerance values into strict physical range [0.0, 1.0].
- Explicit example behavior is enforced: -0.005 converts to 0.0 and 1.04 converts to 1.0.

Historical-data policy decision:
- The retroactive repair approach using scripts/fix_parquet_iof.py was abandoned.
- Reason: integer-casting side effects in the backfill path corrupted historical Survey and HAMO parquet outputs.
- Permanent policy is now Unified Clean Processing: regenerate all phase outputs through the same production SPICE geometry engine, not post-hoc patch scripts.

## The SPICE Engine & Kernel Gaps

Failure mode:
- Production geometry runs previously failed with SPICE(NOFRAMECONNECT), indicating insufficient frame connectivity coverage for late-mission epochs.

Investigation findings:
- Kernel inventory had partial 2011-centric CK coverage and missing 2012 attitude segments required for late LAMO and RC epochs.
- Dynamic metakernel regeneration and kernel discovery logic were expanded to include 2012 spacecraft and solar-array CK patterns, plus quick-look coverage where available.

Operational remediation:
- Missing 2012 CK files were physically downloaded into the scratch kernel store and included in dawn_dynamic.tm.
- A filename compatibility gap existed for one trajectory reference: dawn_sc_110802-110831_110922_v1.bsp was required by validation logic, while NAIF-hosted equivalent existed as dawn_rec_110802-110831_110922_v1.bsp.
- A symlink compatibility bridge was introduced so the required SC filename resolves to the REC file target without duplicating data.

Resulting state:
- Metakernel entry existence checks now resolve true for required trajectory and CK references.
- Single-image 2012 LAMO validation completed without NOFRAMECONNECT and produced full geometry output.

## The Routing & Scale Bugs

Routing failure history:
- A duplicate file-path bug in worklist construction inflated queued workload and effectively doubled SLURM processing burden for affected image sets.

Fixes applied:
- Worklist generation now builds one canonical image path per manifest row using phase-aware directory routing.
- De-dup protection was added so repeated manifest entries cannot enqueue duplicate paths.
- The earlier phase barricade that restricted runs to LAMO and RC was removed for final unified processing.

Idempotency architecture (wall-limit survivability):
- build_worklist checks phase-specific output parquet existence before queueing.
- skipped_ids is treated as a first-class resume primitive.
- This enables safe re-submission after 24-hour wall-limit termination: completed files are skipped, only missing files are queued.

## The CI/CD Testing Perimeter

New test perimeter added to lock regressions:
- tests/test_geometry_engine.py
- tests/test_routing.py

What is mathematically proven:
- I/F conversion correctness: raw FC values are divided by 100.0.
- Guardrail correctness: values outside tolerance bounds trigger hard failure.
- Clip correctness: within-tolerance outliers are clipped into [0.0, 1.0] exactly.
- Routing correctness: worklist behavior is phase-aware and deduplicated.
- Idempotency correctness: existing output parquet files are skipped; only missing work is queued.

Execution evidence:
- Pytest execution in photomc_env completed successfully for the new suites.
- Routing tests were rerun after unified four-phase changes and passed.

## Current Pipeline State

run_geometry.py is now the unified production processor with these guarantees:
- Processes all four phases: Survey, HAMO, LAMO, and RC.
- Uses phase-specific input and output directories consistently.
- Enforces idempotent queueing against scratch output parquet state.
- Preserves duplicate-path protection in worklist construction.
- Operates on the hardened dynamic SPICE metakernel and expanded kernel inventory.

Final operating model:
- No post-hoc parquet repair scripts.
- No phase-fragmented processing logic.
- One deterministic SPICE geometry path for full historical regeneration and incremental resume.

## Phase 2: DuckDB Aggregation & Phase Audit

Exception-handling semantics (official behavior):
- `NotFoundError` and `SpiceyError` are intentionally handled differently in the ray-trace loop.
- `spiceypy.utils.exceptions.NotFoundError` is treated as a per-pixel miss (background space) and is skipped with `continue`; the image run proceeds.
- `spiceypy.support_types.SpiceyError` is treated as a hard geometry failure for that image, logged as fatal context, and re-raised.
- Worker-level failure classification records `SpiceyError` / `SpiceyErrorText` / `Exception`, and failures are appended to `logs/geometry_failure_log.jsonl` with product ID and full message.

Output integrity and frame-size validation:
- A parquet integrity sweep over `data/04_geometry_tables/*/*.parquet` found:
	- `TOTAL_PARQUET=11104`
	- `VALID_PARQUET=11104`
	- `CORRUPT_OR_SKIPPED=0`
- Minimum row-count check result:
	- `MIN_ROW_COUNT=1048576`
	- `MIN_ROW_FILE=data/04_geometry_tables/hamo/FC21B0006742_11246184031F1B_geometry.parquet`
- Engineering conclusion: all current outputs are full 1024x1024 frame products; no truncated low-row outputs remain in the active directory set.

HAMO gap interpretation for methods:
- The previously tracked 51 missing HAMO images are not classified as SPICE-failure products in the current production interpretation.
- They are documented as unreached workload segments caused by SLURM wall-clock exhaustion (job termination before queue exhaustion), not intrinsic SPICE geometry failure.
- This distinction is now part of the Phase 2 audit narrative and should be used in the paper methods section.

## Phase 3: Baseline Modeling & HAMO Aggregation

DuckDB `NotImplementedException` fix:
- Root cause: CSV-style options were being interpreted at the Parquet input boundary.
- Corrective action: enforce plain `read_parquet(?)` for input and keep `HEADER`/`DELIMITER ','` only in the `COPY (...) TO ? (...)` export clause.
- This formally separates input semantics (Parquet scan) from output semantics (CSV materialization) and removes the invalid option path.

HAMO baseline aggregation run context:
- The Phase 3 baseline aggregation is now executed against completed HAMO geometry tables only.
- Aggregation scale documented for reproducibility: **5,496 HAMO Parquet files**.
- Compute environment note: aggregation run executed on compute node **c093** with **64 GB RAM**.
- Output artifact target remains `data/05_aggregated/hamo_phase_curve.csv` as the lightweight optimization-ready phase-curve baseline.

Mission Aggregator architecture shift:
- Aggregation is no longer modeled as a single-phase utility; it has been generalized into a mission-level extractor script: `scripts/aggregate_mission_data.py`.
- The mission aggregator iterates phases `survey`, `rc`, `hamo`, and `lamo`, validates directory/file presence per phase, and materializes one CSV per phase in `data/05_aggregated/`.
- This establishes a reproducible "Data Lake extraction" pattern for Phase 3 baseline products while keeping phase outputs isolated and comparable.

Execution-time observability:
- Per-phase timer instrumentation (`start_time = time.time()`) was added to the mission aggregator.
- The script now logs elapsed seconds for each phase extraction (for example: "Processed HAMO in 642.5 seconds").
- This timing telemetry is part of the Phase 3 performance envelope and is used to monitor DuckDB throughput and detect regressions in extraction runtime.

Live run status:
- Geometry Engine job `vesta_ra` is currently running on node `c148`.
- RC geometry processing (423 images) and Survey geometry processing (1,153 images) are in progress.
- The job has exceeded 6 hours of wall-clock time, which is consistent with heavy geometric computation such as high-resolution shape-model intersections.
- Once the SLURM job completes, the mission aggregator will run to finalize the four-phase baseline dataset.

Timeout incident and resume strategy:
- Incident: Survey phase processing timed out under SLURM in Job `25514161`.
- Resolution: a resume strategy is implemented via file-existence checks before geometry execution.
- Engineering requirement: prevent redundant computation of 16,000+ existing HAMO/LAMO geometry outputs by verifying target parquet paths before initiating ray-tracing.
- Current pending workload: 850 Survey images and 423 RC images.

Subset execution architecture decision:
- Decision: shifted from a monolithic geometry submission model to surgical subset scripts for final mission phases.
- Rationale: avoid repeated I/O overhead from scanning 16,000+ already-produced outputs on Curta network storage.
- Standardization: the manifest-driven execution pattern is now the blueprint for the General Planetary Pipeline to support reliable multi-stage mission processing.

Finish-script verification and integrity status:
- Verification: the finish script operates with file-level manifest logic, not folder-level gating.
- Integrity: 303 existing Survey parquet outputs are preserved.
- Redundant computation is avoided by cross-referencing `01_calibrated_images` inputs against `04_geometry_tables` outputs at the individual image stem level before ray-tracing is queued.

SPICE ray-tracer masking decision:
- Event: replaced strict image-level finiteness validation with pixel-level boolean masking in the geometry ray-tracing path.
- Rationale: distant mission phases (Survey and RC) can contain >90% deep-space pixels, which naturally produce non-finite SPICE intercept vectors; strict whole-image finiteness checks caused false fatal failures.
- The masking strategy now preserves valid asteroid-surface pixels and drops empty-space pixels on a per-pixel basis.
- Impact: pipeline robustness is maintained across all orbital distances, independent of target fractional coverage in the camera field of view.

## Phase 4: SciML Architecture Refactor

Event:
- Deployed the 4-layer SciML architecture alongside the legacy `hapke_mcmc_package` data pipeline.

Design pattern:
- Physics, inference, and data are now explicitly separated into `photometry.models`, `photometry.fitting`, and the existing ETL layer.
- This removes single-responsibility violations from the model layer and keeps MCMC concerns out of photometric physics classes.

Core contracts established:
- Dual-backend evaluation is now a first-class contract: NumPy for bulk CPU computation and PyTorch for gradient-based optimization.
- Minimum vectorization throughput is now specified as `1_000_000` rows for the model contract and benchmarked with pytest.

Next step:
- Implement the baseline `LambertianModel` first so the new testing and backend-dispatch framework can be validated end-to-end before adding the remaining photometric models.
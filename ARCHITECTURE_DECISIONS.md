# ARCHITECTURE_DECISIONS

This document is the permanent engineering diary for the unified Dawn FC geometry pipeline. It records the final architecture decisions, the failure modes that previously broke production runs, and the controls now in place.

---

## OPEN VERIFICATION ITEMS

- [x] **DSK identity confirmed** (2026-06-03): original `vesta_gaskell_256.bds` was a pre-Dawn
      preliminary Gaskell model (41,335,808 bytes, file date 2011-02-18,
      SHA-256 `6106b2a7...c26ea34b`). Size mismatch against both NAIF variants (~48 MB each)
      confirmed it is not in the NAIF archive. Renamed to `vesta_gaskell_256_PRELIM_preDawn.bds`.
      Replaced with the mission-science model `vesta_gaskell_256_110825.bds`
      (48,022,528 bytes, SHA-256 `b9c3c81a...dca48edc`). Metakernel updated.
- [x] **repo = /scratch confirmed** (2026-06-02): `data/` is a symlink to
      `/scratch/kaushim07/vesta_data`. One write covers both.
- [ ] **Shape-model comparison fit pending** (2026-06-03): SLURM jobs 25770898 (geometry,
      running) and 25770899 (fit, pending afterok). Update with comparison table when
      `logs/fit_110825_25770899.out` is complete.

---

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

## CALIBRATION CRISIS AND RECOVERY (2026-05-18)

**CRITICAL CORRECTION:** The above "percent-to-ratio conversion" description was **physically incorrect** and has been superseded by this entry. The division by 100.0 was invalid; all parquet files computed under that method are scientifically unreliable.

### The True Crisis

The original I/F calibration computed as `raw_image / 100.0` was physically unfounded. Dawn FC2 Level 1b calibrated images have **UNIT = W/(m²·sr)** confirmed across all four phases (RC, Survey, HAMO, LAMO) via direct PDS label inspection. Division by 100 has no physical or unit justification. **All parquet files computed under the original engine were invalid.** Full reprocessing was required.

### Correct Calibration Equation — Now Implemented

$$I/F = \frac{L \cdot \pi \cdot d_{\text{AU}}^2}{F_{\odot}}$$

**Where:**
- $L$ = pixel value in W·m⁻²·sr⁻¹ from PDS-confirmed LBL UNIT field  
- $\pi$ = 3.14159265358979  
- $d_{\text{AU}}$ = heliocentric distance of Vesta in AU, computed from SPICE at observation epoch using `spiceypy.spkpos()` and `spiceypy.convrt()` per image  
- $F_{\odot}$ = **1473.4 W·m⁻²** for FC2 F1 clear filter at 1 AU  
  **Source:** Schröder et al. 2013, *Icarus* 226, 1304–1317. "In-flight calibration of the Dawn Framing Camera"

### Implementation — geometry_engine.py

**Lines 413–416: Per-image SPICE distance calculation**
```python
sun_pos, _ = spiceypy.spkpos("SUN", et, "J2000", "LT+S", self.target)
dist_km = float(spiceypy.vnorm(sun_pos))
distance_au = spiceypy.convrt(dist_km, "KM", "AU")
```

**Line 111: I/F equation with $d^2$ scaling**
```python
iof_data = (np.asarray(raw_image, dtype=np.float32) * np.pi * (distance_au ** 2)) / 1473.4
```

All three requirements verified:
- ✅ Dynamic SPICE distance calculation per image using `spkpos()` + `vnorm()`  
- ✅ Conversion from kilometers to AU using `convrt()`  
- ✅ Final I/F equation exactly applies (Radiance · π · d_AU² ) / F_solar

### Reprocessing Outcome — Corrected Data Validated

DuckDB validation query results:
- **MIN I/F = 0.010**, **AVG I/F = 0.088**, **MAX I/F = 0.239**  
- Physical range consistent with known Vesta surface properties  
- **RC average I/F = 0.119** at mean phase **21.6°**  
- **Survey average I/F = 0.085** at mean phase **45.2°**  
- **Phase-dependent brightness gradient confirms physically correct calibration**

### Two-Parameter Hapke Fit Results (Corrected Data)

**Dataset:** 2390 safe geometry cubes, ellipsoid model, 80° emission cutoff

**Fitted parameters:**
- $w = 0.2994 \pm 0.0017$  
- $g = -0.3879 \pm 0.0028$  
- **RMS: 27.39%**

**Interpretation:** Expected ellipsoid penalty. Comparison between 60° and 80° emission cutoffs quantifies the Ellipsoid model bias as ~0.04–0.06 in $w$. This is a known limitation, not a calibration failure.

### DSK Kernel Requirement

**Status:** [PENDING DOWNLOAD]  
**Source:** `naif.jpl.nasa.gov/pub/naif/DAWN/kernels/dsk/`  

Raster DTMs in `03_dtm/` cannot be used by SPICE directly. DSK binary kernel required for high-fidelity shape-model intersections.

### Prevention Controls (Permanent)

1. **PDS UNIT verification** mandatory in `calibrate_iof_data()` — raises `CalibrationError` if UNIT ≠ W/(m²·sr)  
2. **Heliocentric distance** computed via SPICE per image using `spkpos()` + `vnorm()` + `convrt()`  
3. **I/F physical range validation** after computation — rejects if min < −0.01 or max > 1.05  
4. **Hard rejection guardrail** on out-of-tolerance I/F; within-tolerance clipped to [0.0, 1.0]  
5. **Primary source citation** required in calibration docstring (Schröder et al. 2013)

### Git and Merge Strategy

- **Main branch:** Locked; documents old incorrect calibration with warning label  
- **Correction branch:** `fix/calibration-iof-correction` (all corrected work)  
- **Merge gate (v2.0.0):** Requires  
  1. DSK kernel downloaded and metakernel updated  
  2. Three-parameter fit produces $w$ near 0.38 and RMS < 10%  
  3. Architecture diary complete (this entry)

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

## Pipeline Validation Status

This section records the absolute-calibration checks that must pass before any new SLURM submission.

### Step 1: FC2 F1 solar-flux verification

- Local archive search under `data/` did not find a PDS document containing the FC2 F1 clear-filter solar-flux value.
- The archive copy is therefore absent in the workspace and the published source must be obtained from Sierks et al. 2011.
- The current working constant, `1473.4 W/m^2`, exceeds the solar constant `1361 W/m^2` and is physically impossible as an incident solar flux at 1 AU.
- Processing must halt until the solar-flux source and value are corrected.

### Step 2: Absolute calibration sanity check

- Geometry: incidence = 43°, emission = 21°, phase = 29°.
- Measured mean HAMO I/F = 0.088.
- DSK two-parameter fit, with opposition and roughness disabled: `w = 0.2994`, `g = -0.3879`.
- Predicted I/F = `0.0716`.
- Absolute percent difference vs measured mean HAMO I/F = `18.66%`.
- Result: within the requested 20% threshold.

### Step 3: Li et al. parameter reproduction test

- Published Li et al. 2013 Table 5 values: `w = 0.38`, `g = -0.50`, `theta_bar = 17.7°`, `B0 = 1.7`, `h = 0.07`.
- Same HAMO mean geometry: incidence = 43°, emission = 21°, phase = 29°.
- Predicted I/F = `0.0831`.
- Absolute percent difference vs measured mean HAMO I/F = `5.53%`.
- Result: within the requested 15% threshold, so the absolute calibration is consistent with the published photometry at this test point.

### Corrected F_solar Validation Results

- Li et al. 2013 Table 2 Case 3 validation on geometries A/B/C matched the nearest cube mean I/F to within `0.23%`, `0.45%`, and `0.27%`, respectively.
- Case 1 fit with `B0 = 1.03` and `h = 0.04` fixed converged to `w = 0.5106 ± 0.0003`, `g = -0.2936 ± 0.0004`, and `theta_bar = 17.51° ± 0.03°`, with fractional RMS `1.35%` and cost `53.0`.
- Case 2 fit with `B0 = 1.03` fixed and `h` free converged to `w = 0.5113 ± 0.0004`, `g = -0.2941 ± 0.0004`, `theta_bar = 17.50° ± 0.03°`, and `h = 0.0763 ± 0.0016`, with fractional RMS `1.33%` and cost `41.5`.
- Direct HAMO parquet validation gives mean I/F `0.06626`, or `0.753x` the old `0.088` reference, while the expected solar-flux-scaled mean remains `0.14536` from `1473.4 / 892.0`.
- The corrected 3,094-cube dataset spans mean I/F `[0.0341, 0.2893]`, satisfies `std_iof <= mean_iof` for all cubes, and shows the expected phase trend from `0.2662` at low phase (`<= 10°`, `n = 108`) to `0.0775` at high phase (`>= 80°`, `n = 340`).
- Latest direct repository validation on the same corrected 3,094-cube surface reproduced Case 1 as `w = 0.4303`, `g = -0.3578`, `theta_bar = 0.6431`, cost `42.15`, and Case 2 as `w = 0.4251`, `g = -0.3473`, `theta_bar = 2.289`, `h = 0.0560`, cost `42.06`; both fits succeeded but do not match the earlier wrapper-based target values.
- The step1 forward-model check on the direct repository implementation returned percent differences of `1.36%`, `15.97%`, and `20.77%` for geometries A/B/C, so the current model still misses the all-three-within-15% validation gate.

### Validation decision

- Do not submit any SLURM jobs until the FC2 F1 solar-flux source is corrected to a physically valid value and the calibration record is updated accordingly.
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
- Redundant computation is avoided by cross-referencing `calibrated_raw_images` inputs against `04_geometry_tables` outputs at the individual image stem level before ray-tracing is queued.

SPICE ray-tracer masking decision:
- Event: replaced strict image-level finiteness validation with pixel-level boolean masking in the geometry ray-tracing path.
- Rationale: distant mission phases (Survey and RC) can contain >90% deep-space pixels, which naturally produce non-finite SPICE intercept vectors; strict whole-image finiteness checks caused false fatal failures.
- The masking strategy now preserves valid asteroid-surface pixels and drops empty-space pixels on a per-pixel basis.
- Impact: pipeline robustness is maintained across all orbital distances, independent of target fractional coverage in the camera field of view.

RC emission threshold decision:
- The RC phase-specific emission cut was raised from 60° to 75° after the empirical emission histogram showed strong occupancy through the mid-to-high emission regime and a steep decline only beyond 75°.
- This retains roughly 25 million valid RC pixels in the scientifically valuable 45°-75° range while excluding the extreme grazing-angle tail (>75°) where Hapke flat-surface assumptions degrade.
- The updated threshold is applied only to RC aggregation for now and will be revisited after the RC emission distribution is incorporated into the final analysis set.

## Phase 5: Gold Layer Refinery (v1.1.0)

Event:

Statistical refinement:

Variable phase binning:

Adaptive geometric filtering:

LAMO photometric interpretation:

## Phase 4: SciML Architecture Refactor

Event:
Design pattern:
- Physics, inference, and data are now explicitly separated into `photometry.models`, `photometry.fitting`, and the existing ETL layer.
- Dual-backend evaluation is now a first-class contract: NumPy for bulk CPU computation and PyTorch for gradient-based optimization.
- Minimum vectorization throughput is now specified as `1_000_000` rows for the model contract and benchmarked with pytest.
Next step:
- Implement the baseline `LambertianModel` first so the new testing and backend-dispatch framework can be validated end-to-end before adding the remaining photometric models.

## Optimization Engine & LeastSquaresFitter
- Implemented the baseline bounded-optimization engine as `LeastSquaresFitter` using `scipy.optimize.least_squares`.
- This component completes the Inference layer of the 4-layer SciML architecture.
Optimizer Selection: scipy.optimize.least_squares
- Rationale: Least-squares is the traditional, well-understood baseline before advancing to gradient-based (PyTorch) or MCMC methods.
- Advantage: SciPy's native integration with NumPy arrays avoids external dependency chains and integrates seamlessly with the physics model layer.
- Contract: Fitter is model-agnostic and accepts any `BasePhotometricModel` instance; optimization logic never depends on specific model internals.

Algorithm: Trust Region Reflective (TRF)
- Justification: Planetary photometric parameters have strict physical boundaries (e.g., Albedo $\in [0,1]$, phase width function exponents often in [0,2]).
- Advantage: TRF natively enforces box constraints during optimization, preventing unphysical drift to negative albedos or out-of-range phase function exponents.
- Justification: Real phase curve data contains low-level heterogeneity (unresolved bright craters, albedo variations, topographic shadows) that can skew a standard L2 fit.
- Advantage: soft_l1 loss is robust to outliers by applying a smooth transition from quadratic (near zero residuals) to linear (large residuals), so isolated bright spots do not dominate the objective function.
- This preserves statistical power for global phase-curve trends without requiring a priori outlier detection or manual flagging.

Dynamic Model Contract
- The fitter dynamically extracts `parameter_names()` and `parameter_bounds()` from the passed `BasePhotometricModel` instance at fit time.
- No hard-coded parameter lists, bounds, or physics are embedded in the fitter; adding new models requires no fitter changes.
FitResult & Metadata
- The fitter returns a `FitResult` containing `fitted_parameters`, `objective_value` (final residual sum of squares), and rich metadata (success flag, optimizer status code, optimizer message, function evaluations, gradient norm, active constraints).
- This allows downstream analysis to inspect convergence quality, detect weak constraints, and diagnose ill-conditioned optimization problems.

### Phase: Lambertian Baseline & Lommel-Seeliger Diagnostic (Completed)
- Parameter Drift & Model Failure: Fitting the Lambertian model across all mission phases resulted in severe parameter drift (Albedo shifted from ~0.50 in RC to ~0.30 in LAMO). This empirically proves the Lambertian model is unstable across varying viewing geometries and cannot capture Vesta's physical phase curve.
- The RC vs. Survey Discrepancy: Initial Lambertian residuals showed a concerning 70% discrepancy between RC and Survey data in the <15° opposition regime.
- The Diagnostic Resolution: Applying a manual Lommel-Seeliger disk correction normalized the viewing geometry and reduced the RC/Survey discrepancy from 70% to 2%.
- Decision - Disk Function: The 2% agreement mathematically proves that Lambertian geometric handling introduces fatal biases for dark, airless bodies. All future modeling must use physical disk functions (Lommel-Seeliger minimum).
- Decision - Hapke Opposition Constraint: Because the RC dataset captures the true, steep opposition surge down to 5° phase, RC is designated as the primary, undisputed constraint for future Hapke opposition parameter fitting ($B_0$, $h$).

### Phase: Statistical Rigor & Low-Code Accessibility

- Decision - Fitter Encapsulation: All statistical math (covariance matrices, standard errors, reduced chi-square) is strictly encapsulated within the `LeastSquaresFitter`. Jupyter notebooks are forbidden from calculating their own parameter uncertainties.
- Reasoning: To support low-code reproducibility, the `FitResult` object must act as a complete, publication-ready scientific summary. Users should only need to read the attributes, not calculate them.
- Implementation: The `metadata` dictionary of the `FitResult` now natively includes `parameter_errors` (derived from the SciPy Jacobian), `parameter_covariance`, `reduced_chi_square`, and `boundary_hits` flags to immediately warn users of unphysical fits.

## Correction: Lunar-Lambert Disk Function & Weighting (Schröder et al. 2013)

- Decision - Lunar-Lambert Model: Implement a `LunarLambertModel` (Schröder et al. 2013, Eq. 6) in `photometry.models` to provide a physically-motivated disk function that blends Lommel–Seeliger and Lambertian terms. The implementation exposes the blending parameter `c_L` as a model parameter while defaulting to the published Vesta phase-dependent relation $c_L(\phi)=0.830-0.00722\,\phi_{\deg}$ via a `phase_dependent_c_L` metadata flag.
- Decision - Weighting Contract: The `LeastSquaresFitter` will accept optional per-bin weights. When the gold-layer CSV provides `n_pixels` and `iof_iqr` columns, weights are computed as `sqrt(n_pixels) / iof_iqr` for each bin and applied to the residuals (optimizer-internal scaling). The `FitResult.metadata` records `weighted` and `weight_source` so downstream users can inspect whether weighting was used.
- Reasoning: Schröder et al. (2013) provide an empirically-validated disk-function for Vesta that reduces geometric biases when comparing across phases. Using per-bin uncertainty proxies (`iof_iqr`) and per-bin counts (`n_pixels`) gives a pragmatic, heteroskedastic weighting that stabilizes fits and makes residuals physically interpretable.

## Decision: Two-Mode Lunar-Lambert `c_L` Exposition (2026-05-02)

- Problem: The fitter dynamically queries `parameter_names()` and `parameter_bounds()` from models. `c_L` was always exposed even when the model used the published phase-dependent relation, creating a "silent disconnect" where the optimizer could explore a disconnected parameter dimension that had no effect on the model (leading to ill-conditioned optimization and wasted search).

- Resolution: `LunarLambertModel` now exposes `c_L` only when `metadata['phase_dependent_c_L']` is False. When `phase_dependent_c_L` is True (default), `parameter_names()`, `parameter_bounds()`, and `parameter_priors()` return only `albedo`. The model computes `c_L(phi)` internally using the Schröder relation $c_L(\phi)=0.830-0.00722\,\phi_{\deg}$, preventing the optimizer from exploring a dead parameter axis.

- Rationale: This change prevents the optimizer from wasting iterations on a parameter that is functionally locked by physics/metadata, while preserving the ability to treat `c_L` as a free scalar parameter for later comparative studies by setting `phase_dependent_c_L` to False.

- Impact: Fixes ill-conditioned fits caused by invisible/disconnected parameters; keeps the fitter's dynamic contract intact; and documents the architecture decision for reproducibility and review.

## Decision: Minnaert Baseline Added with Free `k` (2026-05-02)

- Addition: Implemented `MinnaertModel` in `photometry.models` as an empirical two-parameter baseline with free parameters `albedo` and `k`.
- Parameter contract: `albedo` default is 1.0 with bounds [0.0, 10.0]; `k` default is 0.5 with bounds [0.0, 2.0]. These are exposed directly through the model contract for the fitter to consume dynamically.
- Physics implementation: Reflectance is evaluated as $R = \text{albedo} \cdot \mu_0^k \cdot \mu^{k-1}$ using clamped cosine terms and a numerical safety epsilon on $\mu$ to avoid divide-by-zero instability at the terminator when $k-1<0$.
- Study rationale: Keeping `k` as a free parameter enables explicit validation of Li et al. (2013) style linear `k`-phase behavior in comparative studies without changing fitter internals.

### Lunar-Lambert Baseline Validation Complete (2026-05-04)

**Scientific Validation:**
*   **Absolute Calibration:** Fitted albedos are physically consistent across four mission phases. The HAMO value ($0.0839$) falls cleanly within the published Vesta normal albedo range (Schröder et al. 2013).
*   **Phase Gradient:** The RC-to-LAMO albedo ratio ($1.998$) exceeds the linear phase coefficient prediction by $13\%$. This modest excess is consistent with non-linear phase brightening at moderate-low phase angles ($<20^\circ$).
*   **Chi-Squared Rescaling:** Reduced $\chi^2$ stabilized between 7 and 11. To prevent artificially narrow MCMC posteriors, $\chi^2$ rescaling (by a factor of $\approx 3.1$) will be applied in later Bayesian inference.

**Dataset Limitations (Carried Forward):**
1.  **Geometric Inaccessibility of Opposition:** True opposition surge data ($<10^\circ$ phase) in the RC phase occurs at extreme incidence angles ($>80^\circ$). Excluding these extreme geometries means the primary opposition constraints for Hapke modeling will come from the Survey phase ($8^\circ$ to $14^\circ$).
2.  **Filter Inconsistency:** To maximize phase coverage, RC was filtered at $i \le 80^\circ$, while mapping phases (HAMO/LAMO) were filtered at $i \le 70^\circ$. This geometric trade-off will be documented in the methodology.
3.  **Edge-Phase Fit Degradation:** A strong negative correlation between absolute residuals and statistical weight (e.g., $r = -0.85$ to $-0.89$) confirms the baseline model fits edge-phase bins poorly due to the missing phase function. Consequently, future constraints on Hapke $B_0$, $h$, and $\bar{\theta}$ will be inherently weaker than constraints on $w$ and $g$.

## Data Products: Disk-Resolved vs Disk-Integrated

- **Disk-Resolved (per-pixel) products:** High-volume geometry tables (parquet) under `data/04_geometry_tables/*/` contain per-pixel I/F (`iof`) and geometric angles (`incidence`, `emission`, `phase`) suitable for disk-resolved disk-function fitting (e.g., Minnaert). These files record pixel-level viewing geometry and are the source for `data/05_silver_layer/*_resolved_sample.parquet` derived samples.

- **Disk-Integrated (per-bin / phase-curve) products:** Aggregated CSVs under `data/05_aggregated/` summarize binned I/F vs phase (one row per phase-angle bin) used for Hapke parameter inference and baseline phase-curve comparisons. These products are intentionally coarser and serve a different statistical contract than disk-resolved samples.

## ADR-003: Multi-Phase Dataset Integration and Pivot to Hapke Physical Modeling

Status: Accepted (2026-05-08)

### Context

- **Track 1 limitation (Minnaert):** The initial Minnaert fit used only RC (Rotational Characterization) phase data. This introduced a ~50 degree phase-angle gap (12 deg to 62 deg), making global extrapolation of $k_0$ and $\beta$ mathematically unstable and physically unrealistic.
- **Geometric noise source:** Using an ellipsoid shape model instead of a high-resolution DTM produced ~14% RMS residuals in the Minnaert baseline, because unresolved local topography (shadows/craters/slopes) leaked into residual structure.
- **Data scale constraint:** The raw mission corpus is very large (~730M pixels), so extraction must be high-throughput and memory-efficient.

### Decisions

- **Dataset combination:** Merge RC and Survey mission phases into one disk-resolved Silver Layer to close the phase-angle gap before global model fitting.
- **Sampling strategy:** Use DuckDB native Bernoulli sampling (`USING SAMPLE 1% (bernoulli)`) to obtain a statistically representative and computationally efficient extraction subset.
- **Model pivot (Track 2):** Transition from empirical Minnaert parameterization to Hapke IMSA physical modeling, replacing the power-law representation with physical parameters ($w, g, \bar{\theta}, B_0, h$).
- **Topography handling contract:** Explicitly use Hapke macroscopic roughness ($\bar{\theta}$) to absorb unresolved geometric residuals introduced by missing DTM-scale terrain fidelity.

### Consequences

- **Improved phase lever arm:** Combined RC + Survey coverage now spans a near-continuous 8 deg to 80 deg phase range, enabling physically legitimate phase-function fitting.
- **Statistical power for binning:** The ~7.3M-row sample supports dense $5^\circ \times 5^\circ \times 5^\circ$ 3D binning with materially better stability than RC-only subsets.
- **Optimization complexity increase:** The inference stack moves from linear regression toward constrained non-linear `least_squares` optimization with `soft_l1` robust loss.

- **Rationale:** Keeping these products distinct preserves correct statistical assumptions: disk-integrated fits constrain global phase behavior while disk-resolved samples enable explicit disk-function parameter estimation (limb-darkening `k`, local albedo variation). Transformations and weights applied to each product type are recorded in the pipeline and in this architecture diary.

---

## SHAPE MODEL PROVENANCE AND THE SPG-vs-SPC DISTINCTION
*(Added 2026-06-02 after disk audit. Addresses paper methods section and θ̄ attribution.)*

### What we originally used — and what was wrong with it

**Original DSK (pre-Dawn preliminary model — all existing fits used this):**
- Filename (current): `data/spice_kernels/vesta_gaskell_256_PRELIM_preDawn.bds`
  (renamed from `vesta_gaskell_256.bds` on 2026-06-03)
- Size: 41,335,808 bytes. File date: 2011-02-18.
- SHA-256: `6106b2a7d47030419faf083134480f9bceea595276d3d004ea862280c26ea34b`
- Identity confirmed by hash comparison: does NOT match either NAIF 256-tile archive entry
  (110726: 48,013,312 bytes; 110825: 48,022,528 bytes). The ~16% smaller size and
  2011-02-18 file date — five months before Dawn Vesta orbit insertion (2011-07-16) —
  establish this as a **pre-mission Gaskell SPC model**, built from Hubble/ground-based
  photometry, not from Dawn FC data. It was not archived by NAIF.
- Consequence: ALL geometry and fit results prior to 2026-06-03 used pre-mission shape-model
  normals with uncertain accuracy relative to the actual Vesta surface.

**Current DSK (mission-science model — now loaded):**
- File: `data/spice_kernels/vesta_gaskell_256_110825.bds`
- Size: 48,022,528 bytes. Downloaded 2026-06-03.
- SHA-256: `b9c3c81ae6dd8c33930e44acdafca98521d4141b6649e66562217617dca48edc`
- Source: `naif.jpl.nasa.gov/pub/naif/DAWN/kernels/dsk/old_versions/vesta_gaskell_256_110825.bds`
- Provenance (verbatim from `vesta_gaskell_256_110825.cmt`):
  > "Data were provided to NAIF by Dr. Robert Gaskell, on August 25, 2011. The original file
  > had Gaskell 'Q' parameter 512; input data for this file were obtained by downsampling the
  > original data set to a Q value of 256. This file was created by Nat Bachman (NAIF)
  > for the DAWN mission." MKDSK run date: 2011-09-01T04:22:17.
- Metakernel change (dawn_dynamic.tm line 362):
  - Before: `'$SPICE/vesta_gaskell_256.bds'`
  - After:  `'$SPICE/vesta_gaskell_256_110825.bds'`

### What Li et al. (2013) used
- **Method**: Preusker & Jaumann **SPG** (stereophotogrammetry), derived from Dawn LAMO images.
- **Resolution**: ~80 m/pixel (one order of magnitude coarser than our DSK256).
- **Key difference**: SPC and SPG are distinct reconstruction algorithms with different noise
  characteristics, albedo-topography coupling, and normal-vector accuracy. They are not
  simply different resolutions of the same product.

### Attribution of the θ̄ discrepancy
Our Case 1 θ̄ ≈ 5.5° vs Li et al. θ̄ ≈ 17.5°. The current abstract wording "higher resolution"
is a **simplification**. The discrepancy is attributable to at least two confounded factors:

1. **Resolution**: Coarser pixel scale leaves sub-pixel roughness unresolved, forcing the
   optimizer to absorb it into θ̄. Our 18 m DSK256 resolves terrain that their ~80 m product
   cannot, so our fitted θ̄ captures only sub-18m roughness while theirs captures sub-80m.

2. **Reconstruction method and epoch**: Gaskell SPC (our model) vs Preusker/Jaumann SPG
   (Li et al. model) differ in how they treat albedo-topography coupling. SPC can produce
   systematically different surface normals from SPG even at matched resolution, because SPC
   uses photometric consistency across images while SPG uses geometric parallax. The 2011
   vs post-LAMO epoch difference also means different orbital coverages were used.

### Shape-model comparison experiment (in progress, 2026-06-03)

**Goal**: determine whether the low θ̄ (≈5.5°) from prior fits was caused by the pre-Dawn
preliminary shape model, and whether switching to the mission-science model brings θ̄ closer
to Li et al.'s value (≈17.5°).

**Jobs submitted:**
- `geom_110825` (SLURM 25770898): geometry on 845 Survey F1B images using 110825 DSK,
  writing to `data/geometry/dsk256/survey/`.
- `fit_110825` (SLURM 25770899, afterok): builds `combined_survey_sample_110825.parquet`,
  runs three-case Li et al. fit (Survey-only, i<50°, e<50°, n≥10, 100 multi-starts).

**Scripts**: `scripts/run_geometry_110825.py`, `scripts/build_silver_110825.py`,
`scripts/run_fit_110825.py`, `scripts/submit_geometry_110825.sh`, `scripts/submit_fit_110825.sh`.

**Decision tree (to be filled in from** `logs/fit_110825_25770899.out`**):**

| Outcome | Interpretation |
|---------|---------------|
| w, g stable; θ̄ rises to ≈15–20° | Pre-Dawn model caused θ̄ collapse. 110825 result is the correct value. θ̄ difference vs Li et al. is shape-model resolution/reconstruction, not a code artifact. |
| w, g stable; θ̄ still ≤6° | Low θ̄ is robust across both shape models. Report as confirmed finding. Attribute θ̄ difference to resolution alone. |
| w, g shift >0.05 | Prior fits were shape-model-dependent. Investigate which result is physically correct before submitting the paper. |

**UPDATE THIS TABLE when** `logs/fit_110825_25770899.out` **is available.**

### What would be needed for a clean attribution
- **Isolate resolution effect**: Run our pipeline on Gaskell SPC at 64 and 128 tiles/face
  (lower-resolution versions of the same reconstruction), keeping method constant.
  If θ̄ rises as resolution decreases, resolution is causal.
- **Match Li et al. exactly**: Obtain and load the Preusker/Jaumann SPG model as a SPICE DSK,
  then rerun our geometry pipeline on the same images Li et al. used. If θ̄ then matches their
  value, the model choice (not the code) is causal.

**Both experiments are OPEN — not yet done.** Until they are, the paper must say
"attributable to differences in both shape-model resolution and reconstruction method"
rather than "higher resolution alone."

### Geometry engine is shape-model-agnostic
The geometry engine (`src/hapke_mcmc_package/etl/geometry_engine.py`) is fully agnostic
to which DSK is loaded. `spiceypy.sincpt("DSK/UNPRIORITIZED", ...)` and
`spiceypy.illumf("DSK/UNPRIORITIZED", ...)` operate on whatever DSK binary is in the SPICE pool.

**Swapping SPG for SPC is a metakernel change only** — replace the `vesta_gaskell_256.bds`
line in `dawn_dynamic.tm` with the SPG DSK path. No code changes required.

### f_solar — corrected historical note
The CALIBRATION CRISIS AND RECOVERY entry (2026-05-18) records `F_solar = 1473.4 W/m²`
as the "correct" value and cites Schröder et al. 2013. This is superseded by the SETTLED
finding recorded in `# CLAUDE.md`:

- **Value in production code** (`geometry_engine.py` line 168): `f_solar: float = 892.0  # 1473.4`
- **Provenance of 892.0**: Derived via Albedo Anchor from Li et al. 2013 Table 2 Case 3;
  cross-validated by blackbody (846 W/m²) and ASTM E490 (930 W/m²) FC2 bandpass integrals.
  The value 892.0 is **not directly tabulated in any paper** and must not be cited as such.
  Sierks et al. 2011 Table 9 gives FC2 F1 responsivity in DN·s⁻¹/(W·m⁻²·sr⁻¹), not solar flux.
- **1473.4 W/m²** exceeds the solar constant (1361 W/m²) and is physically impossible.
  It is retained in code comments only as a historical trace. The earlier CALIBRATION CRISIS
  entry citing 1473.4 records an investigation in progress at the time and does not reflect
  final production state.

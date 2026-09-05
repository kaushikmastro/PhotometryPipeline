# Vesta Photometry Pipeline

SPICE-based geometry ingestion and Hapke/empirical photometric model fitting for Dawn
Framing Camera (FC2) imagery of asteroid (4) Vesta. It turns raw PDS3 images into
per-pixel viewing/illumination geometry, then fits disk-resolved and disk-integrated
reflectance models to the result.

## The science

The pipeline works on Dawn FC2 images across Vesta's Survey, HAMO, LAMO, RC (Rotation
Characterization), and approach mission phases, computing per-pixel incidence, emission,
and phase angles via SPICE (`spiceypy`) against a Gaskell shape model (DSK or ellipsoid).
On top of that geometry it fits:

- **Hapke** (Hapke 2002 approximation by default; the Li et al. 2013 IMSA H-function is
  available via `isotropic_h=True`), with optional macroscopic roughness and the
  shadow-hiding opposition effect (SHOE).
- **Empirical disk functions**: Lambertian, Lommel-Seeliger, Minnaert, and Lunar-Lambert
  are implemented; an Akimov model is planned but not yet built.
- **Disk-integrated photometry**: area-weighted aggregation of resolved-pixel
  reflectance into whole-disk brightness (`photometry.aggregation.disk_integrate`), and
  a sphere-forward integrator that predicts a fitted model's disk-integrated phase curve
  without any real image data (`photometry.aggregation.sphere_forward`), following the
  Li et al. (2013) approach.

### Headline validated result

Case 1 (preliminary Gaskell DSK256 shape model, Survey F1B disk-resolved data,
illuminated regime: `iof>0.01`, incidence<50°, emission<50°, 950 phase-angle bins):

| parameter | value |
|---|---|
| w (single-scattering albedo) | 0.46993 |
| g (asymmetry parameter) | −0.33688 |
| θ̄ (macroscopic roughness) | 8.2662° |
| fRMS | 10.902% |
| reduced χ² | 0.200 |

These parameters are broadly consistent with the Hapke solution for Vesta reported in
Li et al. (2013, *Icarus*), fit independently against Dawn's own Survey-phase disk-resolved
imagery rather than reused from that paper. See `CLAUDE.md` for the full derivation,
the shadow-filter validation behind the `iof>0.01` cut, and the superseded 1%-sample
result this fit replaced.

## Architecture

```
photometry_etl/          SPICE geometry ingestion: PDS3 image reading, per-pixel
  etl/                    incidence/emission/phase/range via sincpt, radiometric
                          calibration (raw DN -> I/F), manifest-driven download from
                          the PDS archive.
        |
        v
  silver / golden layers  Filtered, binned, or aggregated products built from the raw
  (scripts/golden/,       per-pixel geometry tables -- e.g. the 5-degree floor-binned
   data/silver/,          disk-resolved golden parquets, or the aperture-photometry
   data/golden/)          golden layer for barely-resolved approach-phase frames.
        |
        v
photometry/               Model fitting and aggregation, decoupled from how the
  models/                 geometry was produced: BasePhotometricModel subclasses
  fitting/                (Hapke, baselines), LeastSquaresFitter, disk_integrate/
  aggregation/            sphere_forward for whole-disk aggregation.
```

Notebooks (`notebooks/`) are the orchestration layer on top of this: they load a
golden-layer parquet, call into `photometry`'s models/fitting, and produce the analysis
and plots for a specific result. Reusable logic that a notebook depends on belongs in
`src/` or `scripts/`, not duplicated in notebook cells — see `scripts/golden/build_disk_resolved_golden.py`
for an example of binning logic extracted out of `Hapke.ipynb` into version control.

## Install

```bash
git clone git@github.com:kaushikmastro/PhotometryPipeline.git
cd PhotometryPipeline
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.txt   # adds pandas/pyarrow/duckdb/astropy/pytest/etc. on
                                   # top of setup.cfg's core numpy/scipy/pymc/spiceypy
```

This gets you the full `photometry`/`photometry_etl` packages, model fitting, and the
test suite. It does **not** get you Dawn image data, SPICE kernels, or a Gaskell shape
model — the geometry-ingestion and grinding steps require Curta HPC's data layout under
`/scratch` and are not bundled with the repo (see Data provenance below).

### Quickstart: verify the install works

```bash
pytest tests/ -q
```

Expect `151 passed, 8 skipped, 2 xfailed` and nothing red. This runs entirely on
synthetic/mocked data — no SPICE kernels or real imagery needed, verified by running
this exact command in a from-scratch virtual environment with only the packages above
installed. `--run-slow` additionally runs real-data regression fits (bit-for-bit
reproduction against committed reference parquets) that need the actual 12GB+ silver
layer on disk — skipped by default for that reason, not run in CI.

## Data provenance

Raw Dawn FC2 imagery and SPICE kernels live outside the repo, under a Curta HPC
`/scratch` mount (symlinked in as `data/`). Geometry is ground per mission phase:

| phase | what it is | area-column schema |
|---|---|---|
| RC | Rotation Characterization | full (`range_km`, `pixel_solid_angle_sr`, `pixel_area_km2`, `projected_area_km2`) |
| approach | pre-orbit-insertion imaging (point-source to fully-resolved, by range) | full |
| Survey | ~2700 km altitude mapping | **missing** the 4 area columns above |
| HAMO | High-Altitude Mapping Orbit | **missing** |
| LAMO | Low-Altitude Mapping Orbit | **missing** |

**This is a real, currently-live caveat, not a footnote**: `compute_geometry()`'s output
schema gained those four area columns partway through this project's life. RC and
approach were ground (or re-ground) after that change and have them; the ~42,000
existing Survey/HAMO/LAMO raw geometry parquets predate it and don't. Anything built on
`disk_integrate.py` (which needs `projected_area_km2`) against Survey/HAMO/LAMO data
requires re-running the geometry grind for those images first — nothing has been
regenerated yet. See `CLAUDE.md`'s "Disk-Integrated Photometry" section for the full
detail and the physics reasoning behind which area column disk integration uses.

## Honest limitations

- **Disk-integrated absolute radiometric calibration is unverified.** The
  aperture-photometry method used for barely-resolved approach-phase frames has been
  validated for *where* it measures (SPICE-seeded target localization, confirmed against
  real per-pixel geometry) but not for absolute flux scale — two independent controls
  both failed to validate an absolute conversion, for different, well-understood reasons
  (whole-frame background contamination at high frame-fill; crescent-illumination
  centroid/aperture-shape mismatch at moderate-to-high phase). Treat aperture-derived I/F
  as relative only; any fit against it needs a free scale factor, not a fixed conversion.
  Full record in `CLAUDE.md`.
- **Uncertainties are statistical only.** `parameter_errors` (from the least-squares
  fit's covariance matrix) and the aperture-photometry uncertainty terms
  (background/shot noise, aperture-radius and centroid-position sensitivity) capture
  measurement scatter, not a systematic error budget — no propagated uncertainty from
  shape-model accuracy, absolute radiometric calibration, or SPICE pointing knowledge.
- **Lommel-Seeliger's `w` rails at its upper bound (1.0) for all phase bins below ~45°**,
  with `reduced_chi_square` correspondingly high there. Root cause not yet isolated
  between three candidates (the `/4` normalization convention, I/F normalization in the
  golden layer, or LS genuinely being unable to represent Vesta's low-phase brightness).
  See `CLAUDE.md`'s "Parked / revisit later" section.
- The committed Case 1 Hapke result uses a **preliminary** Gaskell DSK256 shape model,
  not the final mission-science shape product.

## Running tests

```bash
pytest tests/ -q              # fast suite: unit tests, synthetic-data fit-recovery,
                               # closed-form physics checks. No real data needed.
pytest tests/ -q --run-slow   # adds real-data regression tests: bit-for-bit
                               # reproduction against committed reference parquets.
                               # Needs the actual silver-layer data on disk (12GB+).
```

CI (`.github/workflows/python-ci.yml`) runs the fast suite on every push and on PRs into
`main`, plus a non-blocking `ruff` lint report.

## Citation / contact

Kaushik Mukherjee ([mukherjeekaushik107@gmail.com](mailto:mukherjeekaushik107@gmail.com)).
No formal citation (paper/DOI) yet — if you use this pipeline or its results, reach out
first. No LICENSE file is present yet; treat the code as all-rights-reserved until one is
added.

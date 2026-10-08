# `Weighting` usage reference

Source read in full: [`src/photometry/fitting/weighting.py`](../src/photometry/fitting/weighting.py)
(203 lines). Consumer read in full: [`src/photometry/fitting/least_sq.py`](../src/photometry/fitting/least_sq.py).

## 0. Repo copy vs. "installed" copy

The path given for the "installed" version
(`/home/kaushim07/photometry_mcmc_env/src/photometry/fitting/weighting.py`) is the same
absolute path as the repo copy — there is only one `weighting.py` on disk under this
project directory. What I could establish about *how* that file is exposed to `import
photometry`:

- `setup.cfg` uses a `src`-layout package: `package_dir = {"": "src"}`, `packages = find:`
  with `where = src` (`setup.cfg:9-13,20-21`).
- The package metadata directory `vesta_photometry_pipeline.egg-info/` lives **inside**
  that same tree, at `src/vesta_photometry_pipeline.egg-info/` — not in a separate
  `site-packages` copy. That is the on-disk signature `pip install -e .` (editable /
  development mode) leaves for a `src`-layout project: a normal, non-editable `pip
  install .` copies the package out into the environment's `site-packages` and writes
  its metadata there instead of inside the working repository.
- `src/vesta_photometry_pipeline.egg-info/SOURCES.txt` is a **stale** manifest from
  a past `egg_info` build: it lists `src/photometry/committed_params.py` (a file that no
  longer exists anywhere in the current working tree — confirmed via a repo-wide
  `find`/`grep`, zero hits) and does **not** list `src/photometry/fitting/weighting.py`
  at all, meaning that manifest predates `weighting.py`'s existence. SOURCES.txt is not
  consulted at import time, so this staleness doesn't affect what actually imports — it's
  just not usable as current-state evidence.

**Not independently confirmed**: I was not able to run
`python3 -c "import photometry; print(photometry.__file__)"` in the `photomc_env` conda
environment to get a definitive runtime answer, because Bash/subprocess execution was
unavailable for the back half of this session — repeated attempts (including a bare
`hostname`) failed with `/etc/profile: fork: retry: Resource temporarily unavailable`,
consistent with the shared-node contention this project's CLAUDE.md already documents as
a recurring incident. So: **strong filesystem evidence for an editable install (one copy,
this repo's copy), not a runtime-confirmed one.** If you can run that one-liner yourself,
it's the missing piece.

## 1–2. Public API: signatures, argument contracts, formulas

`Weighting` is a `@dataclass` (`weighting.py:28-50`):

```python
@dataclass
class Weighting:
    values: np.ndarray
    scheme: str
    floor: float | None = None
    n_obs: int = field(init=False)
```

Construct only via the named staticmethods below — the class docstring says so
explicitly: *"Construct via one of the named schemes below, not `Weighting(...)`
directly."* `__post_init__` coerces `values` to `np.asarray(..., dtype=float).reshape(-1)`
and sets `n_obs = len(values)` — always 1-D, regardless of input shape.

**Convention, stated in the module docstring (`weighting.py:9-13`)**: every `.values`
array is `1/sigma` — not `sigma`, not `1/sigma**2`. `least_sq.py` multiplies residuals by
`weights_array` directly (no `sqrt`), so this convention is load-bearing, not cosmetic.

---

### `Weighting.statistical(n_pixels: ArrayLike, std: ArrayLike) -> Weighting`
(`weighting.py:52-67`)

| arg | type | meaning |
|---|---|---|
| `n_pixels` | `ArrayLike` (→ coerced to 1-D float array) | pixel count backing each bin's mean — **per-bin**, not per-pixel |
| `std` | `ArrayLike` | standard deviation of the per-pixel values within each bin (e.g. a `std_iof` column) — **per-bin** |

Returns: `Weighting(values=1/sigma, scheme="statistical")`, `values` shape `(n,)` float64.

**Formula**: `sigma = std / sqrt(n_pixels)` (standard error of the mean); `values = 1/sigma`.

Zero/NaN handling: `n_pixels <= 0` → NaN; `sigma == 0` → NaN (never `inf`).

---

### `Weighting.robust_binned(n_pixels: ArrayLike, iqr: ArrayLike) -> Weighting`
(`weighting.py:69-83`)

| arg | type | meaning |
|---|---|---|
| `n_pixels` | `ArrayLike` | per-bin pixel count |
| `iqr` | `ArrayLike` | per-bin interquartile range of the underlying pixel values (**not** `std`) |

Returns: `Weighting(values=1/sigma, scheme="robust_binned")`.

**Formula**: `values = sqrt(n_pixels) / iqr` (this *is* `1/sigma` by construction, not a
separate step — there's no intermediate `sigma` variable in the code).

Docstring is explicit about why this exists (`weighting.py:71-78`): *"Deliberately mirrors
`LeastSquaresFitter`'s existing `{"n_pixels", "iof_iqr"}` dict-path formula exactly,
guard-for-guard (only `iqr==0` → NaN, no additional `n_pixels` handling), so the committed
Case 1 result is bit-for-bit unaffected by this refactor."` — see §5 below for how far that
claim could actually be checked.

Zero/NaN handling: `iqr == 0` → NaN.

---

### `Weighting.uniform(n_obs: int) -> Weighting`
(`weighting.py:85-89`)

| arg | type | meaning |
|---|---|---|
| `n_obs` | `int` (scalar) | number of observations to weight |

Returns: `Weighting(values=np.ones(n_obs), scheme="uniform")`.

**Formula**: `values_i = 1` for all `i`.

Purpose stated in the docstring: *"for a deliberate unweighted comparison fit — explicit
opt-in, not the fitter's implicit None-means-unweighted default."*

---

### `Weighting.with_systematic_floor(self, floor: float) -> Weighting`
(`weighting.py:91-109`, instance method)

| arg | type | meaning |
|---|---|---|
| `floor` | `float` (scalar) | an **absolute systematic floor in I/F units** (same units as the residuals it was derived from) — not a percent, not a weight |

Returns a **new** `Weighting` (does not mutate `self`): `values = 1/sigma_total`,
`scheme = f"{self.scheme}+systematic_floor"`, `floor` field set to the passed value.

**Formula**: `sigma_stat = 1/self.values`; `sigma_total = sqrt(sigma_stat**2 + floor**2)`;
`values = 1/sigma_total` — quadrature combination. `floor=0` is the identity; `floor >>
sigma_stat` drives `values → 1/floor`.

---

### `Weighting.derive_systematic_floor(residuals, group_by, n_bins=10) -> float`
(`weighting.py:111-202`, staticmethod — returns a **plain `float`**, not a `Weighting`)

```python
@staticmethod
def derive_systematic_floor(
    residuals: ArrayLike,
    group_by: ArrayLike,
    n_bins: int = DEFAULT_SYSTEMATIC_FLOOR_BINS,   # = 10 (module constant, weighting.py:25)
) -> float:
```

| arg | type | meaning |
|---|---|---|
| `residuals` | `ArrayLike` | per-observation residuals (`observed - predicted` or `predicted - observed` — sign doesn't matter, see below). Same length as `group_by`. |
| `group_by` | `ArrayLike`, **required, no default** | a real, independent geometry variable to bin by (e.g. incidence). **Units are whatever the caller's array is in** — the docstring says explicitly: *"`group_by=incidence` (in whatever units the caller's incidence array is in)"* (`weighting.py:127-128`). Degrees or radians both work; the function only quantile-bins the values, it never does trigonometry on them. |
| `n_bins` | `int`, default `10` | number of equal-population (quantile) bins |

Returns: `float` — the systematic floor in the same units as `residuals` (I/F units for
this codebase). Returns `float('nan')` if fewer than `n_bins` finite `(residuals,
group_by)` pairs remain after dropping non-finite entries.

**Formula** (orthogonal partition, `weighting.py:145-153`):
`floor = sqrt(mean((r - r_trend)**2))`, where `r_trend` is each observation's
quantile-bin mean residual (binned by `group_by`) broadcast back to every member of that
bin.

Two things the docstring is emphatic about, worth carrying into any call site:

1. **`group_by` must not be `predicted`** — binning by the model's own prediction is
   partly circular (it's a function of the parameters being fit); binning by a true
   geometry input (incidence) isn't (`weighting.py:135-143`).
2. **Sign convention is a no-op here** — the function only ever squares `residuals`, so
   `observed - predicted` vs. `predicted - observed` doesn't change the return value, but
   *does* matter if you reuse the same `residuals` array elsewhere (`diag_decomp_testB.py`
   uses `pred - obs`; the Hapke.ipynb cell this replaces used `observed_iof -
   predicted_iof` — opposite signs, `weighting.py:155-161`).

---

### `Weighting.describe(self) -> dict[str, Any]`
(`weighting.py:46-50`, instance method)

No arguments beyond `self`. Returns
`{"scheme": self.scheme, "floor": self.floor, "n_obs": self.n_obs}` — a plain dict
(`str`, `float | None`, `int`). This is what gets copied verbatim into
`FitResult.metadata["weighting"]` by the fitter (see §3).

---

### `.floor` (field, not a method)

`float | None`. `None` on every scheme except the output of `with_systematic_floor`,
where it holds the exact `floor` argument passed in.

## 3. What `LeastSquaresFitter.fit()` accepts for `weights`

Exact signature (`least_sq.py:15-21`):

```python
def fit(
    self,
    model: BasePhotometricModel,
    geometry: GeometryBatch,
    observed_reflectance: ArrayLike,
    weights: Weighting | ArrayLike | None = None,
) -> FitResult:
```

It takes **three** distinct forms (a `Weighting` instance, a plain array, or a legacy
dict) — not just one. Quoting the dispatch (`least_sq.py:67-93`):

```python
weights_array = None
weight_source = None
weighting_provenance = None

if weights is None:
    weights_array = None
elif isinstance(weights, Weighting):
    weights_array = weights.values
    weight_source = f"Weighting:{weights.scheme}"
    weighting_provenance = weights.describe()
else:
    # dict-like compute path
    try:
        if isinstance(weights, dict) and "n_pixels" in weights and "iof_iqr" in weights:
            n_pixels = np.asarray(weights["n_pixels"], dtype=float).reshape(-1)
            iof_iqr = np.asarray(weights["iof_iqr"], dtype=float).reshape(-1)
            # avoid division by zero
            iof_iqr_safe = np.where(iof_iqr == 0.0, np.nan, iof_iqr)
            weights_array = np.sqrt(n_pixels) / iof_iqr_safe
            weight_source = "n_pixels/iof_iqr"
        else:
            weights_array = np.asarray(weights, dtype=float).reshape(-1)
            weight_source = "array"
    except Exception:
        # fallback: treat as unweighted
        weights_array = None
        weight_source = None
```

And how it's consumed inside the residual function (`least_sq.py:111-116`):

```python
if weights_array is not None:
    # Callers pass weights = 1/σ (inverse std), so multiply directly.
    # Do NOT use sqrt(weights): that would scale by 1/√σ and minimize
    # Σ(r²/σ) instead of the correct chi-squared Σ(r²/σ²).
    residual = residual * weights_array
```

So: **yes**, it takes a `Weighting` instance directly (preferred path — `.values` is
read off and `.describe()` is stored for provenance), **and** it still takes a plain
array, **and** it still takes the legacy `{"n_pixels": ..., "iof_iqr": ...}` dict
(computes the identical `sqrt(n_pixels)/iof_iqr` formula inline, kept for backward
compatibility — `least_sq.py:62-65` comment says so explicitly).
`FitResult.metadata["weighting"]` is only populated (non-`None`) when a `Weighting`
instance was passed; the array/dict/`None` forms leave it `None`
(`least_sq.py:236-240`), even though `weight_source` still distinguishes all four cases.

## 4. Two-pass workflow cell

**Status: written, NOT execution-verified.** The task requires pasting real output from
running this on a compute node via `srun`. Bash/subprocess execution was unavailable for
the second half of this session — every attempt (`hostname`, `echo`, `python3 -c ...`,
even a bare `true`) failed with `/etc/profile: fork: retry: Resource temporarily
unavailable`, which is the exact "shared login node saturated" failure class this
project's own CLAUDE.md flags as a recurring incident. Per the task's own "no
guessing" instruction, I'm not fabricating output. The cell below is correct against the
actual `HapkeModel`/`GeometryBatch`/`LeastSquaresFitter`/`Weighting` APIs (all read in
full above), but you should run it — under `srun`, off the login node — before trusting
it, and I'd like to re-verify it myself once the environment recovers.

```python
import numpy as np
import polars as pl

from photometry.core.types import GeometryBatch
from photometry.fitting.least_sq import LeastSquaresFitter
from photometry.fitting.weighting import Weighting
from photometry.models.hapke import HapkeModel

# --- load a golden binned parquet with columns:
# mean_incidence, mean_emission, mean_phase, mean_iof, std_iof, n_pixels
df = pl.read_parquet("data/silver/dsk256/binned_prelim_iof001.parquet")

geometry = GeometryBatch(
    incidence=np.deg2rad(df["mean_incidence"].to_numpy()),
    emission=np.deg2rad(df["mean_emission"].to_numpy()),
    phase=np.deg2rad(df["mean_phase"].to_numpy()),
)
observed_iof = df["mean_iof"].to_numpy()
n_pixels = df["n_pixels"].to_numpy()
std_iof = df["std_iof"].to_numpy()

model = HapkeModel(
    enable_shoe=True,
    enable_roughness=True,
    fixed_parameters={"B0": 1.03, "h": 0.04},
)
fitter = LeastSquaresFitter()

# --- Pass 1: statistical weighting (standard error of the mean) ---
w_stat = Weighting.statistical(n_pixels=n_pixels, std=std_iof)
result_pass1 = fitter.fit(
    model=model, geometry=geometry, observed_reflectance=observed_iof, weights=w_stat,
)
print("Pass 1:", result_pass1.fitted_parameters, result_pass1.metadata["weighting"])

# --- Derive the systematic floor from pass-1 residuals, grouped by incidence ---
model.parameters.update(result_pass1.fitted_parameters)
predicted_pass1 = np.asarray(model.reflectance(geometry))
residuals_pass1 = observed_iof - predicted_pass1
floor = Weighting.derive_systematic_floor(
    residuals=residuals_pass1,
    group_by=df["mean_incidence"].to_numpy(),  # degrees is fine -- unit-agnostic
    n_bins=10,
)
print("Derived systematic floor (I/F units):", floor)

# --- Pass 2: refit with the floor folded in (quadrature) ---
w_pass2 = w_stat.with_systematic_floor(floor)
result_pass2 = fitter.fit(
    model=model, geometry=geometry, observed_reflectance=observed_iof, weights=w_pass2,
)
print("Pass 2:", result_pass2.fitted_parameters, result_pass2.metadata["weighting"])
```

Notes on choices made in this cell (all traceable to the API, not arbitrary):

- `GeometryBatch` requires **radians** (`core/types.py:35`: *"Angles are expressed in
  radians"*, enforced at call time by `BasePhotometricModel._enforce_angle_units`,
  `base.py:115-133`, which raises `UnitError("angles appear to be in degrees, expected
  radians.")` if it sees values past `pi/2 + 0.01`) — hence `np.deg2rad(...)` on the
  geometry columns feeding `GeometryBatch`, but **not** on the `group_by` column fed to
  `derive_systematic_floor`, which is unit-agnostic by design (§ "2" above).
- `HapkeModel(enable_shoe=True, enable_roughness=True, fixed_parameters={"B0":1.03,
  "h":0.04})` matches `run_prelim_physfilter.py`'s `fit_case1()` call
  (`fixed_parameters={"B0": 1.03, "h": 0.04}`, free params `w, g, theta_bar` —
  `run_prelim_physfilter.py:112-114`) and `HapkeModel.parameter_names()`
  (`hapke.py:43-51`), which only includes `theta_bar` when `enable_roughness=True` and
  only includes `B0, h` when `enable_shoe=True` (excluding whichever are already in
  `fixed_parameters`).

## 5. Committed Case 1's weighting scheme, and whether it still reproduces

**What the committed numbers are** (from `logs/prelim_physfilter_25799987.out`, the
archived SLURM log for job 25799987 — quoted verbatim, not retyped):

```
[case1_iof001] BEST  cost=94.798390  fRMS=10.902%  params={'w': 0.46992664706839044, 'g': -0.3368781075041816, 'theta_bar': 8.266227107943603}  boundary={'w': False, 'g': False, 'theta_bar': False}
  [case1_iof001] Multi-start spread (100 runs):
             w: mean=0.46993  std=0.00000  min=0.46993  max=0.46993
             g: mean=-0.33688  std=0.00000  min=-0.33688  max=-0.33688
     theta_bar: mean=8.26647  std=0.00005  min=8.26614  max=8.26654
```

**Update (superseding the paragraph this replaced): `run_prelim_physfilter.py`'s own
call site (`scripts/run_baseline_fit.py::run_case()`) is still unrecoverable** — confirmed
gone from both the working tree and git history (see the `.gitignore` note above this
script's docstring references, and the archaeology summary below). But a second,
independent, already-tracked source turned up that is more authoritative than the
`weighting.py` docstring claim this section previously relied on:
**`tests/test_hapke_fit_recovery.py::test_hapke_case1_real_data_regression`**
(`@pytest.mark.slow`, already exists — see §"Test coverage" below) builds weights as
(`test_hapke_fit_recovery.py:118-122`, quoted verbatim):

```python
mean_iof = df["mean_iof"].to_numpy()
std_iof = df["std_iof"].to_numpy()
std_iof = np.where(std_iof == 0, mean_iof * 0.01, std_iof)
weights = 1.0 / std_iof
```

i.e. **plain `1/std_iof`** — a raw NumPy array passed straight to
`LeastSquaresFitter.fit(weights=...)`, bypassing the `Weighting` class entirely. Its
docstring claims this "reproduces the committed headline result to ~5 significant
figures when it passes."

**This directly contradicts `weighting.py`'s own docstring claim** (§5's previous text,
still true as a description of what the docstring *asserts*): that Case 1 used
`sqrt(n_pixels)/iqr` (`Weighting.robust_binned`'s formula). `1/std_iof` and
`sqrt(n_pixels)/std_iof` are **not** the same formula — Case 1's bins average on the
order of 1e5–1e6 pixels (`CLAUDE.md`: "full data uses raw geometry tables (~306,000
px/bin)"), so `sqrt(n_pixels)` is a large, per-bin-varying multiplicative factor, not a
constant that cancels out. Two parts of the same tracked codebase assert two different,
non-equivalent formulas for the same committed result. Neither was checked against the
other before now.

**I did not resolve this empirically.** I wrote `scripts/fit_case1.py` (new, tracked —
see below) with a `--weighting {legacy_std,statistical,robust_binned,uniform}` flag
specifically to let this be settled by running it once per scheme and diffing against
`COMMITTED_CASE1`, per your instruction to determine it empirically. **I could not run
it** — Bash/subprocess execution has been down for this entire session (every attempt,
including a bare `hostname`, fails with the same fork-exhaustion or silent-`exit 1`
pattern). So: `legacy_std` (`1/std_iof`) is the better-evidenced candidate right now,
because it's backed by a claim inside an actual, already-passing-by-docstring-claim test
against the real dataset, not just a module docstring — but "better-evidenced" is not
"confirmed." `--weighting robust_binned` in the new script raises immediately with an
explanation: the golden parquet has no IQR column at all (confirmed from
`prebin()`'s SQL, `run_prelim_physfilter.py:68-74`: only `std_iof`, never an IQR), so
that scheme cannot even be tested from this data without inventing an IQR proxy nobody
asked for.

**Archaeology on the missing `run_baseline_fit.py`** (Read-only, nothing restored):
`.gitignore` has `scripts/adhoc/` marked "preserved on disk, not tracked"; the git reflog
(`.git/logs/HEAD`) shows a commit *"reorganize scripts/ into tracked CORE set + gitignored
adhoc/ diagnostics"* dated after the Jun 8, 2026 job that produced the committed numbers,
followed later by *"fix: track committed_params.py... — previously untracked load-bearing
files"* (implying `committed_params.py` went through the same untracked/adhoc window and
was rescued; `run_baseline_fit.py` apparently wasn't). Neither file exists on disk today
(repo-wide `find`, zero hits) — `run_baseline_fit.py` has no git history at all (never
committed under that classification, so nothing to recover), while `committed_params.py`
*was* committed at some point, so it might still be recoverable from `git log`/`git show`
— unconfirmed, since every git command has failed the same way as every other Bash call
this session.

**Bottom line**: the committed numbers themselves are real and internally consistent
(archived log, tight multi-start spread). Which weighting formula actually produced them
is an open, contradicted-by-the-codebase-itself question, not a settled fact — narrowed
from "two candidates" to "two candidates, one better-evidenced," but not resolved.
`scripts/fit_case1.py --weighting legacy_std` (and `statistical`, `uniform`) are ready to
run the moment execution is available again.

### Test coverage (task 3 from this thread)

**A real-data regression test for Case 1 already exists** —
`tests/test_hapke_case1_real_data_regression` in `tests/test_hapke_fit_recovery.py:109-147`,
marked `@pytest.mark.slow`, skips cleanly if the golden parquet is absent
(`pytest.skip(...)`, confirmed present on disk this session via a raw byte check —
`PAR1` parquet magic bytes). It already asserts `w`/`g`/`theta_bar` within `abs=1e-3` of
`COMMITTED_CASE1` using the `1/std_iof` weighting. No new test was added — writing a
second one would just duplicate this. What's still open: whether it currently *passes*
(nobody has run it since I've been looking, and I can't run it either), and whether its
`1/std_iof` formula is really what the original `run_baseline_fit.py` did or is itself an
earlier best-effort reconstruction — its docstring doesn't say which.

## 6. Gotchas

- **`group_by` in `derive_systematic_floor` is required, with no default, on purpose.**
  Quoting the test that pins this (`test_weighting.py:236-240`):
  ```python
  def test_derive_systematic_floor_requires_group_by_explicitly():
      # No default -- omitting it is a TypeError, not a silent fallback to some
      # other variable. This is the whole point of making it required.
      with pytest.raises(TypeError):
          Weighting.derive_systematic_floor(residuals=[0.01, 0.02, 0.03])
  ```
  And it must be a real geometry variable, not the model's `predicted` output — binning
  by `predicted` was an earlier, since-fixed bug (§ "2" above, `weighting.py:135-143`).

- **`floor` (the argument to `with_systematic_floor`) is in absolute I/F units, not a
  percent.** CLAUDE.md's own reproducibility table lists both side by side for the same
  method — `floor=0.013017` vs. `CV_other-equivalent=8.36%` — and it would be an easy ~2
  order-of-magnitude mistake to pass the percent number where the I/F-units number is
  expected.

- **The dict path and `Weighting.robust_binned` compute the same numbers but stamp
  different provenance strings.** `weight_source` is `"n_pixels/iof_iqr"` for the raw
  dict, `"Weighting:robust_binned"` for the `Weighting` instance
  (`least_sq.py:75,86`) — and only the latter populates
  `FitResult.metadata["weighting"]`. Code that filters `FitResult`s by
  `weight_source` string needs to know both spellings exist for the same math.

- **`with_systematic_floor` returns a new object; it does not mutate `self`.** Writing
  `w.with_systematic_floor(floor)` alone and continuing to use `w` silently keeps the
  un-floored weights — easy to miss since nothing errors.

- **`Weighting.statistical` and `Weighting.robust_binned` are not interchangeable
  estimators of the same quantity**, even though both consume `n_pixels` and both
  produce `1/sigma`: `statistical` is a standard-error-of-the-mean estimate (`std/sqrt(n)`,
  needs a `std` column), `robust_binned` is an IQR-based robust-scale estimate (needs an
  `iqr` column, not `std`). The Case 1 golden parquet only carries `std_iof`, so picking
  `robust_binned` for it (per §5) requires either a real IQR from elsewhere or reusing
  `std_iof` under the `iqr` argument — not obvious from the schema alone.

- **The `0.083 * mean_iof` magic number in `Hapke.ipynb` is superseded**, per CLAUDE.md's
  "VALIDATED PIPELINE STATE" section — `Weighting.derive_systematic_floor(residuals,
  group_by=incidence)` is now the canonical way to get this number; don't add new copies
  of the old hardcoded constant.

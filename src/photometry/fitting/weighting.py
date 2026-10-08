"""Composable weighting schemes for `LeastSquaresFitter`, extracted out of notebook
cells where each one computed weights inline (a systematic floor as a bare magic
number in one Hapke cell, untested, with no record of which scheme produced a given
fit). This module is data preparation, not fitter logic: `LeastSquaresFitter`'s
residual math is unchanged, it just also accepts a `Weighting` instance (in addition
to the array/dict/None forms it already accepted) and copies `.describe()` into
`FitResult.metadata["weighting"]` for provenance.

Convention (matches `LeastSquaresFitter.fit`'s docstring comment: "Callers pass
weights = 1/sigma (inverse std), so multiply directly"): every `Weighting.values`
array is 1/sigma, not sigma and not 1/sigma**2. This has bitten this codebase before
(see the fitter's own comment on why `sqrt(weights)` would be wrong), so every
constructor here is tested against a hand-computed case for exactly this convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from photometry.core.types import ArrayLike

DEFAULT_SYSTEMATIC_FLOOR_BINS = 10


@dataclass
class Weighting:
    """A set of per-observation fit weights (1/sigma) plus provenance.

    Construct via one of the named schemes below, not `Weighting(...)` directly --
    the constructors are what guarantee the 1/sigma convention and the zero/NaN
    handling tested in `tests/test_weighting.py`.
    """

    values: np.ndarray
    scheme: str
    floor: float | None = None
    n_obs: int = field(init=False)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float).reshape(-1)
        self.n_obs = int(self.values.size)

    def describe(self) -> dict[str, Any]:
        """Provenance record for `FitResult.metadata["weighting"]` -- so two fits'
        weighting choices are provably comparable instead of implicit in notebook
        code that produced them."""
        return {"scheme": self.scheme, "floor": self.floor, "n_obs": self.n_obs}

    @staticmethod
    def statistical(n_pixels: ArrayLike, std: ArrayLike) -> Weighting:
        """1/sigma from the standard error of the mean: sigma = std / sqrt(n).

        n_pixels <= 0 or std == 0 (no meaningful uncertainty estimate) both produce
        NaN weights rather than inf or a raised error, matching the zero-handling
        convention already used for the robust_binned/dict path in least_sq.py.
        """
        n_pixels_arr = np.asarray(n_pixels, dtype=float).reshape(-1)
        std_arr = np.asarray(std, dtype=float).reshape(-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = std_arr / np.sqrt(n_pixels_arr)
        sigma = np.where(n_pixels_arr <= 0, np.nan, sigma)
        sigma_safe = np.where(sigma == 0.0, np.nan, sigma)
        values = 1.0 / sigma_safe
        return Weighting(values=values, scheme="statistical")

    @staticmethod
    def robust_binned(n_pixels: ArrayLike, iqr: ArrayLike) -> Weighting:
        """1/sigma = sqrt(n_pixels) / iqr -- the existing binned-golden-layer scheme.

        Deliberately mirrors LeastSquaresFitter's existing {"n_pixels", "iof_iqr"}
        dict-path formula exactly, guard-for-guard (only iqr==0 -> NaN, no additional
        n_pixels handling), so the committed Case 1 result is bit-for-bit unaffected
        by this refactor -- see
        test_weighting.py::test_robust_binned_reproduces_existing_dict_path_exactly.
        """
        n_pixels_arr = np.asarray(n_pixels, dtype=float).reshape(-1)
        iqr_arr = np.asarray(iqr, dtype=float).reshape(-1)
        iqr_safe = np.where(iqr_arr == 0.0, np.nan, iqr_arr)
        values = np.sqrt(n_pixels_arr) / iqr_safe
        return Weighting(values=values, scheme="robust_binned")

    @staticmethod
    def uniform(n_obs: int) -> Weighting:
        """All-ones weights, for a deliberate unweighted comparison fit -- explicit
        opt-in, not the fitter's implicit None-means-unweighted default."""
        return Weighting(values=np.ones(int(n_obs), dtype=float), scheme="uniform")

    def with_systematic_floor(self, floor: float) -> Weighting:
        """Combine this weighting's implied sigma with an absolute systematic floor
        in quadrature: sigma_total = sqrt(sigma_stat**2 + floor**2), returned as a
        new Weighting with values = 1/sigma_total.

        floor=0 is the identity (sigma_total == sigma_stat, values unchanged);
        floor >> sigma_stat makes sigma_total -> floor, so values -> 1/floor --
        both verified in test_weighting.py, along with the monotonic-decrease
        property (a larger floor can only widen sigma_total, never narrow it).
        """
        floor_value = float(floor)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_stat = 1.0 / self.values
        sigma_total = np.sqrt(sigma_stat**2 + floor_value**2)
        sigma_total_safe = np.where(sigma_total == 0.0, np.nan, sigma_total)
        new_values = 1.0 / sigma_total_safe
        return Weighting(
            values=new_values, scheme=f"{self.scheme}+systematic_floor", floor=floor_value
        )

    @staticmethod
    def derive_systematic_floor(
        residuals: ArrayLike,
        group_by: ArrayLike,
        n_bins: int = DEFAULT_SYSTEMATIC_FLOOR_BINS,
    ) -> float:
        """Compute an absolute systematic floor (I/F units) from a residual
        decomposition, replacing the `0.083 * mean_iof` magic number that lived in
        one Hapke.ipynb cell (search "Derive the floor from YOUR measured residual
        decomposition, don't assert it" -- the comment already said what to do,
        this is that computation, made real and testable).

        PROVENANCE: this reconstructs the orthogonal decomposition in
        scripts/diagnostics/diag_decomp_testB.py (the CV_trend/CV_other split
        CLAUDE.md's "VALIDATED PIPELINE STATE" cites: CV-RMSE=9.223% =
        sqrt(5.125^2+7.669^2), reduced_chi_square floor-weighted by
        CV_other*mean_iof). `group_by=incidence` (in whatever units the caller's
        incidence array is in) reproduces that script's method: bin the residual by
        incidence into `n_bins` groups, isolate the bin-mean ("trend") component,
        and take the RMS of what's left. **Any other `group_by` measures a
        different quantity and is the caller's call to make** -- this parameter is
        REQUIRED (no default) precisely so nobody accidentally gets a floor
        computed against the wrong variable without realizing it.

        Why `group_by` must be a real, independent geometry variable and not the
        model's own `predicted` reflectance (an earlier version of this function
        did exactly that): CV_trend is meant to isolate residual structure correlated with
        GEOMETRY -- a signature of model inadequacy (roughness/shadowing
        mis-modeled at high incidence, say), not with the brightness level per se.
        `predicted` is itself a function of the parameters being fitted, so binning
        by it partly measures trend-with-brightness, a different and partly
        circular quantity. `incidence` (or another true geometry input) doesn't
        have that problem.

        Method (orthogonal partition, exact by construction):
          1. Group observations into `n_bins` equal-population (quantile) bins by
             `group_by`.
          2. r_trend = the bin-mean residual, broadcast back to every observation
             in that bin (the "systematic trend" component).
          3. r_other = residuals - r_trend (orthogonal to r_trend by construction:
             each bin's r_other has zero mean within that bin, same as
             diag_decomp_testB.py's r_trend . r_other = 0 check).
          4. floor = sqrt(mean(r_other**2)) -- the RMS of the within-bin residual.

        Sign convention: this function only ever squares `residuals`, so it is
        insensitive to whether the caller computed `observed - predicted` or
        `predicted - observed` -- but be aware the two don't agree in this
        codebase: diag_decomp_testB.py uses `r = pred - obs`, while the Hapke.ipynb
        cell this replaces uses `residuals = observed_iof - predicted_iof` (the
        opposite sign). Match whichever convention the rest of your call site uses;
        it does not change what this function returns.

        No `observed`/`predicted` parameter, deliberately: CLAUDE.md's
        "CV_other * mean_iof" is CV_other (a percent, defined as
        sqrt(mean(r_other**2))/mean(observed)*100) multiplied back by
        mean(observed) -- that round trip cancels mean(observed) EXACTLY, leaving
        plain RMS(r_other). The floor therefore never actually depends on the
        absolute brightness scale, only on the residual itself; carrying `observed`
        or `predicted` through the computation just to cancel them back out would
        invite someone to "fix" it into the two-step percent form later. Don't.

        NaN/non-finite pairs (by position, between `residuals` and `group_by`) are
        dropped before binning. Returns NaN if fewer than n_bins finite pairs
        remain (not enough data to form the requested number of bins).
        """
        residuals_arr = np.asarray(residuals, dtype=float).reshape(-1)
        group_by_arr = np.asarray(group_by, dtype=float).reshape(-1)
        if residuals_arr.shape != group_by_arr.shape:
            raise ValueError(
                f"residuals shape {residuals_arr.shape} != group_by shape {group_by_arr.shape}"
            )

        valid = np.isfinite(residuals_arr) & np.isfinite(group_by_arr)
        r = residuals_arr[valid]
        g = group_by_arr[valid]
        if r.size < n_bins:
            return float("nan")

        quantile_edges = np.quantile(g, np.linspace(0.0, 1.0, n_bins + 1))
        # Interior edges only: digitize's boundaries are the n_bins-1 cut points
        # between bins, so bin index is naturally in [0, n_bins-1] without clipping
        # away real data at the extremes.
        bin_index = np.digitize(g, quantile_edges[1:-1])

        r_trend = np.empty_like(r)
        for b in range(n_bins):
            mask = bin_index == b
            if mask.any():
                r_trend[mask] = float(r[mask].mean())

        r_other = r - r_trend
        return float(np.sqrt(np.mean(r_other**2)))

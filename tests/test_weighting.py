from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from photometry.core.types import GeometryBatch  # noqa: E402
from photometry.fitting.least_sq import LeastSquaresFitter  # noqa: E402
from photometry.fitting.weighting import Weighting  # noqa: E402
from photometry.models.baselines import LambertianModel  # noqa: E402


def _geometry(n: int = 20) -> GeometryBatch:
    rng = np.random.default_rng(1)
    incidence = rng.uniform(np.deg2rad(5.0), np.deg2rad(70.0), size=n)
    return GeometryBatch(incidence=incidence, emission=np.zeros(n), phase=np.zeros(n))


def _synthetic_observed(albedo: float, geometry: GeometryBatch) -> np.ndarray:
    model = LambertianModel()
    model.parameters["albedo"] = albedo
    return np.asarray(model.reflectance(geometry))


# ---------------------------------------------------------------------------
# 1/sigma convention -- hand-computed, per scheme. This convention has bitten
# this codebase before (see least_sq.py's own comment on why sqrt(weights)
# would minimize the wrong objective), so every constructor gets a direct check.
# ---------------------------------------------------------------------------


def test_statistical_returns_1_over_sigma_hand_computed():
    # sigma = std/sqrt(n) = 0.02/10 = 0.002 -> 1/sigma = 500
    w = Weighting.statistical(n_pixels=[100.0], std=[0.02])
    assert w.values[0] == pytest.approx(500.0)


def test_robust_binned_returns_1_over_sigma_hand_computed():
    # 1/sigma = sqrt(n_pixels)/iqr = sqrt(25)/0.5 = 10
    w = Weighting.robust_binned(n_pixels=[25.0], iqr=[0.5])
    assert w.values[0] == pytest.approx(10.0)


def test_uniform_returns_ones_not_zeros_or_sigma():
    w = Weighting.uniform(4)
    np.testing.assert_array_equal(w.values, np.ones(4))


@pytest.mark.parametrize(
    "scheme_factory",
    [
        lambda: Weighting.statistical(n_pixels=[64.0], std=[0.08]),
        lambda: Weighting.robust_binned(n_pixels=[64.0], iqr=[0.5]),
        lambda: Weighting.uniform(1),
    ],
    ids=["statistical", "robust_binned", "uniform"],
)
def test_all_schemes_produce_finite_positive_values_for_well_formed_input(scheme_factory):
    w = scheme_factory()
    assert np.all(np.isfinite(w.values))
    assert np.all(w.values > 0)


# ---------------------------------------------------------------------------
# Zero / NaN handling
# ---------------------------------------------------------------------------


def test_statistical_zero_std_gives_nan_not_inf():
    w = Weighting.statistical(n_pixels=[100.0], std=[0.0])
    assert np.isnan(w.values[0])


def test_statistical_zero_n_pixels_gives_nan_not_inf_or_error():
    w = Weighting.statistical(n_pixels=[0.0], std=[0.05])
    assert np.isnan(w.values[0])


def test_statistical_nan_input_propagates_as_nan():
    w = Weighting.statistical(n_pixels=[100.0], std=[np.nan])
    assert np.isnan(w.values[0])


def test_robust_binned_zero_iqr_gives_nan_not_inf():
    w = Weighting.robust_binned(n_pixels=[25.0], iqr=[0.0])
    assert np.isnan(w.values[0])


def test_robust_binned_nan_input_propagates_as_nan():
    w = Weighting.robust_binned(n_pixels=[np.nan], iqr=[0.5])
    assert np.isnan(w.values[0])


# ---------------------------------------------------------------------------
# with_systematic_floor: quadrature combination
# ---------------------------------------------------------------------------


def test_systematic_floor_zero_is_identity():
    w = Weighting.statistical(n_pixels=[100.0, 50.0], std=[0.02, 0.03])
    combined = w.with_systematic_floor(0.0)
    np.testing.assert_allclose(combined.values, w.values)


def test_systematic_floor_huge_approaches_1_over_floor():
    w = Weighting.statistical(n_pixels=[100.0], std=[0.001])  # tiny sigma_stat
    huge_floor = 1000.0
    combined = w.with_systematic_floor(huge_floor)
    assert combined.values[0] == pytest.approx(1.0 / huge_floor, rel=1e-6)


def test_systematic_floor_monotonically_decreases_weights_as_floor_grows():
    w = Weighting.statistical(n_pixels=[100.0], std=[0.02])
    floors = [0.0, 0.01, 0.05, 0.1, 1.0]
    values = [w.with_systematic_floor(f).values[0] for f in floors]
    assert all(later <= earlier for earlier, later in zip(values, values[1:]))


def test_systematic_floor_matches_exact_quadrature_formula():
    w = Weighting.statistical(n_pixels=[100.0], std=[0.02])  # sigma_stat = 0.002
    floor = 0.005
    combined = w.with_systematic_floor(floor)
    expected_sigma = np.sqrt(0.002**2 + floor**2)
    assert combined.values[0] == pytest.approx(1.0 / expected_sigma)


def test_systematic_floor_records_scheme_and_floor_in_describe():
    w = Weighting.statistical(n_pixels=[100.0], std=[0.02]).with_systematic_floor(0.01)
    d = w.describe()
    assert d["scheme"] == "statistical+systematic_floor"
    assert d["floor"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# describe() / provenance round-trip into FitResult.metadata
# ---------------------------------------------------------------------------


def test_describe_contents():
    w = Weighting.statistical(n_pixels=[100.0, 50.0], std=[0.02, 0.03])
    d = w.describe()
    assert d == {"scheme": "statistical", "floor": None, "n_obs": 2}


def test_weighting_describe_round_trips_into_fit_result_metadata():
    geometry = _geometry()
    observed = _synthetic_observed(0.5, geometry)
    n = len(observed)
    weighting = Weighting.robust_binned(n_pixels=np.full(n, 100.0), iqr=np.full(n, 0.01))
    model = LambertianModel()
    model.parameters["albedo"] = 0.2

    result = LeastSquaresFitter().fit(
        model=model, geometry=geometry, observed_reflectance=observed, weights=weighting
    )

    assert result.metadata["weighting"] == weighting.describe()
    assert result.metadata["weight_source"] == "Weighting:robust_binned"
    assert result.metadata["weighted"] is True


def test_non_weighting_inputs_leave_metadata_weighting_none():
    geometry = _geometry()
    observed = _synthetic_observed(0.5, geometry)
    model = LambertianModel()
    model.parameters["albedo"] = 0.2

    result = LeastSquaresFitter().fit(
        model=model, geometry=geometry, observed_reflectance=observed, weights=None
    )
    assert result.metadata["weighting"] is None


# ---------------------------------------------------------------------------
# Regression: Weighting.robust_binned must reproduce the existing
# {n_pixels, iof_iqr} dict path exactly -- the committed Case 1 result depends
# on that formula not changing under this refactor.
# ---------------------------------------------------------------------------


def test_robust_binned_reproduces_existing_dict_path_exactly():
    geometry = _geometry()
    observed = _synthetic_observed(0.5, geometry)
    n = len(observed)
    n_pixels = np.full(n, 100.0)
    iof_iqr = np.full(n, 0.01)
    model = LambertianModel()
    model.parameters["albedo"] = 0.2

    result_dict_path = LeastSquaresFitter().fit(
        model=model, geometry=geometry, observed_reflectance=observed,
        weights={"n_pixels": n_pixels, "iof_iqr": iof_iqr},
    )

    model2 = LambertianModel()
    model2.parameters["albedo"] = 0.2
    result_weighting_path = LeastSquaresFitter().fit(
        model=model2, geometry=geometry, observed_reflectance=observed,
        weights=Weighting.robust_binned(n_pixels=n_pixels, iqr=iof_iqr),
    )

    assert result_dict_path.fitted_parameters["albedo"] == pytest.approx(
        result_weighting_path.fitted_parameters["albedo"], abs=1e-12
    )
    assert result_dict_path.objective_value == pytest.approx(
        result_weighting_path.objective_value, abs=1e-12
    )
    assert result_dict_path.metadata["reduced_chi_square"] == pytest.approx(
        result_weighting_path.metadata["reduced_chi_square"], abs=1e-12
    )


def test_robust_binned_values_array_identical_to_hand_rolled_dict_formula():
    rng = np.random.default_rng(2)
    n_pixels = rng.uniform(10, 200, size=15)
    iof_iqr = rng.uniform(0.001, 0.05, size=15)

    hand_rolled = np.sqrt(n_pixels) / iof_iqr
    via_weighting = Weighting.robust_binned(n_pixels=n_pixels, iqr=iof_iqr).values

    np.testing.assert_allclose(via_weighting, hand_rolled)


# ---------------------------------------------------------------------------
# derive_systematic_floor
# ---------------------------------------------------------------------------


def test_derive_systematic_floor_requires_group_by_explicitly():
    # No default -- omitting it is a TypeError, not a silent fallback to some
    # other variable. This is the whole point of making it required.
    with pytest.raises(TypeError):
        Weighting.derive_systematic_floor(residuals=[0.01, 0.02, 0.03])  # type: ignore[call-arg]


def test_derive_systematic_floor_near_zero_for_pure_noise_no_trend():
    rng = np.random.default_rng(3)
    n = 500
    group_by = rng.uniform(0.0, 70.0, size=n)  # e.g. an incidence-angle-like array
    # Pure noise, no relationship to `group_by` at all -> after removing each
    # bin's mean, only noise-scale residual should remain, close to the true
    # noise sigma, not inflated by a real (absent) trend.
    noise_sigma = 0.01
    residuals = rng.normal(0.0, noise_sigma, size=n)

    floor = Weighting.derive_systematic_floor(residuals, group_by, n_bins=10)
    assert floor == pytest.approx(noise_sigma, rel=0.25)


def test_derive_systematic_floor_recovers_known_within_bin_scatter():
    # Construct residuals with an explicit per-bin trend plus a KNOWN within-bin
    # scatter -- the floor should recover approximately that within-bin scatter,
    # not the (much larger) total residual scale, since the trend component is
    # explicitly meant to be removed.
    rng = np.random.default_rng(4)
    n_bins = 5
    n_per_bin = 200
    within_bin_sigma = 0.003
    trend_values = np.array([0.05, -0.03, 0.02, -0.04, 0.01])  # arbitrary per-bin offsets

    group_by = np.concatenate(
        [np.full(n_per_bin, 10.0 * (b + 1)) for b in range(n_bins)]
    )  # e.g. incidence bins at 10, 20, 30, 40, 50 degrees
    residuals = np.concatenate(
        [
            trend_values[b] + rng.normal(0.0, within_bin_sigma, size=n_per_bin)
            for b in range(n_bins)
        ]
    )

    floor = Weighting.derive_systematic_floor(residuals, group_by, n_bins=n_bins)
    assert floor == pytest.approx(within_bin_sigma, rel=0.15)
    # And much smaller than the total (trend-inflated) residual RMS -- confirms
    # the trend really is being removed, not just diluted.
    total_rms = float(np.sqrt(np.mean(residuals**2)))
    assert floor < total_rms / 3.0


def test_derive_systematic_floor_result_depends_on_which_variable_is_grouped_by():
    # The reason group_by must be a real geometry variable, not `predicted`: two
    # different (and equally valid-looking) grouping variables should generally
    # give DIFFERENT floors, because they partition the residual differently.
    # This pins down that derive_systematic_floor is sensitive to that choice --
    # if it weren't, the whole "group_by must be geometry, not predicted"
    # distinction in the docstring would be moot.
    rng = np.random.default_rng(9)
    n = 400
    incidence = rng.uniform(0.0, 70.0, size=n)
    # A trend correlated with incidence (the "real" systematic this floor should
    # isolate)...
    residuals = 0.001 * incidence + rng.normal(0.0, 0.002, size=n)
    # ...and an UNRELATED second variable with no connection to the trend.
    unrelated = rng.uniform(-1.0, 1.0, size=n)

    floor_by_incidence = Weighting.derive_systematic_floor(residuals, incidence, n_bins=10)
    floor_by_unrelated = Weighting.derive_systematic_floor(residuals, unrelated, n_bins=10)

    # Binning by the variable actually correlated with the trend removes much
    # more of it; binning by an unrelated variable leaves most of the trend in
    # r_other, inflating the floor.
    assert floor_by_incidence < floor_by_unrelated


def test_derive_systematic_floor_ignores_nan_pairs():
    rng = np.random.default_rng(5)
    n = 300
    group_by = rng.uniform(0.0, 70.0, size=n)
    residuals = rng.normal(0.0, 0.01, size=n)
    group_by_with_nan = group_by.copy()
    residuals_with_nan = residuals.copy()
    group_by_with_nan[:20] = np.nan
    residuals_with_nan[20:40] = np.nan

    floor_clean = Weighting.derive_systematic_floor(residuals, group_by, n_bins=10)
    floor_with_nan = Weighting.derive_systematic_floor(
        residuals_with_nan, group_by_with_nan, n_bins=10
    )
    # Both should be finite and in the same ballpark (NaNs dropped, not
    # propagated into every bin).
    assert np.isfinite(floor_with_nan)
    assert floor_with_nan == pytest.approx(floor_clean, rel=0.5)


def test_derive_systematic_floor_nan_when_too_few_points_for_requested_bins():
    floor = Weighting.derive_systematic_floor(
        residuals=[0.01, 0.02, 0.03], group_by=[10.0, 20.0, 30.0], n_bins=10
    )
    assert np.isnan(floor)


def test_derive_systematic_floor_raises_on_mismatched_shapes():
    with pytest.raises(ValueError):
        Weighting.derive_systematic_floor(residuals=[0.01, 0.02], group_by=[10.0, 20.0, 30.0])


def test_derive_systematic_floor_is_deterministic_and_nonnegative():
    rng = np.random.default_rng(6)
    group_by = rng.uniform(0.0, 70.0, size=200)
    residuals = rng.normal(0.0, 0.01, size=200)

    floor1 = Weighting.derive_systematic_floor(residuals, group_by)
    floor2 = Weighting.derive_systematic_floor(residuals, group_by)
    assert floor1 == floor2
    assert floor1 >= 0.0

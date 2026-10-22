from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from photometry_etl.etl.geometry_engine import (
    SAMPLE_AXIS_BY_PHASE,
    GeometryEngine,
    calibrate_iof_data,
    check_rays_against_fov_corners,
    compute_pixel_area_km2,
    compute_pixel_rays,
    compute_pixel_solid_angles,
    compute_projected_area_km2,
    exposure_duration_seconds,
    orientation_check,
    sample_axis_for_phase,
)

# DAWN_FC2 (-203120) pinhole parameters, dawn_fc_v02.ti
FC2_CCD_CENTER = (511.5, 511.5)
FC2_IFOV = (0.000093242, 0.000093184)
FC2_SHAPE = (1024, 1024)

F_SOLAR = 898.0

def test_calibrate_iof_physics_conversion() -> None:
    distance_au = 2.36
    # Input that results in I/F < 1.5
    raw = np.array([[0.0, 42.103363]], dtype=np.float32)
    # Expected: (42.103 * pi * 2.36^2) / 898.0 = 0.820379
    expected = np.array([[0.0, 0.820379]], dtype=np.float32)
    
    result = calibrate_iof_data(raw, "UNIT_TEST", distance_au, F_SOLAR)
    np.testing.assert_allclose(result, expected, rtol=1e-5)

def test_calibrate_iof_clipping() -> None:
    distance_au = 1.0
    # Input that results in I/F > 1.5 (e.g., 1.7)
    # Radiance = (1.7 * 898) / pi = 485.6
    raw = np.array([[485.6]], dtype=np.float32)
    
    result = calibrate_iof_data(raw, "UNIT_TEST", distance_au, F_SOLAR)
    # Should be clipped to exactly 1.5
    assert np.max(result) == 1.5

def test_calibrate_iof_fatal_crash() -> None:
    distance_au = 1.0
    # Input that results in I/F > 2.0 (e.g., 2.5)
    # Radiance = (2.5 * 898) / pi = 714.1
    raw = np.array([[714.1]], dtype=np.float32)

    with pytest.raises(RuntimeError, match="FATAL GEOMETRY MISSING"):
        calibrate_iof_data(raw, "UNIT_TEST", distance_au, F_SOLAR)


# ---------------------------------------------------------------------------
# Per-pixel projected area (range_km / pixel_solid_angle_sr / pixel_area_km2)
# ---------------------------------------------------------------------------
# These target the pure functions factored out of GeometryEngine specifically so they
# can be unit-tested without a live SPICE session (no metakernel/instrument kernel needed).


def _spherical_triangle_solid_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Van Oosterom & Strackee formula — independent of the grid-summation method under
    test, used as the ground-truth reference for the self-consistency check below."""
    numerator = abs(np.dot(a, np.cross(b, c)))
    denominator = 1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
    return 2.0 * np.arctan2(numerator, denominator)


def _quad_solid_angle(c0: np.ndarray, c1: np.ndarray, c2: np.ndarray, c3: np.ndarray) -> float:
    return _spherical_triangle_solid_angle(c0, c1, c2) + _spherical_triangle_solid_angle(c0, c2, c3)


def _square_fov_corners(half_angle_rad: float) -> np.ndarray:
    """Synthetic square FOV boresighted on +Z, corners in the order spiceypy.getfov()
    returns them for an ANGLES rectangle: (+X,+Y), (-X,+Y), (-X,-Y), (+X,-Y)."""

    def corner(sx: float, sy: float) -> np.ndarray:
        v = np.array([sx * np.tan(half_angle_rad), sy * np.tan(half_angle_rad), 1.0])
        return v / np.linalg.norm(v)

    return np.array([corner(1, 1), corner(-1, 1), corner(-1, -1), corner(1, -1)])


def _square_pinhole(n: int, half_angle_rad: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """CCD_CENTER/IFOV whose pixel grid exactly spans a square FOV of the given half-angle."""
    ifov = 2.0 * np.tan(half_angle_rad) / n
    return ((n - 1) / 2.0, (n - 1) / 2.0), (ifov, ifov)


def test_compute_pixel_rays_shape_and_unit_norm() -> None:
    rays = compute_pixel_rays((30, 40), (19.5, 14.5), (1e-3, 1e-3))

    assert rays.shape == (30, 40, 3)
    norms = np.linalg.norm(rays, axis=2)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_fc2_pixel_axis_convention_matches_ik() -> None:
    """Pins pixel <-> camera-frame pairing to the IK layout (pixel (0,0) lower-left,
    +X toward increasing sample, +Y toward line 0). (col, row) = (0, 511.5) must look
    toward -X and (511.5, 0) toward +Y; the pre-fix grid had the X sign reversed."""
    rays = compute_pixel_rays(FC2_SHAPE, FC2_CCD_CENTER, FC2_IFOV)
    # (col 0, row 511.5): average the two rows straddling 511.5
    left = rays[511:513, 0].mean(axis=0)
    top = rays[0, 511:513].mean(axis=0)
    centre = rays[511:513, 511:513].reshape(-1, 3).mean(axis=0)

    assert left[0] < 0 and abs(left[1]) < 1e-9
    assert top[1] > 0 and abs(top[0]) < 1e-9
    np.testing.assert_allclose(centre / np.linalg.norm(centre), [0.0, 0.0, 1.0], atol=1e-12)
    # half-width of the FOV in X: 511.5 pixels of IFOV
    np.testing.assert_allclose(np.arctan2(-left[0], left[2]), np.arctan(511.5 * FC2_IFOV[0]), rtol=1e-12)
    # column index increases toward +X, row index toward -Y, everywhere
    assert np.all(np.diff(rays[:, :, 0], axis=1) > 0)
    assert np.all(np.diff(rays[:, :, 1], axis=0) < 0)


def test_rays_agree_with_getfov_corner_order() -> None:
    """The ray grid's outer corners must match getfov()'s boundary vectors by sign; a
    grid built with the old (mirrored) X convention must be rejected."""
    n, half = 200, 0.0477
    centre, ifov = _square_pinhole(n, half)
    rays = compute_pixel_rays((n, n), centre, ifov)
    check_rays_against_fov_corners(rays, _square_fov_corners(half), tol_rad=2.0 * ifov[0])

    with pytest.raises(RuntimeError, match="axis convention mismatch"):
        check_rays_against_fov_corners(rays[:, ::-1], _square_fov_corners(half), tol_rad=2.0 * ifov[0])


@pytest.mark.parametrize(
    ("value", "expected"),
    [("10.000 <millisecond>", 0.010), ("9.0 <ms>", 0.009), ("2.5 <s>", 2.5)],
)
def test_exposure_duration_seconds_reads_units(value: str, expected: float) -> None:
    pvl = pytest.importorskip("pvl")
    q = pvl.loads(f"EXPOSURE_DURATION = {value}\nEND")["EXPOSURE_DURATION"]
    assert float(q) != expected or expected == 2.5  # float() drops the unit
    assert exposure_duration_seconds(q) == pytest.approx(expected)


def test_exposure_duration_seconds_rejects_bare_number() -> None:
    with pytest.raises(ValueError, match="units"):
        exposure_duration_seconds(10.0)


def test_pixel_solid_angle_matches_independent_reference() -> None:
    """Self-consistency check (not a tautology): the finite-difference grid-summation
    method under test is compared against a totally different, independent formula
    (spherical-triangle solid angle via Van Oosterom & Strackee) for the same FOV quad.
    This is what would catch a ray-grid/interpolation bug that per-pixel spot checks miss.

    Tolerance: the grid method has a known, understood O(1/n) boundary discretization
    error (pixel centers are placed exactly at the FOV edges via linspace(0,1,n), and the
    last row/column reuse the adjacent interior step rather than an artificial zero-diff
    edge pad) — empirically n * relative_error ~= 2.0, confirmed constant across
    n=50..1024. At n=200 (this test) that predicts ~1%; comfortably distinguishes a real
    bug (which would be off by an order of magnitude or more, e.g. wrong axis, missing
    normalization, wrong sign) from this well-characterized, shrinking-with-resolution
    approximation artifact.
    """
    half_angle = 0.05  # ~5.7 deg square FOV, order-of-magnitude similar to real FC2
    c0, c1, c2, c3 = _square_fov_corners(half_angle)
    reference = _quad_solid_angle(c0, c1, c2, c3)

    centre, ifov = _square_pinhole(200, half_angle)
    rays = compute_pixel_rays((200, 200), centre, ifov)
    grid_sum = float(compute_pixel_solid_angles(rays).sum())

    assert reference > 0.0
    relative_error = abs(grid_sum - reference) / reference
    assert relative_error < 0.02, (
        f"grid-summed solid angle ({grid_sum:.6e}) should match the independent "
        f"spherical-quad reference ({reference:.6e}) to within ~1-2% at n=200 "
        f"(got {relative_error:.4%})"
    )


def test_pixel_solid_angle_all_positive_and_finite() -> None:
    rays = compute_pixel_rays((50, 60), (29.5, 24.5), (1e-3, 1e-3))
    solid_angles = compute_pixel_solid_angles(rays)

    assert np.all(np.isfinite(solid_angles))
    assert np.all(solid_angles > 0.0)


def test_compute_pixel_area_km2_formula_at_zero_emission() -> None:
    # At emission=0, cos(emission)=1, so area reduces to solid_angle * range^2 exactly.
    solid_angle = np.array([2.0e-8, 3.0e-8])
    range_km = np.array([500.0, 1000.0])
    emission_deg = np.array([0.0, 0.0])

    area = compute_pixel_area_km2(solid_angle, range_km, emission_deg, max_emission_for_area_deg=89.0)

    expected = solid_angle * range_km**2
    np.testing.assert_allclose(area, expected, rtol=1e-12)


def test_compute_pixel_area_km2_nan_above_threshold_not_clipped() -> None:
    solid_angle = np.array([2.0e-8, 2.0e-8, 2.0e-8])
    range_km = np.array([500.0, 500.0, 500.0])
    emission_deg = np.array([10.0, 89.0, 89.5])  # last one exceeds the default threshold

    area = compute_pixel_area_km2(solid_angle, range_km, emission_deg, max_emission_for_area_deg=89.0)

    assert np.isfinite(area[0])
    assert np.isfinite(area[1])  # exactly at threshold is still computable
    assert np.isnan(area[2])  # above threshold: NaN, not a large-but-finite clipped value


def test_compute_pixel_area_km2_nan_inputs_propagate() -> None:
    solid_angle = np.array([2.0e-8, np.nan])
    range_km = np.array([np.nan, 500.0])
    emission_deg = np.array([10.0, 10.0])

    area = compute_pixel_area_km2(solid_angle, range_km, emission_deg, max_emission_for_area_deg=89.0)

    assert np.all(np.isnan(area))


def test_compute_projected_area_km2_formula() -> None:
    solid_angle = np.array([2.0e-8, 3.0e-8])
    range_km = np.array([500.0, 1000.0])

    projected = compute_projected_area_km2(solid_angle, range_km)

    np.testing.assert_allclose(projected, solid_angle * range_km**2, rtol=1e-12)


def test_compute_projected_area_km2_no_singularity_near_90deg_emission() -> None:
    """Unlike compute_pixel_area_km2, projected area has no cos(emission) division, so
    it stays finite (and shrinks smoothly toward zero) as emission -> 90 deg."""
    solid_angle = np.full(3, 2.0e-8)
    range_km = np.full(3, 500.0)

    # projected area doesn't take emission as an argument at all -- confirm that by
    # checking the true-area formula DOES blow up over the same range while the
    # projected-area formula (independent of emission) stays exactly constant.
    projected = compute_projected_area_km2(solid_angle, range_km)
    np.testing.assert_allclose(projected, projected[0], rtol=0.0)
    assert np.all(np.isfinite(projected))

    true_area_near_90 = compute_pixel_area_km2(
        solid_angle, range_km, np.array([89.9, 89.99, 89.999]), max_emission_for_area_deg=90.0
    )
    assert true_area_near_90[-1] > true_area_near_90[0] > projected[0] * 10, (
        "sanity: true surface area should diverge upward as emission -> 90 deg, "
        "unlike the flat, bounded projected area"
    )


def test_compute_projected_area_km2_nan_inputs_propagate() -> None:
    solid_angle = np.array([2.0e-8, np.nan])
    range_km = np.array([np.nan, 500.0])

    projected = compute_projected_area_km2(solid_angle, range_km)

    assert np.all(np.isnan(projected))

# ---------------------------------------------------------------------------
# Per-phase sample axis (orientation is NOT uniform across the mission)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("phase", "axis"), [("survey", "col0_minus_x"), ("hamo", "col0_plus_x"), ("lamo", "col0_plus_x")]
)
def test_sample_axis_for_configured_phases(phase: str, axis: str) -> None:
    assert sample_axis_for_phase(phase) == axis


@pytest.mark.parametrize("phase", ["rc", "approach", "cruise", "unknown", None])
def test_sample_axis_raises_for_unconfigured_phases(phase: str | None) -> None:
    with pytest.raises(ValueError, match="No FC2 sample-axis orientation"):
        sample_axis_for_phase(phase)
    assert phase not in SAMPLE_AXIS_BY_PHASE


def test_strict_phase_lookup_has_no_silent_default() -> None:
    strict = GeometryEngine._strict_phase_from_image_path
    assert strict(Path("/x/calibrated_raw_images/hamo/FC21B0009251_11277030017F1G.IMG")) == "hamo"
    assert strict(Path("/tmp/adhoc/FC21B0009251_11277030017F1G.IMG")) is None


def test_col0_plus_x_grid_is_the_column_mirror_of_col0_minus_x() -> None:
    minus = compute_pixel_rays(FC2_SHAPE, FC2_CCD_CENTER, FC2_IFOV, "col0_minus_x")
    plus = compute_pixel_rays(FC2_SHAPE, FC2_CCD_CENTER, FC2_IFOV, "col0_plus_x")
    np.testing.assert_allclose(plus[:, :, 0], -minus[:, :, 0], atol=0)
    np.testing.assert_allclose(plus[:, :, 1:], minus[:, :, 1:], atol=0)
    assert plus[511, 0, 0] > 0 and minus[511, 0, 0] < 0
    with pytest.raises(ValueError, match="sample_axis"):
        compute_pixel_rays(FC2_SHAPE, FC2_CCD_CENTER, FC2_IFOV, "col0_up")


def test_corner_check_accepts_each_axis_only_for_its_own_grid() -> None:
    n, half = 200, 0.0477
    centre, ifov = _square_pinhole(n, half)
    corners, tol = _square_fov_corners(half), 2.0 * ifov[0]
    for axis in ("col0_minus_x", "col0_plus_x"):
        rays = compute_pixel_rays((n, n), centre, ifov, axis)
        check_rays_against_fov_corners(rays, corners, tol_rad=tol, sample_axis=axis)
        other = "col0_plus_x" if axis == "col0_minus_x" else "col0_minus_x"
        with pytest.raises(RuntimeError, match="axis convention mismatch"):
            check_rays_against_fov_corners(rays, corners, tol_rad=tol, sample_axis=other)


# ---------------------------------------------------------------------------
# Per-image orientation check
# ---------------------------------------------------------------------------


def _synthetic_disk(n: int = 200) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sphere filling part of the frame, Sun from the image-plane +x side: incidence and
    emission grids (NaN off-body) plus a Lommel-Seeliger I/F with an x gradient."""
    yy, xx = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    r2 = xx**2 + (yy * 1.1) ** 2
    on = r2 < 0.8
    nz = np.sqrt(np.clip(1.0 - r2, 0, None))
    sun = np.array([0.6, 0.0, 0.8])
    mu0 = np.where(on, xx * sun[0] + nz * sun[2], np.nan)
    mu = np.where(on, nz, np.nan)
    with np.errstate(invalid="ignore"):
        lit = on & (mu0 > 0)
        inc = np.where(on, np.rad2deg(np.arccos(np.clip(mu0, -1, 1))), np.nan)
        emi = np.where(on, np.rad2deg(np.arccos(np.clip(mu, -1, 1))), np.nan)
        iof = np.where(lit, 0.3 * mu0 / (mu0 + mu), 0.0)
    return iof, inc, emi


def test_orientation_check_passes_matching_and_fails_mirrored_image() -> None:
    iof, inc, emi = _synthetic_disk()
    ok = orientation_check(iof, inc, emi)
    assert ok["verdict"] == "as_is" and ok["status"] == "PASS"
    assert ok["corr_asis"] > ok["corr_xflip"] and ok["iou_asis"] > ok["iou_xflip"]
    bad = orientation_check(np.fliplr(iof), inc, emi)
    assert bad["verdict"] == "x_flipped" and bad["status"] == "FAILED"


def test_orientation_check_flags_featureless_image() -> None:
    _, inc, emi = _synthetic_disk()
    flat = np.where(np.isfinite(inc), 0.1, 0.0)
    out = orientation_check(flat, inc, emi)
    assert out["verdict"] == "undetermined" and out["status"] == "FLAGGED"


# ---------------------------------------------------------------------------
# Real-data regression: orientation of known controls (slow; needs the products)
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]
_V2_ROOT = Path(os.environ.get("GEOMETRY_V2_ROOT", _REPO / "data/geometry/gaskell_dsk256_110825_v2"))
_V1_ROOT = Path(os.environ.get("GEOMETRY_V1_ROOT", _REPO / "data/geometry/gaskell_dsk256_110825"))
_ORIENTATION_CONTROLS = [
    # Survey: correct only with the new mapping (col0_minus_x) -> v2 product.
    ("survey", "FC21B0006203_11236015531F1B", _V2_ROOT),
    # HAMO / LAMO: correct with the old mapping (col0_plus_x), which the v1 products used.
    ("hamo", "FC21B0009251_11277030017F1G", _V1_ROOT),
    ("lamo", "FC21B3021977_12071112848F1C", _V1_ROOT),
]


def _grid(df, col: str) -> np.ndarray:
    g = np.full((1024, 1024), np.nan)
    g[df["pixel_y"].to_numpy(), df["pixel_x"].to_numpy()] = df[col].to_numpy()
    return g


@pytest.mark.slow
@pytest.mark.parametrize(("phase", "image_id", "root"), _ORIENTATION_CONTROLS)
def test_orientation_controls_pass(phase: str, image_id: str, root: Path) -> None:
    """DSK pilots (Oct 2026): corr(iof, LS) +0.451 Survey (new mapping), +0.368 HAMO and
    +0.507 LAMO (old mapping); the opposite mapping gives +0.068 / -0.40 for HAMO / LAMO."""
    import pandas as pd

    path = root / phase / f"{image_id}_geometry.parquet"
    if not path.exists():
        pytest.skip(f"geometry product not available: {path}")
    df = pd.read_parquet(path, columns=["pixel_x", "pixel_y", "iof", "incidence", "emission"])
    out = orientation_check(_grid(df, "iof"), _grid(df, "incidence"), _grid(df, "emission"))
    assert out["status"] == "PASS", out

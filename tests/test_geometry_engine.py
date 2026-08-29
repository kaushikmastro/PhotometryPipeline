from __future__ import annotations

import numpy as np
import pytest

from photometry_etl.etl.geometry_engine import (
    calibrate_iof_data,
    compute_pixel_area_km2,
    compute_pixel_rays,
    compute_pixel_solid_angles,
    compute_projected_area_km2,
)

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
    """Synthetic square FOV boresighted on +Z, matching compute_pixel_rays' expected
    corner winding (c0=top-left, c1=top-right, c2=bottom-right, c3=bottom-left)."""

    def corner(sx: float, sy: float) -> np.ndarray:
        v = np.array([sx * np.tan(half_angle_rad), sy * np.tan(half_angle_rad), 1.0])
        return v / np.linalg.norm(v)

    return np.array([corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)])


def test_compute_pixel_rays_shape_and_unit_norm() -> None:
    bounds = _square_fov_corners(0.05)
    rays = compute_pixel_rays(bounds, (30, 40))

    assert rays.shape == (30, 40, 3)
    norms = np.linalg.norm(rays, axis=2)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)

    # Corners of the ray grid should reproduce the FOV corner vectors exactly.
    np.testing.assert_allclose(rays[0, 0], bounds[0], atol=1e-12)
    np.testing.assert_allclose(rays[0, -1], bounds[1], atol=1e-12)
    np.testing.assert_allclose(rays[-1, -1], bounds[2], atol=1e-12)
    np.testing.assert_allclose(rays[-1, 0], bounds[3], atol=1e-12)


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

    rays = compute_pixel_rays(np.array([c0, c1, c2, c3]), (200, 200))
    grid_sum = float(compute_pixel_solid_angles(rays).sum())

    assert reference > 0.0
    relative_error = abs(grid_sum - reference) / reference
    assert relative_error < 0.02, (
        f"grid-summed solid angle ({grid_sum:.6e}) should match the independent "
        f"spherical-quad reference ({reference:.6e}) to within ~1-2% at n=200 "
        f"(got {relative_error:.4%})"
    )


def test_pixel_solid_angle_all_positive_and_finite() -> None:
    bounds = _square_fov_corners(0.05)
    rays = compute_pixel_rays(bounds, (50, 60))
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
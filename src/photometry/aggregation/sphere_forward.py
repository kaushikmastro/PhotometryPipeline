"""Sphere-based forward integration: predict a model's disk-integrated phase curve
analytically, without any real image pixels.

This is the Li et al. (2013)-style workflow, not the observed-frame aggregation
disk_integrate.py also supports: fit Hapke (or any BasePhotometricModel) to disk-RESOLVED
data (already done -- committed Case 1), then forward-integrate the fitted model over a
synthetic illuminated sphere at each phase angle of interest to predict the disk-
INTEGRATED phase curve. The observed curve to compare against comes from external
literature/ground-based photometry, not from Dawn image frames -- this module produces
only the predicted side.

No new physics: this builds a synthetic (theta, phi) sphere geometry + projected-area
array (same projected_area_km2 convention as geometry_engine/disk_integrate -- see
disk_integrate.py's module docstring for the full reasoning on why projected, not true,
surface area is the correct disk-integration weight) and hands it to the existing,
already-tested integrate_modeled(). This module's only job is constructing that synthetic
geometry; all aggregation math is reused, not reimplemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from photometry.aggregation.disk_integrate import (
    DEFAULT_MAX_EMISSION_DEG,
    DEFAULT_MAX_INCIDENCE_DEG,
    DiskIntegrationResult,
    integrate_modeled,
)
from photometry.core.types import ArrayLike, GeometryBatch
from photometry.models.base import BasePhotometricModel

# Convergence measured directly (see tests/test_sphere_forward.py's convergence test and
# the conversation record): grid-summed sphere integral vs. the exact Lambertian closed
# form Phi_L(alpha) shows clean O(1/n) convergence, n*relative_error roughly constant
# (~0.9 at alpha=30 deg, ~0.65 at alpha=90 deg, worse at low phase / near-terminator
# geometry). At n=300 measured relative error is ~0.2-0.3%; n=500 gets to ~0.13-0.19%.
# 300 is the default: comfortably under 1% for a fast, vectorized grid.
DEFAULT_N_THETA = 300
DEFAULT_N_PHI = 300


@dataclass
class SpherePhaseCurve:
    """Predicted disk-integrated phase curve from sphere_forward_integrate.

    phase_deg / integrated_value / total_area_km2 are parallel arrays, one entry per
    input phase angle. integrated_value has the same units and convention as
    DiskIntegrationResult.integrated_value (sum(reflectance * projected_area_km2)) --
    NaN at any phase angle where the illuminated+visible region is empty (alpha ~ 180 deg).
    """

    phase_deg: np.ndarray
    integrated_value: np.ndarray
    total_area_km2: np.ndarray


def _sphere_geometry_and_projected_area(
    phase_rad: float, radius_km: float, n_theta: int, n_phi: int
) -> tuple[GeometryBatch, np.ndarray]:
    """Synthetic (theta, phi)-grid sphere at a single phase angle: sun and observer
    directions separated by phase_rad, both distant (phase is constant over the body --
    the standard disk-integrated-phase-curve simplification, and exact for a body whose
    angular size as seen from the Sun/observer is negligible, which holds for Vesta at
    any real observing distance).

    Only points with mu0>0 AND mu>0 (illuminated AND visible) get finite incidence/
    emission; everything else is NaN, matching how a real geometry table only carries
    on-body, illuminated, visible rows. projected_area_km2 = mu * R^2 * dOmega -- the
    sky-plane/foreshortened area, i.e. exactly geometry_engine.compute_projected_area_km2's
    quantity (pixel_solid_angle_sr * range_km^2), just for an analytic sphere sample
    instead of a camera pixel grid.
    """
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]
    TH, PH = np.meshgrid(theta, phi, indexing="ij")

    sin_th = np.sin(TH)
    z = np.cos(TH)
    x = sin_th * np.cos(PH)

    s = np.sin(phase_rad / 2.0)
    c = np.cos(phase_rad / 2.0)
    mu0 = x * s + z * c
    mu = -x * s + z * c

    lit_and_visible = (mu0 > 0) & (mu > 0)

    d_omega = sin_th * dtheta * dphi
    projected_area_km2 = np.where(lit_and_visible, mu * radius_km**2 * d_omega, np.nan)
    incidence = np.where(lit_and_visible, np.arccos(np.clip(mu0, -1.0, 1.0)), np.nan)
    emission = np.where(lit_and_visible, np.arccos(np.clip(mu, -1.0, 1.0)), np.nan)
    phase = np.full_like(incidence, phase_rad)

    geometry = GeometryBatch(
        incidence=incidence.reshape(-1),
        emission=emission.reshape(-1),
        phase=phase.reshape(-1),
    )
    return geometry, projected_area_km2.reshape(-1)


def sphere_forward_integrate(
    model: BasePhotometricModel,
    phase_deg: ArrayLike,
    *,
    radius_km: float,
    n_theta: int = DEFAULT_N_THETA,
    n_phi: int = DEFAULT_N_PHI,
    max_incidence_deg: float = DEFAULT_MAX_INCIDENCE_DEG,
    max_emission_deg: float = DEFAULT_MAX_EMISSION_DEG,
) -> SpherePhaseCurve:
    """Predict a model's disk-integrated reflectance at each given phase angle by
    forward-integrating it over a synthetic illuminated sphere -- no real image data
    involved. radius_km is required (not defaulted) since this is generic, body-agnostic
    infrastructure; pass the body's actual mean radius (e.g. 262.0 for Vesta) for a
    physically meaningful result.

    model-agnostic by construction: works identically for any BasePhotometricModel
    (HapkeModel, LommelSeeligerModel, MinnaertModel, LambertianModel, future models) via
    integrate_modeled()'s reflectance() interface -- no model-specific branching here.
    """
    phase_deg_array = np.asarray(phase_deg, dtype=np.float64).reshape(-1)

    integrated_values = np.empty_like(phase_deg_array)
    total_areas = np.empty_like(phase_deg_array)

    for idx, phase in enumerate(phase_deg_array):
        geometry, projected_area_km2 = _sphere_geometry_and_projected_area(
            np.deg2rad(float(phase)), radius_km, n_theta, n_phi
        )
        result: DiskIntegrationResult = integrate_modeled(
            model,
            geometry,
            projected_area_km2,
            max_incidence_deg=max_incidence_deg,
            max_emission_deg=max_emission_deg,
        )
        integrated_values[idx] = result.integrated_value
        total_areas[idx] = result.total_area_km2

    return SpherePhaseCurve(
        phase_deg=phase_deg_array,
        integrated_value=integrated_values,
        total_area_km2=total_areas,
    )

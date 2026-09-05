"""Disk-integrated aggregation of disk-resolved reflectance.

"Disk-integrated" in this codebase means: the area-weighted SUM of reflectance over one
image's illuminated+visible disk at a single epoch -- i.e. the quantity proportional to
total flux received by a distant observer, matching Li et al. (2013) and the standard
disk-integrated-phase-curve convention. This is explicitly NOT the phase-angle-binned
aggregation that ARCHITECTURE_DECISIONS.md calls "disk-integrated" (aggregating many
pixels from many different images/geometries into one phase bin) -- that document is
stale on this point; CLAUDE.md is authoritative.

This module adds no new physics and no separate disk-integrated equation. It is pure
aggregation on top of the existing disk-resolved machinery: per-pixel reflectance comes
from BasePhotometricModel.reflectance() (or directly-observed I/F), unchanged; this
module only sums it, weighted by area, over a selected set of pixels. It is model-agnostic
by construction -- it never branches on which BasePhotometricModel subclass it was given.

WHICH AREA COLUMN TO USE -- this is a genuine physics decision, not a detail:
Use `projected_area_km2` (geometry_engine.compute_projected_area_km2:
pixel_solid_angle_sr * range_km^2). Do NOT use `pixel_area_km2` (the TRUE,
unforeshortened surface-area column, .../cos(emission)) to weight a disk integral.

Why: "disk-integrated brightness" is defined as total flux received by a distant
observer, which integrates reflectance over the PROJECTED (sky-plane) area each surface
element presents to the observer -- i.e. the foreshortened footprint, not its true
physical area on the body. Weighting by true surface area instead over-weights
foreshortened limb geometry (where true area diverges as emission -> 90 deg, exactly
backwards from its vanishing contribution to observed flux) and does not reproduce
established results.

This was verified concretely, not asserted: for a Lambertian sphere (I/F = albedo*mu0/pi),
weighting by projected area (mu0*mu*dOmega) reproduces the classical Lambert phase
function Phi_L(alpha) = (1/pi)*(sin(alpha) + (pi-alpha)*cos(alpha)) to numerical precision;
weighting by true surface area (mu0*dOmega, no mu factor) does not -- e.g. at alpha=90 deg
the two integrals differ by ~57%. See tests/test_disk_integrate.py::
test_lambertian_sphere_matches_phase_function_closed_form (the load-bearing correctness
test for this whole module) and its companion
test_true_surface_area_weighting_does_not_match_phase_function (which pins the negative
result in the test suite so a future refactor can't silently swap the weight back).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from photometry.core.types import ArrayLike, GeometryBatch
from photometry.models.base import BasePhotometricModel

DEFAULT_MAX_INCIDENCE_DEG = 90.0
DEFAULT_MAX_EMISSION_DEG = 90.0


@dataclass
class DiskIntegrationResult:
    """Result of aggregating disk-resolved reflectance over one image's illuminated+
    visible disk.

    integrated_value: sum(reflectance_i * projected_area_km2_i) over the included
        pixels. Units: [reflectance] * km^2. NaN if no pixels were included -- empty
        disk selection, everything excluded by the incidence/emission cutoffs, or every
        selected pixel had a NaN area -- rather than 0.0, since 0.0 would misleadingly
        read as "measured zero flux" instead of "no usable data."
    total_area_km2: sum of projected_area_km2_i over the included pixels. Dividing
        integrated_value by this gives the area-weighted MEAN reflectance instead of the
        sum, if that's ever wanted; NaN under the same no-data condition as above.
    n_pixels_included: pixels that passed the disk selection AND had a finite area.
    n_pixels_excluded_disk_selection: pixels excluded by the incidence/emission cutoffs
        (including any with non-finite incidence/emission).
    n_pixels_excluded_nan_area: pixels that passed the disk selection but had a NaN
        projected area. projected_area_km2 has no emission-driven singularity (unlike
        pixel_area_km2), so for real geometry_engine output this should be rare or zero;
        kept so NaN handling is explicit and auditable rather than silently summed away.
    """

    integrated_value: float
    total_area_km2: float
    n_pixels_included: int
    n_pixels_excluded_disk_selection: int
    n_pixels_excluded_nan_area: int


def _to_numpy(value: ArrayLike) -> np.ndarray:
    """Aggregation is a post-hoc reporting/validation step, not a differentiable
    training-time op, so this module works in plain numpy regardless of which backend
    produced its inputs."""
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


def _disk_selection_mask(
    geometry: GeometryBatch,
    max_incidence_deg: float,
    max_emission_deg: float,
) -> np.ndarray:
    """Illuminated+visible pixel mask: finite incidence/emission (on-body; geometry
    tables already drop off-body pixels upstream, but a caller-assembled GeometryBatch
    might not have) and within the given cutoffs. GeometryBatch angles are radians
    (its documented contract); cutoffs are accepted in degrees to match every other
    domain cutoff in this codebase (e.g. the committed Case 1 i<50,e<50 methodology)."""
    incidence_deg = np.rad2deg(_to_numpy(geometry.incidence))
    emission_deg = np.rad2deg(_to_numpy(geometry.emission))

    return (
        np.isfinite(incidence_deg)
        & np.isfinite(emission_deg)
        & (incidence_deg <= max_incidence_deg)
        & (emission_deg <= max_emission_deg)
    )


def _weighted_sum(
    reflectance: np.ndarray,
    geometry: GeometryBatch,
    projected_area_km2: ArrayLike,
    max_incidence_deg: float,
    max_emission_deg: float,
) -> DiskIntegrationResult:
    reflectance = np.asarray(reflectance, dtype=np.float64).reshape(-1)
    areas = _to_numpy(projected_area_km2).astype(np.float64).reshape(-1)

    if reflectance.shape != areas.shape:
        raise ValueError(
            f"reflectance shape {reflectance.shape} does not match areas shape {areas.shape}"
        )

    disk_mask = _disk_selection_mask(geometry, max_incidence_deg, max_emission_deg)
    if disk_mask.shape != reflectance.shape:
        raise ValueError(
            f"geometry shape {disk_mask.shape} does not match reflectance shape {reflectance.shape}"
        )

    n_excluded_disk = int(np.count_nonzero(~disk_mask))

    area_finite = np.isfinite(areas)
    n_excluded_nan_area = int(np.count_nonzero(disk_mask & ~area_finite))

    included = disk_mask & area_finite
    n_included = int(np.count_nonzero(included))

    if n_included == 0:
        integrated_value = float("nan")
        total_area_km2 = float("nan")
    else:
        integrated_value = float(np.sum(reflectance[included] * areas[included]))
        total_area_km2 = float(np.sum(areas[included]))

    return DiskIntegrationResult(
        integrated_value=integrated_value,
        total_area_km2=total_area_km2,
        n_pixels_included=n_included,
        n_pixels_excluded_disk_selection=n_excluded_disk,
        n_pixels_excluded_nan_area=n_excluded_nan_area,
    )


def integrate_observed(
    observed_reflectance: ArrayLike,
    geometry: GeometryBatch,
    projected_area_km2: ArrayLike,
    *,
    max_incidence_deg: float = DEFAULT_MAX_INCIDENCE_DEG,
    max_emission_deg: float = DEFAULT_MAX_EMISSION_DEG,
) -> DiskIntegrationResult:
    """Area-weighted sum of OBSERVED I/F over one image's illuminated+visible pixels.

    max_incidence_deg / max_emission_deg default to 90 deg -- the literal physical
    definition of "illuminated" (sun above local horizon) and "visible" (not
    self-occluded), not the tighter science-domain fit cutoffs (e.g. i<50,e<50) used
    elsewhere in this pipeline for a specific fit methodology. Override per use case.
    """
    return _weighted_sum(
        _to_numpy(observed_reflectance),
        geometry,
        projected_area_km2,
        max_incidence_deg,
        max_emission_deg,
    )


def integrate_modeled(
    model: BasePhotometricModel,
    geometry: GeometryBatch,
    projected_area_km2: ArrayLike,
    *,
    max_incidence_deg: float = DEFAULT_MAX_INCIDENCE_DEG,
    max_emission_deg: float = DEFAULT_MAX_EMISSION_DEG,
) -> DiskIntegrationResult:
    """Area-weighted sum of MODEL-PREDICTED reflectance over one image's illuminated+
    visible pixels. Model-agnostic: works identically for any BasePhotometricModel
    subclass (LommelSeeligerModel, MinnaertModel, LambertianModel, HapkeModel, and any
    future model, e.g. AkimovModel) via the shared reflectance() interface -- this
    function never branches on model type. model.reflectance(geometry) is evaluated over
    every pixel in geometry (its own internal physics already zeroes out unilluminated/
    invisible geometry), then the same disk-selection + area-weighted-sum logic as
    integrate_observed is applied on top.
    """
    predicted = _to_numpy(model.reflectance(geometry))
    return _weighted_sum(
        predicted, geometry, projected_area_km2, max_incidence_deg, max_emission_deg
    )

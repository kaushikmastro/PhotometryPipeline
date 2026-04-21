"""Lommel-Seeliger photometric model."""

from __future__ import annotations

import numpy as np


def lommel_seeliger_reflectance(
    incidence: np.ndarray | float,
    emission: np.ndarray | float,
    phase: np.ndarray | float | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    """Compute Lommel-Seeliger reflectance.

    Parameters
    ----------
    incidence, emission, phase
        Angles in radians. `phase` is accepted for API consistency with the
        wider model zoo but is not used by the pure Lommel-Seeliger form.
    scale
        Multiplicative scale factor (equivalent to an albedo-like amplitude).

    Returns
    -------
    np.ndarray
        Modeled reflectance with the same broadcast shape as the inputs.
    """
    i = np.asarray(incidence, dtype=np.float64)
    e = np.asarray(emission, dtype=np.float64)
    mu0 = np.cos(i)
    mu = np.cos(e)
    denominator = mu0 + mu

    with np.errstate(divide="ignore", invalid="ignore"):
        reflectance = scale * (2.0 * mu0 / denominator)

    reflectance = np.where((mu0 <= 0.0) | (mu <= 0.0), 0.0, reflectance)
    reflectance = np.where(np.isfinite(reflectance), reflectance, 0.0)
    return reflectance

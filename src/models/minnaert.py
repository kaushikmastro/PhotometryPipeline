"""Minnaert photometric model."""

from __future__ import annotations

import numpy as np


def minnaert_reflectance(
    incidence: np.ndarray | float,
    emission: np.ndarray | float,
    phase: np.ndarray | float | None = None,
    k: float = 1.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Compute Minnaert reflectance.

    Parameters
    ----------
    incidence, emission, phase
        Angles in radians. `phase` is accepted for zoo-level API consistency
        but is not used by the core Minnaert law.
    k
        Minnaert limb-darkening exponent.
    scale
        Multiplicative amplitude.

    Returns
    -------
    np.ndarray
        Modeled reflectance with the same broadcast shape as the inputs.
    """
    i = np.asarray(incidence, dtype=np.float64)
    e = np.asarray(emission, dtype=np.float64)
    mu0 = np.cos(i)
    mu = np.cos(e)

    with np.errstate(divide="ignore", invalid="ignore"):
        reflectance = scale * np.power(mu0, k) * np.power(mu, k - 1.0)

    reflectance = np.where((mu0 <= 0.0) | (mu <= 0.0), 0.0, reflectance)
    reflectance = np.where(np.isfinite(reflectance), reflectance, 0.0)
    return reflectance

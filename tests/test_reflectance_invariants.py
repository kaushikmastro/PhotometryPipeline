"""Physical invariants every reflectance model must satisfy, independent of any one
implementation's formulas (added after the Oct 2026 roughness audit: the old Hapke
roughness passed a hand-written reference that encoded the same mistake, but failed
reciprocity by a factor of 25).

- Reciprocity (Helmholtz): the BRDF, here I/F / mu0, is unchanged when incidence and
  emission are swapped at fixed phase angle.
- Smooth limit: Hapke roughness with theta_bar -> 0 reduces to the smooth model.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from photometry.core.types import GeometryBatch  # noqa: E402
from photometry.models.baselines import (  # noqa: E402
    LambertianModel,
    LommelSeeligerModel,
    LunarLambertModel,
    MinnaertModel,
)
from photometry.models.hapke import HapkeModel  # noqa: E402

SHOE = {"B0": 1.03, "h": 0.04}


def _hapke(theta_bar: float | None, isotropic_h: bool = False) -> HapkeModel:
    rough = theta_bar is not None
    return HapkeModel(
        enable_shoe=True, enable_roughness=rough, isotropic_h=isotropic_h, fixed_parameters=dict(SHOE),
        parameters={"w": 0.5, "g": -0.3, **SHOE, **({"theta_bar": theta_bar} if rough else {})},
    )


MODELS: dict[str, Callable[[], object]] = {
    "lambertian": lambda: LambertianModel(parameters={"albedo": 0.3}),
    "lommel_seeliger": lambda: LommelSeeligerModel(parameters={"w": 0.5}),
    "minnaert": lambda: MinnaertModel(parameters={"albedo": 0.3, "k": 0.7}),
    "lunar_lambert_phase_dep": lambda: LunarLambertModel(parameters={"albedo": 0.3}),
    "lunar_lambert_scalar": lambda: LunarLambertModel(
        parameters={"albedo": 0.3, "c_L": 0.4}, metadata={"phase_dependent_c_L": False}
    ),
    "hapke_smooth": lambda: _hapke(None),
    "hapke_rough_10": lambda: _hapke(10.0),
    "hapke_rough_30": lambda: _hapke(30.0),
    "hapke_rough_45_isotropic_h": lambda: _hapke(45.0, isotropic_h=True),
}


def _swap_pairs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(i, e, alpha) in degrees with i != e; alpha from an azimuth grid so it is physical."""
    i, e, psi = (x.ravel() for x in np.meshgrid(np.arange(2.0, 85.0, 7.0), np.arange(2.0, 85.0, 7.0),
                                                 np.arange(0.0, 181.0, 30.0), indexing="ij"))
    keep = i != e
    i, e, psi = i[keep], e[keep], psi[keep]
    ri, re, rp = np.deg2rad(i), np.deg2rad(e), np.deg2rad(psi)
    alpha = np.rad2deg(np.arccos(np.clip(np.cos(ri) * np.cos(re) + np.sin(ri) * np.sin(re) * np.cos(rp), -1, 1)))
    return i, e, alpha


def _iof(model, i, e, a) -> np.ndarray:
    return np.asarray(model._reflectance_numpy(GeometryBatch(
        incidence=np.deg2rad(i), emission=np.deg2rad(e), phase=np.deg2rad(a))), dtype=np.float64)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_reciprocity_of_brdf(name: str) -> None:
    model = MODELS[name]()
    i, e, a = _swap_pairs()
    brdf_ie = _iof(model, i, e, a) / np.cos(np.deg2rad(i))
    brdf_ei = _iof(model, e, i, a) / np.cos(np.deg2rad(e))
    ok = (brdf_ie > 0) & (brdf_ei > 0)
    assert ok.mean() > 0.9
    np.testing.assert_allclose(brdf_ie[ok], brdf_ei[ok], rtol=1e-9)


@pytest.mark.parametrize("isotropic_h", [False, True])
def test_hapke_roughness_smooth_limit(isotropic_h: bool) -> None:
    i, e, a = _swap_pairs()
    np.testing.assert_allclose(
        _iof(_hapke(1e-3, isotropic_h), i, e, a), _iof(_hapke(None, isotropic_h), i, e, a), rtol=1e-6, atol=1e-12
    )

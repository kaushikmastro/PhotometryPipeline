"""Macroscopic roughness must follow Hapke (2012) ch. 12 (eqs. 12.45-12.55).

Pinned values come from an independent scalar implementation of those equations (agrees
with a separate vectorised implementation to < 1e-7 over 4212 geometries x 6 theta_bar).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from photometry.core.types import GeometryBatch  # noqa: E402
from photometry.models.hapke import HapkeModel  # noqa: E402

W, G, B0, H = 0.5, -0.3, 1.03, 0.04


def _iof(inc, emi, phase, theta_bar: float) -> np.ndarray:
    rough = theta_bar > 0
    model = HapkeModel(
        enable_shoe=True, enable_roughness=rough, fixed_parameters={"B0": B0, "h": H},
        parameters={"w": W, "g": G, "B0": B0, "h": H, **({"theta_bar": theta_bar} if rough else {})},
    )
    geo = GeometryBatch(
        incidence=np.deg2rad(np.atleast_1d(np.asarray(inc, float))),
        emission=np.deg2rad(np.atleast_1d(np.asarray(emi, float))),
        phase=np.deg2rad(np.atleast_1d(np.asarray(phase, float))),
    )
    return model._reflectance_numpy(geo)


def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physical (i, e, alpha) triples from an (i, e, psi) grid."""
    i, e, psi = (x.ravel() for x in np.meshgrid(np.arange(0, 85, 5.0), np.arange(0, 85, 5.0),
                                                 np.arange(0, 181, 15.0), indexing="ij"))
    ri, re, rp = np.deg2rad(i), np.deg2rad(e), np.deg2rad(psi)
    alpha = np.rad2deg(np.arccos(np.clip(np.cos(ri) * np.cos(re) + np.sin(ri) * np.sin(re) * np.cos(rp), -1, 1)))
    return i, e, alpha


@pytest.mark.parametrize(
    ("geometry", "expected"),
    [
        ((60, 30, 60), (0.0839726, 0.0790672, 0.0710325)),
        ((30, 60, 60), (0.1454448, 0.1369484, 0.1230319)),
        ((60, 60, 90), (0.0760890, 0.0593529, 0.0415589)),
        ((45, 20, 50), (0.1163903, 0.1140478, 0.1071257)),
    ],
)
def test_roughness_matches_hapke_2012_reference(geometry, expected) -> None:
    got = [float(_iof(*geometry, t)[0]) for t in (10.0, 20.0, 30.0)]
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_roughness_obeys_reciprocity() -> None:
    """Hapke's rough-surface BRDF (I/F / mu0) is symmetric under i <-> e at fixed alpha;
    the two branches of eqs. 12.46-12.54 exist precisely to guarantee this."""
    i, e, a = _grid()
    sel = (i != e) & (i < 85) & (e < 85)
    i, e, a = i[sel], e[sel], a[sel]
    for theta in (10.0, 30.0, 45.0):
        brdf_ie = _iof(i, e, a, theta) / np.cos(np.deg2rad(i))
        brdf_ei = _iof(e, i, a, theta) / np.cos(np.deg2rad(e))
        np.testing.assert_allclose(brdf_ie, brdf_ei, rtol=1e-9)


def test_roughness_darkens_at_large_phase_monotonically() -> None:
    i, e, a = _grid()
    sel = a >= 60
    series = np.array([_iof(i[sel], e[sel], a[sel], t) for t in (0.0, 10.0, 20.0, 30.0, 45.0)])
    assert np.all(np.diff(series, axis=0) <= 1e-12)
    assert np.all(series[-1] < series[0] * 0.95)    # and not marginally: >5% darker at 45 deg


def test_roughness_never_brightens_away_from_opposition() -> None:
    """Rough <= smooth everywhere except near alpha = 0 at grazing i = e, where the
    Hapke model genuinely brightens (no visible shadows, facets tilted toward the viewer)."""
    i, e, a = _grid()
    sel = a >= 5
    smooth = _iof(i[sel], e[sel], a[sel], 0.0)
    for theta in (10.0, 30.0, 60.0):
        assert np.all(_iof(i[sel], e[sel], a[sel], theta) <= smooth * (1 + 1e-9))


def test_small_theta_recovers_smooth_model() -> None:
    i, e, a = _grid()
    np.testing.assert_allclose(_iof(i, e, a, 1e-3), _iof(i, e, a, 0.0), rtol=1e-6, atol=1e-12)


def test_branches_meet_continuously_at_i_equals_e() -> None:
    for theta in (10.0, 30.0):
        for (ang, alpha) in ((40.0, 30.0), (60.0, 80.0)):
            below = _iof(ang - 1e-6, ang, alpha, theta)[0]
            above = _iof(ang + 1e-6, ang, alpha, theta)[0]
            assert abs(below - above) / above < 1e-5

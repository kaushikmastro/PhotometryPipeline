from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from photometry.core.types import GeometryBatch  # noqa: E402
from photometry.models.hapke import HapkeModel  # noqa: E402


def _hand_calculated_hapke_if(
    w: float,
    g: float,
    theta_bar: float,
    B0: float,
    h: float,
    inc: float,
    emi: float,
    phase: float,
) -> np.ndarray:
    """Manual IMSA Hapke I/F — simplified Hapke (1984) roughness as used by Li et al. (2013).

    Inputs are in degrees.
    """
    eps = 1e-12

    i = np.deg2rad(np.asarray(inc, dtype=np.float64))
    e = np.deg2rad(np.asarray(emi, dtype=np.float64))
    alpha = np.deg2rad(np.asarray(phase, dtype=np.float64))
    theta = np.deg2rad(np.asarray(theta_bar, dtype=np.float64))

    mu0 = np.cos(i)
    mu = np.cos(e)

    # Single-term HG phase function
    cos_alpha = np.cos(alpha)
    P = (1.0 - g**2) / np.power(1.0 + 2.0 * g * cos_alpha + g**2, 1.5)

    # SHOE
    B_sh = B0 / (1.0 + (1.0 / h) * np.tan(alpha / 2.0))

    # H-function (Hapke 2002 approximation)
    gamma = np.sqrt(np.clip(1.0 - w, 0.0, 1.0))
    r0 = (1.0 - gamma) / (1.0 + gamma)

    def H(x: np.ndarray) -> np.ndarray:
        x_safe = np.clip(x, eps, None)
        bracket = r0 + ((1.0 - 2.0 * r0 * x_safe) / 2.0) * np.log((1.0 + x_safe) / x_safe)
        return 1.0 / (1.0 - w * x_safe * bracket)

    H_mu0 = H(np.asarray(mu0, dtype=np.float64))
    H_mu  = H(np.asarray(mu,  dtype=np.float64))

    # Macroscopic roughness — simplified Hapke (1984) form (Li et al. 2013)
    cos_psi = (cos_alpha - mu0 * mu) / np.clip(np.sin(i) * np.sin(e), eps, None)
    psi = np.arccos(np.clip(cos_psi, -1.0, 1.0))
    f_psi = np.exp(-2.0 * np.tan(psi / 2.0))

    tan_theta = np.tan(theta)
    chi = 1.0 / np.sqrt(1.0 + np.pi * tan_theta**2)

    def E1(x_tan: np.ndarray) -> np.ndarray:
        x_safe = np.clip(np.asarray(x_tan, dtype=np.float64), eps, None)
        return np.exp(-2.0 / (np.pi * tan_theta * x_safe))

    def E2(x_tan: np.ndarray) -> np.ndarray:
        x_safe = np.clip(np.asarray(x_tan, dtype=np.float64), eps, None)
        return np.exp(-1.0 / (np.pi * (tan_theta**2) * (x_safe**2)))

    tan_i = np.clip(np.tan(i), eps, None)
    tan_e = np.clip(np.tan(e), eps, None)

    E1_i, E1_e = E1(tan_i), E1(tan_e)
    E2_i, E2_e = E2(tan_i), E2(tan_e)

    sin_i = np.sin(i)
    sin_e = np.sin(e)
    sin2_psi = np.sin(psi / 2.0)**2

    # CASE 1: i ≤ e
    mu0_eff_1 = chi * (mu0 + sin_i * tan_theta * (E2_e + sin2_psi * E2_i))
    mu_eff_1  = chi * (mu  + sin_e * tan_theta * (E2_i - sin2_psi * E2_i))
    shadow_denom_1 = 1.0 - f_psi * E1_i - (1.0 - f_psi) * E1_e

    # CASE 2: i > e
    mu0_eff_2 = chi * (mu0 + sin_i * tan_theta * (E2_e - sin2_psi * E2_e))
    mu_eff_2  = chi * (mu  + sin_e * tan_theta * (E2_i + sin2_psi * E2_e))
    shadow_denom_2 = 1.0 - f_psi * E1_e - (1.0 - f_psi) * E1_i

    ile = i <= e
    mu0_eff = np.where(ile, mu0_eff_1, mu0_eff_2)
    mu_eff  = np.where(ile, mu_eff_1, mu_eff_2)
    shadow_denom = np.where(ile, shadow_denom_1, shadow_denom_2)

    # S = (μₑ/μ)·(μ₀/μ₀ₑ)·χ/Denom
    S = (mu_eff / np.clip(mu, eps, None)) * (mu0 / np.clip(mu0_eff, eps, None))
    S *= chi / np.clip(shadow_denom, eps, None)

    iof = (w / 4.0) * (mu0 / np.clip(mu0 + mu, eps, None)) * (((1.0 + B_sh) * P) + H_mu0 * H_mu - 1.0) * S
    return np.asarray(iof, dtype=np.float64)


def test_hapke_imsa_baseline() -> None:
    # Li et al. (2013) baseline parameters, Vesta F1 clear filter.
    w = 0.23
    g = -0.30
    theta_bar = 18.0
    B0 = 1.6
    h = 0.06

    inc, emi, phase = 30.0, 20.0, 40.0

    expected = _hand_calculated_hapke_if(w, g, theta_bar, B0, h, inc, emi, phase)

    model = HapkeModel(
        enable_shoe=True,
        enable_roughness=True,
        parameters={"w": w, "g": g, "theta_bar": theta_bar, "B0": B0, "h": h},
    )
    geometry = GeometryBatch(
        incidence=np.array([np.deg2rad(inc)], dtype=np.float64),
        emission=np.array([np.deg2rad(emi)], dtype=np.float64),
        phase=np.array([np.deg2rad(phase)], dtype=np.float64),
    )

    actual = model._reflectance_numpy(geometry)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)


@given(
    geometry_deg=st.tuples(
        st.floats(min_value=0.0, max_value=89.9, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=89.9, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=180.0, allow_nan=False, allow_infinity=False),
    ).filter(lambda g: abs(g[0] - g[1]) <= g[2] <= (g[0] + g[1])),
    w=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    g=st.floats(min_value=-0.99, max_value=0.99, allow_nan=False, allow_infinity=False),
    theta_bar=st.floats(min_value=0.0, max_value=60.0, allow_nan=False, allow_infinity=False),
    B0=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    h=st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_hapke_numerical_stability(
    geometry_deg: tuple[float, float, float],
    w: float, g: float, theta_bar: float, B0: float, h: float,
) -> None:
    """Ensure IMSA evaluation remains finite across valid geometry and parameter space."""
    inc_deg, emi_deg, phase_deg = geometry_deg
    model = HapkeModel(
        enable_shoe=True,
        enable_roughness=True,
        parameters={"w": w, "g": g, "theta_bar": theta_bar, "B0": B0, "h": h},
    )
    geometry = GeometryBatch(
        incidence=np.array([np.deg2rad(inc_deg)], dtype=np.float64),
        emission=np.array([np.deg2rad(emi_deg)], dtype=np.float64),
        phase=np.array([np.deg2rad(phase_deg)], dtype=np.float64),
    )
    assert np.isfinite(model._reflectance_numpy(geometry)).all()


def test_isotropic_h_formula() -> None:
    """Verify the isotropic H-function H(x) = (1+2x)/(1+2γx) is wired correctly.

    Three checks:
    1. At w=0 both forms collapse to H(x)=1 for all x (γ=1 when w=0).
    2. At w>0 the two forms give numerically distinct values (not the same code path).
    3. The isotropic form matches its closed-form at a known (w, i, e, α).
    """
    eps = 1e-12

    geom_simple = GeometryBatch(
        incidence=np.array([np.deg2rad(30.0)]),
        emission=np.array([np.deg2rad(20.0)]),
        phase=np.array([np.deg2rad(40.0)]),
    )

    # --- check 1: w=0 → both H-functions = 1, model output identical ----------------
    for iso in (False, True):
        m = HapkeModel(
            enable_shoe=False, enable_roughness=False, isotropic_h=iso,
            parameters={"w": 0.0, "g": 0.0},
        )
        out = m._reflectance_numpy(geom_simple)
        assert np.isfinite(out).all(), "non-finite at w=0"

    m0_default = HapkeModel(enable_shoe=False, enable_roughness=False, isotropic_h=False,
                            parameters={"w": 0.0, "g": 0.0})
    m0_iso     = HapkeModel(enable_shoe=False, enable_roughness=False, isotropic_h=True,
                            parameters={"w": 0.0, "g": 0.0})
    np.testing.assert_allclose(
        m0_default._reflectance_numpy(geom_simple),
        m0_iso._reflectance_numpy(geom_simple),
        rtol=0, atol=1e-12,
        err_msg="At w=0 both H-forms must give identical I/F",
    )

    # --- check 2: w>0 → the two forms give different I/F values ---------------------
    w_test = 0.5
    m_default = HapkeModel(enable_shoe=False, enable_roughness=False, isotropic_h=False,
                           parameters={"w": w_test, "g": -0.3})
    m_iso     = HapkeModel(enable_shoe=False, enable_roughness=False, isotropic_h=True,
                           parameters={"w": w_test, "g": -0.3})
    iof_default = float(m_default._reflectance_numpy(geom_simple)[0])
    iof_iso     = float(m_iso._reflectance_numpy(geom_simple)[0])
    assert abs(iof_default - iof_iso) > 1e-6, (
        f"Isotropic and Hapke-2002 forms should differ at w={w_test}, "
        f"got default={iof_default:.8f} iso={iof_iso:.8f}"
    )

    # --- check 3: isotropic I/F matches hand-computed closed form -------------------
    # At (i=30, e=20, α=40), no roughness, no SHOE, w=0.23, g=-0.30:
    #   μ₀ = cos(30°), μ = cos(20°), γ = √(1-0.23)
    #   P(40°) = (1-(-0.3)²)/(1+2(-0.3)cos40°+(-0.3)²)^1.5
    #   H(μ₀) = (1+2μ₀)/(1+2γμ₀),  H(μ) = (1+2μ)/(1+2γμ)
    #   I/F = (w/4)*(μ₀/(μ₀+μ))*(P + H(μ₀)*H(μ) - 1)
    w, g = 0.23, -0.30
    mu0 = np.cos(np.deg2rad(30.0))
    mu  = np.cos(np.deg2rad(20.0))
    ca  = np.cos(np.deg2rad(40.0))
    gamma_hand = np.sqrt(1.0 - w)
    P   = (1 - g**2) / (1 + 2*g*ca + g**2)**1.5
    H_mu0 = (1 + 2*mu0) / (1 + 2*gamma_hand*mu0)
    H_mu  = (1 + 2*mu)  / (1 + 2*gamma_hand*mu)
    iof_hand = (w / 4.0) * (mu0 / (mu0 + mu)) * (P + H_mu0 * H_mu - 1.0)

    m_hand = HapkeModel(enable_shoe=False, enable_roughness=False, isotropic_h=True,
                        parameters={"w": w, "g": g})
    iof_model = float(m_hand._reflectance_numpy(geom_simple)[0])
    np.testing.assert_allclose(iof_model, iof_hand, rtol=0, atol=1e-10,
                               err_msg="Isotropic H-function I/F does not match hand calculation")


def test_hapke_architecture_toggles() -> None:
    """Test that the unified Hapke model correctly toggles parameter bounds and names."""
    model_full = HapkeModel(enable_shoe=True, enable_roughness=True)
    assert set(model_full.parameter_names()) == {"w", "g", "theta_bar", "B0", "h"}

    model_no_shoe = HapkeModel(enable_shoe=False, enable_roughness=True)
    assert set(model_no_shoe.parameter_names()) == {"w", "g", "theta_bar"}
    assert "B0" not in model_no_shoe.parameter_bounds()

    model_basic = HapkeModel(enable_shoe=False, enable_roughness=False)
    assert set(model_basic.parameter_names()) == {"w", "g"}
    assert "theta_bar" not in model_basic.parameter_bounds()

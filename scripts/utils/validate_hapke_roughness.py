from __future__ import annotations

import math

import numpy as np


EPS = 1e-12
THETA_BAR_DEG = 18.0


def _prepare_angles(i_deg: float, e_deg: float, alpha_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    incidence = np.deg2rad(np.asarray([i_deg], dtype=np.float64))
    emission = np.deg2rad(np.asarray([e_deg], dtype=np.float64))
    phase = np.deg2rad(np.asarray([alpha_deg], dtype=np.float64))
    return incidence, emission, phase


def calc_roughness_old(i: float, e: float, alpha: float, theta_bar: float) -> np.ndarray:
    incidence, emission, phase = _prepare_angles(i, e, alpha)

    mu0 = np.cos(incidence)
    mu = np.cos(emission)
    cos_alpha = np.cos(phase)

    theta = np.deg2rad(float(theta_bar))
    tan_theta = np.tan(theta)
    chi = 1.0 / np.sqrt(1.0 + np.pi * tan_theta**2)

    cos_psi = (cos_alpha - mu0 * mu) / np.clip(np.sin(incidence) * np.sin(emission), EPS, None)
    psi = np.arccos(np.clip(cos_psi, -1.0, 1.0))
    f_psi = np.exp(-2.0 * np.tan(psi / 2.0))

    def _e1(x_tan: np.ndarray) -> np.ndarray:
        x_safe = np.clip(np.asarray(x_tan, dtype=np.float64), EPS, None)
        return np.exp(-2.0 / (np.pi * tan_theta * x_safe))

    def _e2(x_tan: np.ndarray) -> np.ndarray:
        x_safe = np.clip(np.asarray(x_tan, dtype=np.float64), EPS, None)
        return np.exp(-1.0 / (np.pi * (tan_theta**2) * (x_safe**2)))

    tan_i = np.tan(incidence)
    tan_e = np.tan(emission)
    e1_i, e1_e = _e1(tan_i), _e1(tan_e)
    e2_i, e2_e = _e2(tan_i), _e2(tan_e)
    sin_i = np.sin(incidence)
    sin_e = np.sin(emission)
    sin2_psi = np.sin(psi / 2.0) ** 2

    mu0_eff = np.empty_like(mu0)
    mu_eff = np.empty_like(mu)
    shadow_denom = np.empty_like(mu0)
    ile = incidence <= emission

    mu0_eff[ile] = chi * (mu0[ile] + sin_i[ile] * tan_theta * (e2_e[ile] + sin2_psi[ile] * e2_i[ile]))
    mu_eff[ile] = chi * (mu[ile] + sin_e[ile] * tan_theta * (e2_i[ile] - sin2_psi[ile] * e2_i[ile]))
    shadow_denom[ile] = 1.0 - f_psi[ile] * e1_i[ile] - (1.0 - f_psi[ile]) * e1_e[ile]

    mu0_eff[~ile] = chi * (mu0[~ile] + sin_i[~ile] * tan_theta * (e2_e[~ile] - sin2_psi[~ile] * e2_e[~ile]))
    mu_eff[~ile] = chi * (mu[~ile] + sin_e[~ile] * tan_theta * (e2_i[~ile] + sin2_psi[~ile] * e2_e[~ile]))
    shadow_denom[~ile] = 1.0 - f_psi[~ile] * e1_e[~ile] - (1.0 - f_psi[~ile]) * e1_i[~ile]

    roughness = (mu_eff / np.clip(mu, EPS, None)) * (mu0 / np.clip(mu0_eff, EPS, None))
    roughness *= chi / np.clip(shadow_denom, EPS, None)
    return roughness


def calc_roughness_new(i: float, e: float, alpha: float, theta_bar: float) -> np.ndarray:
    incidence, emission, phase = _prepare_angles(i, e, alpha)

    mu0 = np.cos(incidence)
    mu = np.cos(emission)
    cos_alpha = np.cos(phase)

    theta = np.deg2rad(float(theta_bar))
    tan_theta = np.tan(theta)
    chi = 1.0 / np.sqrt(1.0 + np.pi * tan_theta**2)

    cos_psi = (cos_alpha - mu0 * mu) / np.clip(np.sin(incidence) * np.sin(emission), EPS, None)
    psi = np.arccos(np.clip(cos_psi, -1.0, 1.0))
    f_psi = np.exp(-2.0 * np.tan(psi / 2.0))

    def _e1(x_tan: np.ndarray) -> np.ndarray:
        x_safe = np.clip(np.asarray(x_tan, dtype=np.float64), EPS, None)
        return np.exp(-2.0 / (np.pi * tan_theta * x_safe))

    def _e2(x_tan: np.ndarray) -> np.ndarray:
        x_safe = np.clip(np.asarray(x_tan, dtype=np.float64), EPS, None)
        return np.exp(-1.0 / (np.pi * (tan_theta**2) * (x_safe**2)))

    tan_i = np.tan(incidence)
    tan_e = np.tan(emission)
    e1_i, e1_e = _e1(tan_i), _e1(tan_e)
    e2_i, e2_e = _e2(tan_i), _e2(tan_e)
    sin_i = np.sin(incidence)
    sin_e = np.sin(emission)
    sin2_psi = np.sin(psi / 2.0) ** 2

    mu0_eff = np.empty_like(mu0)
    mu_eff = np.empty_like(mu)
    shadow_denom = np.empty_like(mu0)
    ile = incidence <= emission

    denom_mu0_ile = np.clip(2.0 - e1_e[ile] - (psi[ile] / np.pi) * e1_i[ile], EPS, None)
    denom_mu_ile = np.clip(2.0 - e1_i[ile] - (psi[ile] / np.pi) * e1_i[ile], EPS, None)
    num_mu0_ile = cos_psi[ile] * e2_e[ile] + sin2_psi[ile] * e2_i[ile]
    num_mu_ile = e2_i[ile] - sin2_psi[ile] * e2_i[ile]
    mu0_eff[ile] = chi * (mu0[ile] + sin_i[ile] * tan_theta * (num_mu0_ile / denom_mu0_ile))
    mu_eff[ile] = chi * (mu[ile] + sin_e[ile] * tan_theta * (num_mu_ile / denom_mu_ile))
    shadow_denom[ile] = 1.0 - f_psi[ile] * e1_i[ile] - (1.0 - f_psi[ile]) * e1_e[ile]

    denom_mu0_igt = np.clip(2.0 - e1_e[~ile] - (psi[~ile] / np.pi) * e1_e[~ile], EPS, None)
    denom_mu_igt = np.clip(2.0 - e1_i[~ile] - (psi[~ile] / np.pi) * e1_e[~ile], EPS, None)
    num_mu0_igt = e2_e[~ile] - sin2_psi[~ile] * e2_e[~ile]
    num_mu_igt = cos_psi[~ile] * e2_i[~ile] + sin2_psi[~ile] * e2_e[~ile]
    mu0_eff[~ile] = chi * (mu0[~ile] + sin_i[~ile] * tan_theta * (num_mu0_igt / denom_mu0_igt))
    mu_eff[~ile] = chi * (mu[~ile] + sin_e[~ile] * tan_theta * (num_mu_igt / denom_mu_igt))
    shadow_denom[~ile] = 1.0 - f_psi[~ile] * e1_e[~ile] - (1.0 - f_psi[~ile]) * e1_i[~ile]

    roughness = (mu_eff / np.clip(mu, EPS, None)) * (mu0 / np.clip(mu0_eff, EPS, None))
    roughness *= chi / np.clip(shadow_denom, EPS, None)
    return roughness


def report(label: str, i: float, e: float, alpha: float) -> None:
    old = calc_roughness_old(i, e, alpha, THETA_BAR_DEG)
    new = calc_roughness_new(i, e, alpha, THETA_BAR_DEG)

    old_value = float(old[0])
    new_value = float(new[0])
    delta = new_value - old_value
    pct = (delta / old_value * 100.0) if old_value != 0.0 else math.nan

    print(f"{label}: i={i:.2f}°, e={e:.2f}°, alpha={alpha:.2f}°")
    print(f"  old S = {old_value:.12f}")
    print(f"  new S = {new_value:.12f}")
    print(f"  delta = {delta:.12f} ({pct:.3f}%)")
    print(f"  old finite = {np.isfinite(old).all()}, new finite = {np.isfinite(new).all()}")


def main() -> None:
    tests = [
        ("standard mapping", 43.0, 21.0, 29.0),
        ("low phase edge case", 12.0, 12.0, 2.0),
        ("high phase", 78.0, 18.0, 120.0),
    ]

    print(f"theta_bar = {THETA_BAR_DEG:.1f} deg")
    print("Testing old vs proposed Hapke roughness multiplier S\n")

    for label, i, e, alpha in tests:
        report(label, i, e, alpha)
        print()

    boundary_cases = [
        (0.0, 0.0, 0.0),
        (90.0, 90.0, 90.0),
        (30.0, 30.0, 0.0),
    ]
    print("Boundary stability check")
    for i, e, alpha in boundary_cases:
        old = calc_roughness_old(i, e, alpha, THETA_BAR_DEG)
        new = calc_roughness_new(i, e, alpha, THETA_BAR_DEG)
        print(
            f"  i={i:.1f}, e={e:.1f}, alpha={alpha:.1f} -> "
            f"old finite={np.isfinite(old).all()}, new finite={np.isfinite(new).all()}, "
            f"old={old[0]:.12f}, new={new[0]:.12f}"
        )


if __name__ == "__main__":
    main()
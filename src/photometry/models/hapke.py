from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from photometry.core.types import Backend, GeometryBatch, ParameterPrior
from photometry.models.base import BasePhotometricModel



@dataclass
class HapkeModel(BasePhotometricModel):
    """
    Unified Configurable Hapke Model.
    Toggle physical components via enable_shoe and enable_roughness.
    """
    model_name: str = "hapke_full"
    enable_shoe: bool = False
    enable_roughness: bool = False
    fixed_parameters: dict = field(default_factory=dict)
    # isotropic_h=True → H(x) = (1+2x)/(1+2γx), the IMSA form used by Li et al. 2013.
    # isotropic_h=False (default) → Hapke (2002) approximation (current historical default).
    isotropic_h: bool = False

    def __post_init__(self) -> None:

        """Initialize all 5 parameters dynamically based on enabled physics."""

        self.parameters.setdefault("w", 0.491) # from Li et al. 2013, table 2 case3
        self.parameters.setdefault("g", -0.223) # from Li et al. 2013, table 2 case3

        if self.enable_roughness:
            self.parameters.setdefault("theta_bar", 18.3) # from Li et al. 2013, table 2 case3
        if self.enable_shoe:
            self.parameters.setdefault("B0", 1.03) # Helfenstein & Veverka 1989
            self.parameters.setdefault("h", 0.04)  # Helfenstein & Veverka 1989

        self.parameters.update(self.fixed_parameters)  # Override with any user-specified fixed parameters

    def parameter_names(self) -> list[str]:

        """Return active parameter names based on enabled physics components."""
        names = ["w", "g"]
        if self.enable_roughness:
            names.append("theta_bar")
        if self.enable_shoe:
            names.extend(["B0", "h"])
        return [n for n in names if n not in self.fixed_parameters]

    def parameter_bounds(self) -> dict[str, tuple[float, float]]:

        bounds = {"w": (0.0, 1.0), "g": (-1.0, 1.0)}
        if self.enable_roughness:
            bounds["theta_bar"] = (0.0, 60.0)
        if self.enable_shoe:
            bounds["B0"] = (0.0, 2.0)
            bounds["h"] = (0.001, 1.0)
        return bounds

    def parameter_priors(self) -> dict[str, ParameterPrior]:

        priors = {
            "w": ParameterPrior(prior_type="uniform", lower_bound=0.0, upper_bound=1.0),
            "g": ParameterPrior(prior_type="uniform", lower_bound=-1.0, upper_bound=1.0),
        }
        if self.enable_roughness:
            priors["theta_bar"] = ParameterPrior(prior_type="uniform", lower_bound=0.0, upper_bound=60.0)
        if self.enable_shoe:
            priors["B0"] = ParameterPrior(prior_type="uniform", lower_bound=0.0, upper_bound=2.0)
            priors["h"] = ParameterPrior(prior_type="uniform", lower_bound=0.001, upper_bound=1.0)
        return priors

    def _reflectance_numpy(self, geometry: GeometryBatch) -> np.ndarray:

        incidence = np.asarray(geometry.incidence, dtype=np.float64)
        emission = np.asarray(geometry.emission, dtype=np.float64)
        phase = np.asarray(geometry.phase, dtype=np.float64)

        w = float(self.parameters["w"])
        g = float(self.parameters["g"])
        eps = 1e-12

        mu0 = np.cos(incidence)
        mu = np.cos(emission)
        cos_alpha = np.cos(phase)

        # Single-term Henyey-Greenstein phase function (Li et al. 2013, Eq. 3)
        phase_function = (1.0 - g**2) / np.power(1.0 + 2.0 * g * cos_alpha + g**2, 1.5)

        # Chandrasekhar H-function — two forms selectable via isotropic_h flag.
        gamma = np.sqrt(np.clip(1.0 - w, 0.0, 1.0))

        if self.isotropic_h:
            # Isotropic multiple scattering approximation (Hapke 1984, Eq. 8.57 in Hapke 2012).
            # H(x) = (1+2x) / (1+2γx)  where γ = √(1-w)
            # Li et al. (2013) used this form for Vesta disk-resolved fits and found the
            # Hapke (2002) form gave worse residuals for their dataset.
            def _h_function(x: np.ndarray) -> np.ndarray:
                return (1.0 + 2.0 * x) / (1.0 + 2.0 * gamma * x)
        else:
            # Hapke (2002) approximation — more accurate but gave worse residuals
            # for Vesta disk-resolved fits (Li et al. 2013).
            r0 = (1.0 - gamma) / (1.0 + gamma)
            def _h_function(x: np.ndarray) -> np.ndarray:
                x_safe = np.clip(x, eps, None)
                bracket = r0 + ((1.0 - 2.0 * r0 * x_safe) / 2.0) * np.log((1.0 + x_safe) / x_safe)
                return 1.0 / (1.0 - w * x_safe * bracket)

        h_mu0 = _h_function(mu0)
        h_mu = _h_function(mu)

        # ---------------------------------------------------------
        # SHOE: B_SH(α) = B0 / (1 + (1/h)·tan(α/2))
        # Li et al. 2013, Eq. 2 — SHOE only, no CBOE
        # ---------------------------------------------------------

        if self.enable_shoe:
            b0 = float(self.parameters["B0"])
            h_val = float(self.parameters["h"])
            b_sh = b0 / (1.0 + (1.0 / h_val) * np.tan(phase / 2.0))
        else:
            b_sh = 0.0

        # ---------------------------------------------------------
        # Macroscopic Roughness S(i, e, α, θ̄)
        # Simplified Hapke (1984) form as used by Li et al. (2013).
        # This is the standard disk-resolved implementation — no denominator
        # inside the effective-cosine expressions.
        # ---------------------------------------------------------

        if self.enable_roughness:

            theta_bar_deg = float(self.parameters["theta_bar"])
            theta_bar_deg = np.clip(theta_bar_deg, 1e-4, 60.0)
            theta_rad = np.deg2rad(theta_bar_deg)

            # Step A: azimuth angle ψ and base quantities
            cos_psi = (cos_alpha - mu0 * mu) / np.clip(np.sin(incidence) * np.sin(emission), eps, None)
            psi = np.arccos(np.clip(cos_psi, -1.0, 1.0))
            f_psi = np.exp(-2.0 * np.tan(psi / 2.0))

            tan_theta = np.tan(theta_rad)
            chi = 1.0 / np.sqrt(1.0 + np.pi * tan_theta**2)

            # Step B: shadowing integrals E1, E2 (Hapke 1984)
            def _e1(x_tan: np.ndarray) -> np.ndarray:
                x_safe = np.clip(x_tan, eps, None)
                return np.exp(-2.0 / (np.pi * tan_theta * x_safe))

            def _e2(x_tan: np.ndarray) -> np.ndarray:
                x_safe = np.clip(x_tan, eps, None)
                return np.exp(-1.0 / (np.pi * (tan_theta**2) * (x_safe**2)))

            tan_i = np.clip(np.tan(incidence), eps, None)
            tan_e = np.clip(np.tan(emission), eps, None)

            e1_i, e1_e = _e1(tan_i), _e1(tan_e)
            e2_i, e2_e = _e2(tan_i), _e2(tan_e)

            sin_i = np.sin(incidence)
            sin_e = np.sin(emission)
            sin2_psi = np.sin(psi / 2.0)**2  # sin²(ψ/2)

            mu0_eff = np.empty_like(mu0)
            mu_eff = np.empty_like(mu)
            shadow_denom = np.empty_like(mu0)

            ile = incidence <= emission

            # CASE 1: i ≤ e  (Hapke 1984, simplified form used by Li et al. 2013)
            # μ₀ₑ = χ[μ₀ + sin(i)·tan(θ̄)·(E₂(e) + sin²(ψ/2)·E₂(i))]
            # μₑ  = χ[μ  + sin(e)·tan(θ̄)·(E₂(i) − sin²(ψ/2)·E₂(i))]
            # Denom = 1 − f(ψ)·E₁(i) − (1−f(ψ))·E₁(e)

            mu0_eff[ile] = chi * (mu0[ile] + sin_i[ile] * tan_theta * (e2_e[ile] + sin2_psi[ile] * e2_i[ile]))
            mu_eff[ile]  = chi * (mu[ile]  + sin_e[ile] * tan_theta * (e2_i[ile] - sin2_psi[ile] * e2_i[ile]))
            shadow_denom[ile] = 1.0 - f_psi[ile] * e1_i[ile] - (1.0 - f_psi[ile]) * e1_e[ile]

            # CASE 2: i > e  (Hapke 1984, simplified form used by Li et al. 2013)
            # μ₀ₑ = χ[μ₀ + sin(i)·tan(θ̄)·(E₂(e) − sin²(ψ/2)·E₂(e))]
            # μₑ  = χ[μ  + sin(e)·tan(θ̄)·(E₂(i) + sin²(ψ/2)·E₂(e))]
            # Denom = 1 − f(ψ)·E₁(e) − (1−f(ψ))·E₁(i)

            mu0_eff[~ile] = chi * (mu0[~ile] + sin_i[~ile] * tan_theta * (e2_e[~ile] - sin2_psi[~ile] * e2_e[~ile]))
            mu_eff[~ile]  = chi * (mu[~ile]  + sin_e[~ile] * tan_theta * (e2_i[~ile] + sin2_psi[~ile] * e2_e[~ile]))
            shadow_denom[~ile] = 1.0 - f_psi[~ile] * e1_e[~ile] - (1.0 - f_psi[~ile]) * e1_i[~ile]

            # Step D: S = (μₑ/μ)·(μ₀/μ₀ₑ)·χ/Denom
            
            roughness = (mu_eff / np.clip(mu, eps, None)) * (mu0 / np.clip(mu0_eff, eps, None))
            roughness *= chi / np.clip(shadow_denom, eps, None)

        else:
            roughness = 1.0

        # ---------------------------------------------------------
        # Final IMSA equation (Li et al. 2013, Eq. 1)
        # I/F = (ϖ₀/4)·(μ₀/(μ₀+μ))·[(1+B_SH)·P + H(μ₀)·H(μ) − 1]·S
        # ---------------------------------------------------------

        base = (w / 4.0) * (mu0 / np.clip(mu0 + mu, eps, None))
        scatter = (1.0 + b_sh) * phase_function + h_mu0 * h_mu - 1.0
        iof = base * scatter * roughness

        valid = (mu0 > 0.0) & (mu > 0.0)
        return np.where(valid & np.isfinite(iof), iof, 0.0)

    def _reflectance_torch(self, geometry: GeometryBatch) -> Any:
        raise NotImplementedError("Torch backend not yet implemented.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HapkeModel:
        return cls(
            enable_shoe=payload.get("enable_shoe", True),
            enable_roughness=payload.get("enable_roughness", True),
            isotropic_h=payload.get("isotropic_h", False),
            parameters=dict(payload.get("parameters", {})),
            metadata=dict(payload.get("metadata", {})),
            backend=Backend(payload.get("backend", Backend.AUTO.value))
        )

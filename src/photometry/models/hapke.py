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
        # Macroscopic roughness, Hapke (2012) ch. 12 (= Hapke 1984), eqs. 12.45-12.55.
        # The rough-surface I/F is the smooth expression evaluated at the effective cosines
        # mu0e, mue, multiplied by the shadowing function S:
        #   I/F = (w/4) * mu0e/(mu0e+mue) * [(1+B)P + H(mu0e)H(mue) - 1] * S(i, e, psi)
        # psi is the azimuth between the incidence and emission planes,
        #   cos psi = (cos alpha - cos i cos e) / (sin i sin e).
        # ---------------------------------------------------------
        if self.enable_roughness:
            theta_rad = np.deg2rad(float(np.clip(float(self.parameters["theta_bar"]), 1e-4, 60.0)))
            tan_t = np.tan(theta_rad)
            cot_t = 1.0 / tan_t
            chi = 1.0 / np.sqrt(1.0 + np.pi * tan_t**2)                                  # 12.45a

            sin_i, sin_e = np.sin(incidence), np.sin(emission)
            sin_ie = sin_i * sin_e
            with np.errstate(invalid="ignore", divide="ignore"):
                cos_psi = np.where(sin_ie > eps, (cos_alpha - mu0 * mu) / np.where(sin_ie > eps, sin_ie, 1.0), 1.0)
            cos_psi = np.clip(cos_psi, -1.0, 1.0)
            psi = np.arccos(cos_psi)
            sin2_half_psi = np.sin(psi / 2.0) ** 2
            with np.errstate(over="ignore"):
                f_psi = np.where(psi < np.pi, np.exp(-2.0 * np.tan(psi / 2.0)), 0.0)          # 12.53

            # cot(x) -> inf at x = 0 (normal incidence/emission), where E1 = E2 = exp(-inf) = 0:
            # the overflow is the intended limit, so it is silenced rather than clipped.
            def _cot(x: np.ndarray) -> np.ndarray:
                with np.errstate(divide="ignore", over="ignore"):
                    return np.where(x > 0.0, 1.0 / np.tan(np.where(x > 0.0, x, 1.0)), np.inf)

            def _e1(x: np.ndarray) -> np.ndarray:                                          # 12.45b
                with np.errstate(over="ignore"):
                    return np.exp(-(2.0 / np.pi) * cot_t * _cot(x))

            def _e2(x: np.ndarray) -> np.ndarray:                                          # 12.45c
                with np.errstate(over="ignore"):
                    return np.exp(-(1.0 / np.pi) * cot_t**2 * _cot(x) ** 2)

            def _eta(x: np.ndarray) -> np.ndarray:                                         # 12.48
                return chi * (np.cos(x) + np.sin(x) * tan_t * _e2(x) / (2.0 - _e1(x)))

            e1_i, e1_e, e2_i, e2_e = _e1(incidence), _e1(emission), _e2(incidence), _e2(emission)
            eta_i, eta_e = _eta(incidence), _eta(emission)
            ile = incidence <= emission

            # i <= e (12.46, 12.47, 12.52) and e <= i (12.50, 12.51, 12.54)
            d_ile = 2.0 - e1_e - (psi / np.pi) * e1_i
            d_eli = 2.0 - e1_i - (psi / np.pi) * e1_e
            mu0_eff = np.where(
                ile,
                chi * (mu0 + sin_i * tan_t * (cos_psi * e2_e + sin2_half_psi * e2_i) / d_ile),
                chi * (mu0 + sin_i * tan_t * (e2_i - sin2_half_psi * e2_e) / d_eli),
            )
            mu_eff = np.where(
                ile,
                chi * (mu + sin_e * tan_t * (e2_e - sin2_half_psi * e2_i) / d_ile),
                chi * (mu + sin_e * tan_t * (cos_psi * e2_i + sin2_half_psi * e2_e) / d_eli),
            )
            shadow = (mu_eff / eta_e) * (mu0 / eta_i) * chi / np.where(
                ile,
                1.0 - f_psi + f_psi * chi * mu0 / eta_i,
                1.0 - f_psi + f_psi * chi * mu / eta_e,
            )
        else:
            mu0_eff, mu_eff, shadow = mu0, mu, 1.0

        # ---------------------------------------------------------
        # IMSA (Li et al. 2013, Eq. 1; Hapke 2012 eq. 12.55 with roughness)
        # ---------------------------------------------------------
        base = (w / 4.0) * (mu0_eff / np.clip(mu0_eff + mu_eff, eps, None))
        scatter = (1.0 + b_sh) * phase_function + _h_function(mu0_eff) * _h_function(mu_eff) - 1.0
        iof = base * scatter * shadow

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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from photometry.core.types import Backend, GeometryBatch, ParameterPrior
from photometry.models.base import BasePhotometricModel, ModelRegistry


try:
    import torch  # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover - optional dependency at skeleton stage
    torch = None


@ModelRegistry.register
@dataclass
class LambertianModel(BasePhotometricModel):
    """
    Trivial baseline photometric model.

    This implementation is intentionally simple and fully vectorized so it can
    act as the first working model in the new four-layer architecture.
    """

    model_name: str = "lambertian"

    def __post_init__(self) -> None:
        """Normalize default parameter state for the baseline model."""
        self.parameters.setdefault("albedo", 1.0)

    def parameter_names(self) -> list[str]:
        """Return the ordered free parameters for the Lambertian model."""
        return ["albedo"]

    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Return parameter bounds for the Lambertian model."""
        return {"albedo": (0.0, 1.0)}

    def parameter_priors(self) -> dict[str, ParameterPrior]:
        """Return parameter priors for the Lambertian model."""
        return {
            "albedo": ParameterPrior(
                prior_type="uniform",
                lower_bound=0.0,
                upper_bound=1.0,
            )
        }

    def _albedo(self) -> float:
        """Return the active albedo parameter."""
        return float(self.parameters.get("albedo", 1.0))

    def _reflectance_numpy(self, geometry: GeometryBatch) -> np.ndarray:
        """NumPy reflectance kernel for the Lambertian baseline."""
        incidence = np.asarray(geometry.incidence, dtype=np.float64)
        mu0 = np.cos(incidence)
        return self._albedo() * np.clip(mu0, 0.0, None) / np.pi

    def _reflectance_torch(self, geometry: GeometryBatch) -> Any:
        """PyTorch reflectance kernel for the Lambertian baseline."""
        if torch is None:
            raise RuntimeError("PyTorch is not available in this environment.")
        incidence = geometry.incidence if isinstance(geometry.incidence, torch.Tensor) else torch.as_tensor(geometry.incidence)
        mu0 = torch.cos(incidence.to(dtype=torch.float64))
        return torch.clamp(mu0, min=0.0) * (self._albedo() / np.pi)

    def predict_with_uncertainty(self, geometry: GeometryBatch, parameter_samples: Any, sample_axis: int = 0, return_summary: bool = True):
        """Baseline uncertainty propagation via batched Lambertian evaluation."""
        raise NotImplementedError("LambertianModel uncertainty propagation is not implemented yet.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LambertianModel":
        """Reconstruct a LambertianModel from serialized state."""
        backend_value = payload.get("backend", Backend.AUTO.value)
        backend = Backend(backend_value)
        parameters = dict(payload.get("parameters", {}))
        metadata = dict(payload.get("metadata", {}))
        return cls(parameters=parameters, metadata=metadata, backend=backend)



@ModelRegistry.register
@dataclass
class LommelSeeligerModel(BasePhotometricModel):
    """Lommel-Seeliger baseline photometric model."""

    model_name: str = "lommel_seeliger"

    def __post_init__(self) -> None:
        self.parameters.setdefault("w", 1.0)

    def parameter_names(self) -> list[str]:
        return ["w"]

    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return {"w": (0.0, 1.0)}

    def parameter_priors(self) -> dict[str, ParameterPrior]:
        return {
            "w": ParameterPrior(
                prior_type="uniform",
                lower_bound=0.0,
                upper_bound=1.0,
            )
        }

    def _w(self) -> float:
        return float(self.parameters.get("w", 1.0))

    def _reflectance_numpy(self, geometry: GeometryBatch) -> np.ndarray:
        incidence = np.asarray(geometry.incidence, dtype=np.float64)
        emission = np.asarray(geometry.emission, dtype=np.float64)
        mu0 = np.clip(np.cos(incidence), 0.0, None)
        mu = np.clip(np.cos(emission), 0.0, None)
        denominator = mu0 + mu
        reflectance = (self._w() / 4.0) * (mu0 / (denominator + 1e-10))
        return np.where(denominator > 1e-12, reflectance, 0.0)

    def _reflectance_torch(self, geometry: GeometryBatch) -> Any:
        if torch is None:
            raise RuntimeError("PyTorch is not available in this environment.")
        incidence = geometry.incidence if isinstance(geometry.incidence, torch.Tensor) else torch.as_tensor(geometry.incidence)
        emission = geometry.emission if isinstance(geometry.emission, torch.Tensor) else torch.as_tensor(geometry.emission)
        incidence = incidence.to(dtype=torch.float64)
        emission = emission.to(dtype=torch.float64)
        mu0 = torch.clamp(torch.cos(incidence), min=0.0)
        mu = torch.clamp(torch.cos(emission), min=0.0)
        denominator = mu0 + mu
        reflectance = (self._w() / 4.0) * (mu0 / (denominator + 1e-10))
        return torch.where(denominator > 1e-12, reflectance, torch.zeros_like(reflectance))

    def predict_with_uncertainty(self, geometry: GeometryBatch, parameter_samples: Any, sample_axis: int = 0, return_summary: bool = True):
        raise NotImplementedError("LommelSeeligerModel uncertainty propagation is not implemented yet.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LommelSeeligerModel":
        backend_value = payload.get("backend", Backend.AUTO.value)
        backend = Backend(backend_value)
        parameters = dict(payload.get("parameters", {}))
        metadata = dict(payload.get("metadata", {}))
        return cls(parameters=parameters, metadata=metadata, backend=backend)
@ModelRegistry.register
@dataclass
class MinnaertModel(BasePhotometricModel):
    """Empirical Minnaert disk-function model."""

    model_name: str = "minnaert"

    def __post_init__(self) -> None:
        self.parameters.setdefault("albedo", 1.0)
        self.parameters.setdefault("k", 0.5)

    def parameter_names(self) -> list[str]:
        return ["albedo", "k"]

    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        return {"albedo": (0.0, 10.0), "k": (0.0, 2.0)}

    def parameter_priors(self) -> dict[str, ParameterPrior]:
        return {
            "albedo": ParameterPrior(prior_type="uniform", lower_bound=0.0, upper_bound=10.0),
            "k": ParameterPrior(prior_type="uniform", lower_bound=0.0, upper_bound=2.0),
        }

    def _albedo(self) -> float:
        return float(self.parameters.get("albedo", 1.0))

    def _k(self) -> float:
        return float(self.parameters.get("k", 0.5))

    def _reflectance_numpy(self, geometry: GeometryBatch) -> np.ndarray:
        incidence = np.asarray(geometry.incidence, dtype=np.float64)
        emission = np.asarray(geometry.emission, dtype=np.float64)

        mu0 = np.clip(np.cos(incidence), 0.0, None)
        mu = np.clip(np.cos(emission), 0.0, None)

        k = self._k()
        mu_safe = mu + 1e-10
        return self._albedo() * (mu0**k) * (mu_safe ** (k - 1.0))

    def _reflectance_torch(self, geometry: GeometryBatch) -> Any:
        if torch is None:
            raise RuntimeError("PyTorch is not available in this environment.")

        incidence = geometry.incidence if isinstance(geometry.incidence, torch.Tensor) else torch.as_tensor(geometry.incidence)
        emission = geometry.emission if isinstance(geometry.emission, torch.Tensor) else torch.as_tensor(geometry.emission)

        incidence = incidence.to(dtype=torch.float64)
        emission = emission.to(dtype=torch.float64)

        mu0 = torch.clamp(torch.cos(incidence), min=0.0)
        mu = torch.clamp(torch.cos(emission), min=0.0)

        k = self._k()
        mu_safe = mu + 1e-10
        return self._albedo() * torch.pow(mu0, k) * torch.pow(mu_safe, (k - 1.0))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MinnaertModel":
        backend_value = payload.get("backend", Backend.AUTO.value)
        backend = Backend(backend_value)
        parameters = dict(payload.get("parameters", {}))
        metadata = dict(payload.get("metadata", {}))
        return cls(parameters=parameters, metadata=metadata, backend=backend)


@ModelRegistry.register
@dataclass
class LunarLambertModel(BasePhotometricModel):
    """
    Lunar-Lambert disk function implementation following Schröder et al. (2013), Equation 6.

    Reference: Schröder et al. 2013, Eq. 6 — Lunar-Lambert blending of Lommel-Seeliger and Lambertian terms.

    The blending parameter `c_L` may be used as a scalar model parameter. By default the model
    uses the published Vesta phase-dependent values via the linear relation
    c_L(phi) = 0.830 - 0.00722 * phi_deg when `phase_dependent_c_L` metadata flag is True.
    Set `phase_dependent_c_L` to False to treat `c_L` as a scalar free parameter.
    """

    model_name: str = "lunar_lambert"

    def __post_init__(self) -> None:
        # primary amplitude (unitless scaling similar to Lambertian albedo)
        self.parameters.setdefault("albedo", 1.0)
        # scalar c_L default (will be overridden per-phase if phase_dependent_c_L is True)
        self.parameters.setdefault("c_L", 0.830)
        # by default use phase-dependent published c_L(phi)
        self.metadata.setdefault("phase_dependent_c_L", True)

    def parameter_names(self) -> list[str]:
        use_phase_dep = bool(self.metadata.get("phase_dependent_c_L", True))
        if use_phase_dep:
            return ["albedo"]
        return ["albedo", "c_L"]

    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        use_phase_dep = bool(self.metadata.get("phase_dependent_c_L", True))
        bounds = {"albedo": (0.0, 10.0)}
        if not use_phase_dep:
            bounds["c_L"] = (0.0, 1.0)
        return bounds

    def parameter_priors(self) -> dict[str, ParameterPrior]:
        use_phase_dep = bool(self.metadata.get("phase_dependent_c_L", True))
        priors: dict[str, ParameterPrior] = {
            "albedo": ParameterPrior(prior_type="uniform", lower_bound=0.0, upper_bound=10.0),
        }
        if not use_phase_dep:
            priors["c_L"] = ParameterPrior(prior_type="uniform", lower_bound=0.0, upper_bound=1.0)
        return priors

    def _albedo(self) -> float:
        return float(self.parameters.get("albedo", 1.0))

    def _c_L_scalar(self) -> float:
        return float(self.parameters.get("c_L", 0.830))

    def _compute_c_L_array(self, phase: Any) -> Any:
        """
        Compute c_L per-sample. If metadata['phase_dependent_c_L'] is True, compute
        c_L = 0.830 - 0.00722 * phi_deg where phi is the phase angle in degrees.
        Otherwise return a scalar filled array with the scalar c_L parameter.
        """
        use_phase_dep = bool(self.metadata.get("phase_dependent_c_L", True))
        if use_phase_dep:
            # phase input expected in radians in GeometryBatch; convert to degrees
            # handle both numpy arrays and torch tensors via backend resolution
            try:
                import torch as _torch  # type: ignore
            except Exception:
                _torch = None

            if _torch is not None and isinstance(phase, _torch.Tensor):
                phi_deg = phase * (180.0 / np.pi)
                return 0.830 - 0.00722 * phi_deg
            else:
                phi = np.asarray(phase, dtype=float)
                phi_deg = phi * (180.0 / np.pi)
                return 0.830 - 0.00722 * phi_deg

        # scalar fallback
        scalar = self._c_L_scalar()
        try:
            import torch as _torch  # type: ignore
        except Exception:
            _torch = None
        if _torch is not None and isinstance(phase, _torch.Tensor):
            return _torch.full_like(phase, fill_value=float(scalar), dtype=_torch.float64)
        return np.full_like(np.asarray(phase, dtype=float), fill_value=float(scalar))

    def _reflectance_numpy(self, geometry: GeometryBatch) -> np.ndarray:
        incidence = np.asarray(geometry.incidence, dtype=np.float64)
        emission = np.asarray(geometry.emission, dtype=np.float64)
        phase = np.asarray(geometry.phase, dtype=np.float64)

        mu0 = np.clip(np.cos(incidence), 0.0, None)
        mu = np.clip(np.cos(emission), 0.0, None)

        # Lunar-Lambert disk function: D = c_L * (mu0/(mu0+mu)) + (1 - c_L) * mu0
        c_L_arr = self._compute_c_L_array(phase)
        denom = mu0 + mu + 1e-10
        ls_term = (2.0*mu0) / denom
        D = c_L_arr * ls_term + (1.0 - c_L_arr) * mu0

        # Simple multiplicative albedo scaling
        R = self._albedo() * D
        return np.where(denom > 1e-12, R, 0.0)

    def _reflectance_torch(self, geometry: GeometryBatch) -> Any:
        if torch is None:
            raise RuntimeError("PyTorch is not available in this environment.")
        incidence = geometry.incidence if isinstance(geometry.incidence, torch.Tensor) else torch.as_tensor(geometry.incidence)
        emission = geometry.emission if isinstance(geometry.emission, torch.Tensor) else torch.as_tensor(geometry.emission)
        phase = geometry.phase if isinstance(geometry.phase, torch.Tensor) else torch.as_tensor(geometry.phase)

        incidence = incidence.to(dtype=torch.float64)
        emission = emission.to(dtype=torch.float64)
        phase = phase.to(dtype=torch.float64)

        mu0 = torch.clamp(torch.cos(incidence), min=0.0)
        mu = torch.clamp(torch.cos(emission), min=0.0)
        denom = mu0 + mu

        c_L_arr = self._compute_c_L_array(phase)
        ls_term = (2.0*mu0) / (denom + 1e-10)
        D = c_L_arr * ls_term + (1.0 - c_L_arr) * mu0

        R = self._albedo() * D
        return torch.where(denom > 1e-12, R, torch.zeros_like(R))

    def predict_with_uncertainty(self, geometry: GeometryBatch, parameter_samples: Any, sample_axis: int = 0, return_summary: bool = True):
        raise NotImplementedError("LunarLambertModel uncertainty propagation is not implemented yet.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LunarLambertModel":
        backend_value = payload.get("backend", Backend.AUTO.value)
        backend = Backend(backend_value)
        parameters = dict(payload.get("parameters", {}))
        metadata = dict(payload.get("metadata", {}))
        return cls(parameters=parameters, metadata=metadata, backend=backend)


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


@dataclass
class MinnaertModel(BasePhotometricModel):
    """Empirical baseline model skeleton."""

    model_name: str = "minnaert"

    def _reflectance_numpy(self, geometry: GeometryBatch) -> np.ndarray:
        raise NotImplementedError

    def _reflectance_torch(self, geometry: GeometryBatch) -> Any:
        raise NotImplementedError

    def parameter_names(self) -> list[str]:
        raise NotImplementedError

    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        raise NotImplementedError

    def parameter_priors(self) -> dict[str, ParameterPrior]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MinnaertModel":
        raise NotImplementedError


@dataclass
class AkimovModel(BasePhotometricModel):
    """Akimov model skeleton."""

    model_name: str = "akimov"

    def _reflectance_numpy(self, geometry: GeometryBatch) -> np.ndarray:
        raise NotImplementedError

    def _reflectance_torch(self, geometry: GeometryBatch) -> Any:
        raise NotImplementedError

    def parameter_names(self) -> list[str]:
        raise NotImplementedError

    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        raise NotImplementedError

    def parameter_priors(self) -> dict[str, ParameterPrior]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AkimovModel":
        raise NotImplementedError


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional dependency at skeleton stage
    torch = None

from photometry.core.types import Backend, GeometryBatch, ParameterPrior
from photometry.models.base import BasePhotometricModel, ModelRegistry


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

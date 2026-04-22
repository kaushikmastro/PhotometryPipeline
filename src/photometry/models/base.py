from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - optional dependency at skeleton stage
    torch = None

from photometry.core.types import ArrayLike, Backend, GeometryBatch, ParameterPrior, PredictionDistribution


class UnitError(ValueError):
    """Raised when model input angles appear to use the wrong units."""


class ModelRegistry:
    """
    Registry for photometric model classes.

    This keeps model construction independent from fitting code and supports
    reconstruction from serialized payloads using model_name.
    """

    _registry: ClassVar[dict[str, type["BasePhotometricModel"]]] = {}

    @classmethod
    def register(cls, model_cls: type["BasePhotometricModel"]) -> type["BasePhotometricModel"]:
        """Decorator used to register a model class by its model_name."""
        model_name = getattr(model_cls, "model_name", None)
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("Registered model classes must define a non-empty model_name.")
        cls._registry[model_name] = model_cls
        return model_cls

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BasePhotometricModel":
        """Reconstruct any registered model by looking up payload['model_name']."""
        model_name = str(payload.get("model_name", "")).strip()
        if model_name not in cls._registry:
            raise KeyError(f"Unknown model_name: {model_name}")
        return cls._registry[model_name].from_dict(payload)

    @classmethod
    def get_registered_models(cls) -> dict[str, type["BasePhotometricModel"]]:
        """Return a copy of the current registry mapping."""
        return dict(cls._registry)


@dataclass
class BasePhotometricModel(ABC):
    """
    Abstract base class for physics-only photometric models.

    Separation rule:
    - No fitting, MCMC, sampling, or optimizer logic here.
    - Models contain only mathematics and serialization for model state.

    Backend rule:
    - Every model must support NumPy and PyTorch backends.
    - Backend selection may be explicit or inferred from the input types.

    Vectorization rule:
    - Reflectance computations must be fully vectorized and suitable for
      arrays with at least 1_000_000 elements.
    """

    model_name: str = "base"
    parameters: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    backend: Backend = Backend.AUTO

    minimum_vectorization_n: ClassVar[int] = 1_000_000
    _angle_tolerance_rad: ClassVar[float] = 0.01

    def set_backend(self, backend: Backend) -> None:
        """Set the preferred backend for subsequent reflectance calls."""
        self.backend = backend

    def resolve_backend(self, geometry: GeometryBatch) -> Backend:
        """
        Resolve the effective backend.

        Policy:
        - If backend is explicitly NUMPY or TORCH, use it.
        - If backend is AUTO, infer TORCH when any geometry field is a tensor.
        - Otherwise use NUMPY.
        """
        if self.backend in (Backend.NUMPY, Backend.TORCH):
            return self.backend

        if torch is not None:
            fields = (geometry.incidence, geometry.emission, geometry.phase, geometry.azimuth)
            if any(isinstance(value, torch.Tensor) for value in fields if value is not None):
                return Backend.TORCH

        return Backend.NUMPY

    def _max_value(self, value: ArrayLike) -> float:
        """Return a scalar max for unit checks across NumPy or PyTorch inputs."""
        if torch is not None and isinstance(value, torch.Tensor):
            return float(torch.max(value).detach().cpu().item())
        return float(np.max(np.asarray(value)))

    def _min_value(self, value: ArrayLike) -> float:
        """Return a scalar min for unit checks across NumPy or PyTorch inputs."""
        if torch is not None and isinstance(value, torch.Tensor):
            return float(torch.min(value).detach().cpu().item())
        return float(np.min(np.asarray(value)))

    def _enforce_angle_units(self, geometry: GeometryBatch) -> None:
        """
        Enforce radians before backend dispatch.

        Requirement:
        - incidence and emission must be in [0, pi/2] radians.
        - If any value exceeds pi/2 + 0.01, raise UnitError with the exact
          message:
          angles appear to be in degrees, expected radians.
        """
        angle_limit = float(np.pi / 2.0 + self._angle_tolerance_rad)

        if self._max_value(geometry.incidence) > angle_limit:
            raise UnitError("angles appear to be in degrees, expected radians.")
        if self._max_value(geometry.emission) > angle_limit:
            raise UnitError("angles appear to be in degrees, expected radians.")

        if self._min_value(geometry.incidence) < 0.0 or self._min_value(geometry.emission) < 0.0:
            raise ValueError("incidence and emission angles must be non-negative.")

    def reflectance(self, geometry: GeometryBatch) -> ArrayLike:
        """
        Public reflectance entry point.

        This method:
        - validates angle units
        - resolves the backend
        - dispatches to the appropriate backend-specific implementation
        """
        self._enforce_angle_units(geometry)
        backend = self.resolve_backend(geometry)

        if backend is Backend.NUMPY:
            return self._reflectance_numpy(geometry)
        return self._reflectance_torch(geometry)

    @abstractmethod
    def _reflectance_numpy(self, geometry: GeometryBatch) -> np.ndarray:
        """NumPy backend reflectance kernel. Must be vectorized."""

    @abstractmethod
    def _reflectance_torch(self, geometry: GeometryBatch) -> Any:
        """PyTorch backend reflectance kernel. Must be vectorized."""

    @abstractmethod
    def parameter_names(self) -> list[str]:
        """Return the ordered list of free parameter names."""

    @abstractmethod
    def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """Return bounds for each free parameter."""

    @abstractmethod
    def parameter_priors(self) -> dict[str, ParameterPrior]:
        """Return prior definitions for each free parameter."""

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the model state.

        Expected payload includes:
        - model_name
        - parameters
        - metadata
        - backend
        """
        return {
            "model_name": self.model_name,
            "parameters": dict(self.parameters),
            "metadata": dict(self.metadata),
            "backend": self.backend.value,
        }

    @classmethod
    @abstractmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BasePhotometricModel":
        """Reconstruct a model from a serialized payload."""

    def predict_with_uncertainty(
        self,
        geometry: GeometryBatch,
        parameter_samples: ArrayLike,
        sample_axis: int = 0,
        return_summary: bool = True,
    ) -> PredictionDistribution:
        """
        Propagate parameter uncertainty through the model.

        parameter_samples contract:
        - Array of shape [n_samples, n_parameters] in parameter_names order.
        """
        raise NotImplementedError("Skeleton only: uncertainty implementation deferred.")

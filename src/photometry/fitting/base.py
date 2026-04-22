from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from photometry.core.types import ArrayLike, GeometryBatch
from photometry.models.base import BasePhotometricModel


@dataclass
class FitResult:
    """Generic fitter output."""

    model_name: str
    fitted_parameters: dict[str, float]
    objective_value: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCMCFitResult(FitResult):
    """MCMC-specific fit output."""

    posterior_samples: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    acceptance_fraction: float = 0.0
    n_steps: int = 0
    convergence_rhat: Optional[float] = None


class FittingStrategy(ABC):
    """Fitting-layer strategy base."""

    @abstractmethod
    def fit(
        self,
        model: BasePhotometricModel,
        geometry: GeometryBatch,
        observed_reflectance: ArrayLike,
        weights: Optional[ArrayLike] = None,
    ) -> FitResult:
        """Fit model parameters via dependency-injected model instance."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

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
    # One 1-sigma standard error per FREE parameter, keyed identically to
    # fitted_parameters. NaN (not omitted) marks a parameter whose error could
    # not be estimated (underdetermined fit, or railed at a parameter bound) —
    # see metadata["error_estimation_warning"] for why. Also mirrored into
    # metadata["parameter_errors"] for backward compatibility.
    parameter_errors: dict[str, float] | None = None

    # Expected metadata keys (added by fitters):
    # - "parameter_errors": dict[str, float] (mirrors the field above)
    # - "parameter_covariance": list[list[float]] (covariance matrix of fitted params)
    # - "reduced_chi_square": float (reduced chi-square of the fit)
    # - "boundary_hits": dict[str, bool] (which parameters hit bounds)
    # - "error_estimation_warning": str | None (why a parameter_errors entry is NaN, if any)


@dataclass
class MCMCFitResult(FitResult):
    """MCMC-specific fit output."""

    posterior_samples: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    acceptance_fraction: float = 0.0
    n_steps: int = 0
    convergence_rhat: float | None = None


class FittingStrategy(ABC):
    """Fitting-layer strategy base."""

    @abstractmethod
    def fit(
        self,
        model: BasePhotometricModel,
        geometry: GeometryBatch,
        observed_reflectance: ArrayLike,
        weights: ArrayLike | None = None,
    ) -> FitResult:
        """Fit model parameters via dependency-injected model instance."""

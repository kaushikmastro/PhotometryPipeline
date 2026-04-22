from __future__ import annotations

from photometry.fitting.base import FittingStrategy, MCMCFitResult
from photometry.core.types import ArrayLike, GeometryBatch
from photometry.models.base import BasePhotometricModel


class MCMCFitter(FittingStrategy):
    """MCMC fitter strategy skeleton."""

    def fit(
        self,
        model: BasePhotometricModel,
        geometry: GeometryBatch,
        observed_reflectance: ArrayLike,
        weights: ArrayLike | None = None,
    ) -> MCMCFitResult:
        raise NotImplementedError("MCMCFitter is not implemented yet.")

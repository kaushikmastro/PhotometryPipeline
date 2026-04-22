from __future__ import annotations

from photometry.fitting.base import FittingStrategy, FitResult
from photometry.core.types import ArrayLike, GeometryBatch
from photometry.models.base import BasePhotometricModel


class LeastSquaresFitter(FittingStrategy):
    """Least-squares fitter strategy skeleton."""

    def fit(
        self,
        model: BasePhotometricModel,
        geometry: GeometryBatch,
        observed_reflectance: ArrayLike,
        weights: ArrayLike | None = None,
    ) -> FitResult:
        raise NotImplementedError("LeastSquaresFitter is not implemented yet.")

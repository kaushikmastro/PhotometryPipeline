from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from photometry.core.types import GeometryBatch, ParameterPrior
from photometry.models.base import BasePhotometricModel


@dataclass
class SimplifiedHapkeModel(BasePhotometricModel):
    """Hapke model without opposition surge skeleton."""

    model_name: str = "hapke_simplified"

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
    def from_dict(cls, payload: Mapping[str, Any]) -> "SimplifiedHapkeModel":
        raise NotImplementedError


@dataclass
class FullHapkeModel(BasePhotometricModel):
    """Full Hapke model with opposition surge skeleton."""

    model_name: str = "hapke_full"

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
    def from_dict(cls, payload: Mapping[str, Any]) -> "FullHapkeModel":
        raise NotImplementedError

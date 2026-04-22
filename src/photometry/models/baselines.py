from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from photometry.core.types import GeometryBatch, ParameterPrior
from photometry.models.base import BasePhotometricModel


@dataclass
class LommelSeeligerModel(BasePhotometricModel):
    """Physical baseline model skeleton."""

    model_name: str = "lommel_seeliger"

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
    def from_dict(cls, payload: Mapping[str, Any]) -> "LommelSeeligerModel":
        raise NotImplementedError


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

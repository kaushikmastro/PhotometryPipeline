from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

import numpy as np

try:
    import torch
    from torch import Tensor
except Exception:  # pragma: no cover - optional dependency at skeleton stage
    torch = None
    Tensor = Any

ArrayLike = Union[np.ndarray, "Tensor"]


class Backend(Enum):
    """Execution backend selector for model evaluation."""

    AUTO = "auto"
    NUMPY = "numpy"
    TORCH = "torch"


@dataclass
class GeometryBatch:
    """
    Vectorized geometry container.

    Contract:
    - incidence, emission, and phase are same-length arrays.
    - Designed for efficient bulk arrays up to at least 1_000_000 elements.
    - Angles are expressed in radians.
    """

    incidence: ArrayLike
    emission: ArrayLike
    phase: ArrayLike
    azimuth: Optional[ArrayLike] = None


@dataclass
class ParameterPrior:
    """
    Prior specification for a single model parameter.

    prior_type examples:
    - uniform
    - gaussian
    """

    prior_type: str
    lower_bound: float
    upper_bound: float
    loc: Optional[float] = None
    scale: Optional[float] = None


@dataclass
class PredictionDistribution:
    """
    Output container for uncertainty propagation.

    predictions:
    - Typically shape [n_samples, n_points].
    """

    predictions: ArrayLike
    mean: Optional[ArrayLike] = None
    std: Optional[ArrayLike] = None
    q05: Optional[ArrayLike] = None
    q50: Optional[ArrayLike] = None
    q95: Optional[ArrayLike] = None

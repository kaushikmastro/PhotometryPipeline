from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares

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
        parameter_names = list(model.parameter_names())
        if not parameter_names:
            raise ValueError("Model must define at least one free parameter.")

        bounds_map = model.parameter_bounds()
        lower_bounds = []
        upper_bounds = []
        initial_guess = []

        for name in parameter_names:
            if name not in bounds_map:
                raise KeyError(f"Missing bounds for parameter: {name}")

            lower_bound, upper_bound = bounds_map[name]
            lower_bounds.append(float(lower_bound))
            upper_bounds.append(float(upper_bound))

            current_value = model.parameters.get(name)
            if current_value is None:
                initial_guess.append(0.5 * (float(lower_bound) + float(upper_bound)))
            else:
                initial_guess.append(float(current_value))

        lower_bounds_array = np.asarray(lower_bounds, dtype=float)
        upper_bounds_array = np.asarray(upper_bounds, dtype=float)
        initial_guess_array = np.asarray(initial_guess, dtype=float)
        initial_guess_array = np.clip(initial_guess_array, lower_bounds_array, upper_bounds_array)

        observed_array = np.asarray(observed_reflectance, dtype=float).reshape(-1)
        weights_array = None if weights is None else np.asarray(weights, dtype=float).reshape(-1)

        if observed_array.ndim != 1:
            raise ValueError("observed_reflectance must be one-dimensional after flattening.")
        if weights_array is not None and weights_array.shape != observed_array.shape:
            raise ValueError("weights must have the same shape as observed_reflectance.")

        original_parameters = dict(model.parameters)

        def residuals(parameter_vector: np.ndarray) -> np.ndarray:
            trial_parameters = dict(original_parameters)
            for index, name in enumerate(parameter_names):
                trial_parameters[name] = float(parameter_vector[index])

            model.parameters = trial_parameters
            predicted = np.asarray(model.reflectance(geometry), dtype=float).reshape(-1)
            residual = predicted - observed_array
            if weights_array is not None:
                residual = residual * np.sqrt(weights_array)
            return residual

        try:
            result = least_squares(
                residuals,
                x0=initial_guess_array,
                bounds=(lower_bounds_array, upper_bounds_array),
                method="trf",
                loss="soft_l1",
            )
        finally:
            model.parameters = original_parameters

        fitted_parameters = {
            name: float(result.x[index]) for index, name in enumerate(parameter_names)
        }

        return FitResult(
            model_name=model.model_name,
            fitted_parameters=fitted_parameters,
            objective_value=float(result.cost),
            metadata={
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "nfev": int(result.nfev),
                "njev": int(result.njev) if result.njev is not None else None,
                "optimality": float(result.optimality),
                "active_mask": result.active_mask.tolist() if result.active_mask is not None else None,
            },
        )

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

        # Support three kinds of `weights` inputs:
        # - None: unweighted fit
        # - ArrayLike: direct per-observation weights
        # - Mapping/dict with keys 'n_pixels' and 'iof_iqr': compute weights = sqrt(n_pixels) / iof_iqr
        weights_array = None
        weight_source = None
        if weights is None:
            weights_array = None
        else:
            # dict-like compute path
            try:
                if isinstance(weights, dict) and "n_pixels" in weights and "iof_iqr" in weights:
                    n_pixels = np.asarray(weights["n_pixels"], dtype=float).reshape(-1)
                    iof_iqr = np.asarray(weights["iof_iqr"], dtype=float).reshape(-1)
                    # avoid division by zero
                    iof_iqr_safe = np.where(iof_iqr == 0.0, np.nan, iof_iqr)
                    weights_array = np.sqrt(n_pixels) / iof_iqr_safe
                    weight_source = "n_pixels/iof_iqr"
                else:
                    weights_array = np.asarray(weights, dtype=float).reshape(-1)
                    weight_source = "array"
            except Exception:
                # fallback: treat as unweighted
                weights_array = None
                weight_source = None

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
                # least_squares has no explicit weights arg; apply via residual scaling
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

        # Compute covariance and parameter standard errors from Jacobian
        jac = result.jac
        param_cov = None
        param_errors = None
        reduced_chi2 = None
        covariance_pinv_used = False

        try:
            if jac is not None:
                # jac shape: (m, n) where m = n_observations, n = n_parameters
                m_obs, n_params = jac.shape
                # degrees of freedom
                dof = max(0, m_obs - n_params)
                # least_squares.cost is 0.5 * sum(residual**2)
                ssr = 2.0 * float(result.cost)

                if dof > 0:
                    sigma2 = ssr / float(dof)
                else:
                    sigma2 = float("nan")

                # Normal equations matrix
                jtj = np.dot(jac.T, jac)
                try:
                    inv_jtj = np.linalg.inv(jtj)
                except np.linalg.LinAlgError:
                    inv_jtj = np.linalg.pinv(jtj)
                    covariance_pinv_used = True

                cov = inv_jtj * sigma2
                param_cov = cov
                # standard errors: sqrt of diagonal (guard negative due to numeric noise)
                diag = np.diag(cov)
                diag_safe = np.where(diag >= 0.0, diag, np.abs(diag))
                param_errors = np.sqrt(diag_safe)
                reduced_chi2 = ssr / float(dof) if dof > 0 else float("nan")
        except Exception:
            param_cov = None
            param_errors = None
            reduced_chi2 = None
            covariance_pinv_used = False

        # Boundary hits: use active_mask when available, otherwise fall back to equality test
        boundary_hits = {}
        if getattr(result, "active_mask", None) is not None:
            mask = np.asarray(result.active_mask, dtype=bool)
            for idx, name in enumerate(parameter_names):
                boundary_hits[name] = bool(mask[idx])
        else:
            # fallback: check if fitted param equals bound (within tolerance)
            tol = 1e-12
            for idx, name in enumerate(parameter_names):
                val = float(result.x[idx])
                lb = float(lower_bounds_array[idx])
                ub = float(upper_bounds_array[idx])
                at_lower = abs(val - lb) <= tol * max(1.0, abs(lb))
                at_upper = abs(val - ub) <= tol * max(1.0, abs(ub))
                boundary_hits[name] = bool(at_lower or at_upper)

        metadata = {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "njev": int(result.njev) if result.njev is not None else None,
            "optimality": float(result.optimality),
            "active_mask": result.active_mask.tolist() if result.active_mask is not None else None,
            "weighted": bool(weights_array is not None),
            "weight_source": weight_source,
            "parameter_errors": {name: float(param_errors[i]) for i, name in enumerate(parameter_names)} if param_errors is not None else None,
            "parameter_covariance": param_cov.tolist() if param_cov is not None else None,
            "reduced_chi_square": float(reduced_chi2) if reduced_chi2 is not None and not np.isnan(reduced_chi2) else None,
            "boundary_hits": boundary_hits,
            "covariance_pinv_used": bool(covariance_pinv_used),
        }

        return FitResult(
            model_name=model.model_name,
            fitted_parameters=fitted_parameters,
            objective_value=float(result.cost),
            metadata=metadata,
        )

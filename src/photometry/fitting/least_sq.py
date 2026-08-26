from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from photometry.core.types import ArrayLike, GeometryBatch
from photometry.fitting.base import FitResult, FittingStrategy
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
                # Callers pass weights = 1/σ (inverse std), so multiply directly.
                # Do NOT use sqrt(weights): that would scale by 1/√σ and minimize
                # Σ(r²/σ) instead of the correct chi-squared Σ(r²/σ²).
                residual = residual * weights_array
            return residual

        try:
            result = least_squares(
                residuals,
                x0=initial_guess_array,
                bounds=(lower_bounds_array, upper_bounds_array),
                method="trf",
                # Changed from soft_l1 → linear: binned data has Gaussian errors (CLT,
                # each bin averages ≥10 pixels), so standard chi-squared is correct.
                # soft_l1 treated large systematic misfits at intermediate phase angles
                # as outliers, letting the optimizer ignore them; that caused B0 to pin
                # at its upper bound and theta_bar to collapse to ~0.
                loss="linear",
                # Default finite-diff step (1.49e-8) is too small for theta_bar:
                # at theta_bar < 2°, E1 = exp(-2/(π·tan_θ·tan_i)) hits machine
                # epsilon (~5e-16), making the numerical Jacobian identically zero.
                # 1e-4 keeps gradients detectable across the full [0°, 60°] range.
                diff_step=1e-4,
            )
        finally:
            model.parameters = original_parameters

        fitted_parameters = {
            name: float(result.x[index]) for index, name in enumerate(parameter_names)
        }

        # Boundary hits: use active_mask when available, otherwise fall back to equality test.
        # Computed before error estimation because a parameter railed at a bound has no
        # meaningful local-curvature uncertainty, regardless of what the covariance matrix says.
        boundary_hits: dict[str, bool] = {}
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

        # Parameter standard errors from the Jacobian. Convention: weights are treated as
        # RELATIVE (scipy curve_fit's absolute_sigma=False default) — cov = inv(J^T J) *
        # (2*cost/dof). Residuals are weighted exactly once upstream (in residuals()), so
        # this rescaling is not double-counting; it calibrates the absolute covariance
        # scale from the observed residual scatter rather than trusting the caller's
        # weights as an absolute noise level. Verified against scipy.optimize.curve_fit
        # on a known synthetic problem (see tests/test_least_sq_fitter.py).
        param_cov = None
        param_errors: dict[str, float] = {name: float("nan") for name in parameter_names}
        reduced_chi2 = None
        covariance_pinv_used = False
        warnings: list[str] = []

        jac = result.jac
        if jac is None:
            warnings.append("scipy did not return a Jacobian; all parameter errors are NaN.")
        else:
            m_obs, n_params = jac.shape
            dof = m_obs - n_params

            if dof <= 0:
                warnings.append(
                    f"underdetermined: n_obs ({m_obs}) <= n_free_params ({n_params}); "
                    "all parameter errors are NaN."
                )
            else:
                ssr = 2.0 * float(result.cost)  # least_squares.cost is 0.5 * sum(residual**2)
                sigma2 = ssr / float(dof)
                reduced_chi2 = sigma2

                jtj = np.dot(jac.T, jac)
                try:
                    inv_jtj = np.linalg.inv(jtj)
                except np.linalg.LinAlgError:
                    inv_jtj = np.linalg.pinv(jtj)
                    covariance_pinv_used = True
                    warnings.append(
                        "J^T J was singular; used pseudo-inverse (covariance may be unreliable)."
                    )

                cov = inv_jtj * sigma2
                param_cov = cov
                diag = np.diag(cov)

                for idx, name in enumerate(parameter_names):
                    if boundary_hits[name]:
                        # Railed at a bound: no meaningful local-curvature uncertainty,
                        # regardless of what the (possibly pinv'd) covariance reports.
                        continue
                    if diag[idx] < 0.0:
                        warnings.append(
                            f"'{name}': negative covariance diagonal ({diag[idx]:.3e}), "
                            "likely from an ill-conditioned fit; error set to NaN."
                        )
                        continue
                    param_errors[name] = float(np.sqrt(diag[idx]))

                railed = [name for name in parameter_names if boundary_hits[name]]
                if railed:
                    warnings.append(
                        f"parameters railed at a bound (no meaningful uncertainty): {railed}."
                    )

        error_estimation_warning = " ".join(warnings) if warnings else None

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
            "parameter_errors": dict(param_errors),
            "parameter_covariance": param_cov.tolist() if param_cov is not None else None,
            "reduced_chi_square": float(reduced_chi2) if reduced_chi2 is not None else None,
            "boundary_hits": boundary_hits,
            "covariance_pinv_used": bool(covariance_pinv_used),
            "error_estimation_warning": error_estimation_warning,
        }

        return FitResult(
            model_name=model.model_name,
            fitted_parameters=fitted_parameters,
            objective_value=float(result.cost),
            metadata=metadata,
            parameter_errors=dict(param_errors),
        )

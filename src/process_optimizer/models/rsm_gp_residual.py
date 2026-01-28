from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import FitResult, ProbabilisticRegressor
from .gp import GaussianProcessModel
from .response_surface import ResponseSurfaceModel


class RSMGPResidualModel(ProbabilisticRegressor):
    """Semi-parametric surrogate: Response Surface + GP residual.

    Structure:
      y(x) = f_rsm(x) + g_gp(x) + noise

    - f_rsm: regularized polynomial-like trend (RSM) that generalizes well on DOE data
    - g_gp: flexible residual model to capture remaining structure

    Uncertainty:
      - Combine trend and residual predictive std in quadrature (approximation).
    """

    name = "rsm_gp"

    def __init__(
        self,
        # RSM hyperparameters
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
        # GP hyperparameters
        nu: float = 2.5,
        length_scale: float = 1.0,
        noise_level: float = 1e-4,
        length_scale_bounds: Tuple[float, float] = (1e-5, 1e5),
        noise_level_bounds: Tuple[float, float] = (1e-8, 1e-1),
        alpha: float = 1e-8,
        optimize_kernel: bool = True,
        n_restarts_optimizer: int = 5,
        random_state: int = 42,
    ):
        self.trend = ResponseSurfaceModel(
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            random_state=random_state,
        )
        self.gp = GaussianProcessModel(
            nu=nu,
            length_scale=length_scale,
            noise_level=noise_level,
            length_scale_bounds=length_scale_bounds,
            noise_level_bounds=noise_level_bounds,
            alpha=alpha,
            optimize_kernel=optimize_kernel,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )
        self._fit_info: Dict[str, Any] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        tinfo = self.trend.fit(X, y).info or {}
        mu, _ = self.trend.predict(X, return_std=False)
        resid = y - np.asarray(mu).ravel()

        ginfo = self.gp.fit(X, resid).info or {}

        self._fit_info = {
            "trend": dict(tinfo),
            "gp": dict(ginfo),
        }
        return FitResult(info=self._fit_info)

    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X = np.asarray(X, dtype=float)

        mu_t, sd_t = self.trend.predict(X, return_std=return_std)
        mu_r, sd_r = self.gp.predict(X, return_std=return_std)

        mu = np.asarray(mu_t).ravel() + np.asarray(mu_r).ravel()

        if not return_std:
            return mu, None

        # Combine uncertainties in quadrature (assumes independence).
        if sd_t is None and sd_r is None:
            return mu, None
        if sd_t is None:
            return mu, np.asarray(sd_r).ravel()
        if sd_r is None:
            return mu, np.asarray(sd_t).ravel()

        sd = np.sqrt(np.asarray(sd_t).ravel() ** 2 + np.asarray(sd_r).ravel() ** 2)
        return mu, sd

from __future__ import annotations

"""DOE-informed neural surrogate: Bayesian RSM trend + neural residual.

This model complements the existing RSM+GP residual hybrid by swapping the residual
Gaussian Process with an MLP ensemble.

Motivation (DOE-sized data):
  • A regularized response surface (BayesianRidge) provides a stable low-order trend.
  • A small neural network ensemble captures remaining nonlinearity.
  • Ensemble dispersion provides a practical uncertainty proxy, which PEO further
    calibrates via its uncertainty calibration wrapper.

Formally:
    y(x) = f_rsm(x) + r_nn(x) + ε

Where:
  - f_rsm is the DOE-native response surface trend (linear in features).
  - r_nn is a neural residual model fit to (y - f_rsm).

Notes:
  - This class expects the caller to provide an appropriate feature set.
    In PEO, you typically enable DOE-informed features via doe_to_ml.enabled=True,
    which wraps the model with FeatureTransformedRegressor.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import FitResult, ProbabilisticRegressor
from .mlp_ensemble import MLPEnsembleModel
from .response_surface import ResponseSurfaceModel


class RSMMLPResidualModel(ProbabilisticRegressor):
    """Semi-parametric surrogate: Response Surface + MLP residual ensemble."""

    name = "rsm_mlp"

    def __init__(
        self,
        # RSM hyperparameters
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
        fit_intercept: bool = True,
        # Residual MLP-ensemble hyperparameters
        n_models: int = 15,
        hidden_layer_sizes=(128, 64),
        nn_alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter: int = 1500,
        # Misc
        random_state: int = 42,
    ):
        self.trend = ResponseSurfaceModel(
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            fit_intercept=fit_intercept,
            random_state=random_state,
        )
        self.residual = MLPEnsembleModel(
            n_models=n_models,
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=nn_alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        self._fit_info: Dict[str, Any] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        tinfo = self.trend.fit(X, y).info or {}
        mu_t, _ = self.trend.predict(X, return_std=False)
        resid = y - np.asarray(mu_t).ravel()

        rinfo = self.residual.fit(X, resid).info or {}

        self._fit_info = {
            "trend": dict(tinfo),
            "residual_nn": dict(rinfo),
        }
        return FitResult(info=self._fit_info)

    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X = np.asarray(X, dtype=float)

        mu_t, sd_t = self.trend.predict(X, return_std=return_std)
        mu_r, sd_r = self.residual.predict(X, return_std=return_std)

        mu = np.asarray(mu_t).ravel() + np.asarray(mu_r).ravel()
        if not return_std:
            return mu, None

        # Combine uncertainties in quadrature (approximate independence).
        if sd_t is None and sd_r is None:
            return mu, None
        if sd_t is None:
            return mu, np.asarray(sd_r).ravel()
        if sd_r is None:
            return mu, np.asarray(sd_t).ravel()

        sd = np.sqrt(np.asarray(sd_t).ravel() ** 2 + np.asarray(sd_r).ravel() ** 2)
        return mu, sd

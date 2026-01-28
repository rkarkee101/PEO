from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import BayesianRidge

from .base import FitResult, ProbabilisticRegressor


class ResponseSurfaceModel(ProbabilisticRegressor):
    """DOE-native response surface model (RSM).

    Implementation: Bayesian ridge regression.

    Why BayesianRidge?
    - Works well on small datasets (DOE-scale)
    - Built-in regularization guards against overfitting
    - Provides a predictive std estimate (return_std=True)

    Note:
    - This model expects the caller to provide an appropriate feature set.
      In practice, pair it with DOE-informed FeatureSpec (main effects + interactions + quadratics).
    """

    name = "rsm"

    def __init__(
        self,
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
        fit_intercept: bool = True,
        random_state: int = 42,
    ):
        self.model = BayesianRidge(
            alpha_1=float(alpha_1),
            alpha_2=float(alpha_2),
            lambda_1=float(lambda_1),
            lambda_2=float(lambda_2),
            fit_intercept=bool(fit_intercept),
            compute_score=True,
        )
        self.random_state = int(random_state)
        self._fit_info: Dict[str, Any] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        self.model.fit(X, y)
        self._fit_info = {
            "n_features": int(X.shape[1]),
            "alpha_": float(getattr(self.model, "alpha_", np.nan)),
            "lambda_": float(getattr(self.model, "lambda_", np.nan)),
        }
        return FitResult(info=self._fit_info)

    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X = np.asarray(X, dtype=float)
        if return_std:
            mu, sd = self.model.predict(X, return_std=True)
            return np.asarray(mu).ravel(), np.asarray(sd).ravel()
        mu = self.model.predict(X, return_std=False)
        return np.asarray(mu).ravel(), None

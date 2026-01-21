from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .base import FitResult, ProbabilisticRegressor


class GaussianProcessModel(ProbabilisticRegressor):
    name = "gp"

    def __init__(
        self,
        nu: float = 2.5,
        length_scale: float = 1.0,
        noise_level: float = 1e-4,
        normalize_y: bool = True,
        random_state: int = 42,
    ):
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=length_scale, nu=nu) + WhiteKernel(
            noise_level=noise_level, noise_level_bounds=(1e-8, 1e-1)
        )
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=normalize_y,
            random_state=random_state,
            n_restarts_optimizer=5,
        )
        self._fit_info: Dict[str, Any] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        self.model.fit(X, y)
        self._fit_info = {
            "kernel_": str(self.model.kernel_),
            "log_marginal_likelihood": float(self.model.log_marginal_likelihood_value_),
        }
        return FitResult(info=self._fit_info)

    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if return_std:
            mu, sd = self.model.predict(X, return_std=True)
            return np.asarray(mu).ravel(), np.asarray(sd).ravel()
        mu = self.model.predict(X, return_std=False)
        return np.asarray(mu).ravel(), None

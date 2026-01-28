from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .base import FitResult, ProbabilisticRegressor


class GaussianProcessModel(ProbabilisticRegressor):
    """Gaussian Process regressor wrapper with optional kernel optimization control.

    Notes
    -----
    - This project frequently performs *outer-loop* hyperparameter tuning (Optuna).
      In that setting, leaving scikit-learn's *inner* kernel optimizer enabled can:
        1) significantly increase runtime (nested optimization),
        2) push length-scales/noise toward extreme bounds (often overfitting),
        3) emit ConvergenceWarnings when optima land on bounds.

    - To support both workflows, this class exposes `optimize_kernel`.
      When False, the kernel hyperparameters are treated as fixed and scikit-learn
      does not run the L-BFGS optimization.
    """

    name = "gp"

    def __init__(
        self,
        nu: float = 2.5,
        length_scale: float = 1.0,
        noise_level: float = 1e-4,
        normalize_y: bool = True,
        random_state: int = 42,
        length_scale_bounds: Tuple[float, float] = (1e-5, 1e5),
        noise_level_bounds: Tuple[float, float] = (1e-8, 1e-1),
        alpha: float = 1e-8,
        optimize_kernel: bool = True,
        n_restarts_optimizer: int = 5,
    ):
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=length_scale, length_scale_bounds=tuple(length_scale_bounds), nu=nu
        ) + WhiteKernel(noise_level=noise_level, noise_level_bounds=tuple(noise_level_bounds))

        optimizer = None if not bool(optimize_kernel) else "fmin_l_bfgs_b"
        n_restarts = int(n_restarts_optimizer) if bool(optimize_kernel) else 0

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=normalize_y,
            random_state=random_state,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts,
            alpha=float(alpha),
            copy_X_train=False,
        )
        self._fit_info: Dict[str, Any] = {
            "optimize_kernel": bool(optimize_kernel),
            "alpha": float(alpha),
            "length_scale_bounds": [float(length_scale_bounds[0]), float(length_scale_bounds[1])],
            "noise_level_bounds": [float(noise_level_bounds[0]), float(noise_level_bounds[1])],
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        self.model.fit(X, y)
        self._fit_info.update(
            {
                "kernel_": str(self.model.kernel_),
                "log_marginal_likelihood": float(self.model.log_marginal_likelihood_value_),
            }
        )
        return FitResult(info=self._fit_info)

    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if return_std:
            mu, sd = self.model.predict(X, return_std=True)
            return np.asarray(mu).ravel(), np.asarray(sd).ravel()
        mu = self.model.predict(X, return_std=False)
        return np.asarray(mu).ravel(), None

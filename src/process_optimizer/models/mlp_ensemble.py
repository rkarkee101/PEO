from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings

from .base import FitResult, ProbabilisticRegressor


class MLPEnsembleModel(ProbabilisticRegressor):
    """Bootstrap ensemble of sklearn MLP regressors.

    Uncertainty: std across members.

    This is a lightweight deep-learning-style baseline that does not require torch.
    """

    name = "mlp"

    def __init__(
        self,
        n_models: int = 15,
        hidden_layer_sizes=(128, 64),
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter: int = 1500,
        random_state: int = 42,
    ):
        self.n_models = int(n_models)
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.alpha = float(alpha)
        self.learning_rate_init = float(learning_rate_init)
        self.max_iter = int(max_iter)
        self.rng = np.random.default_rng(int(random_state))
        self.models = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        n = X.shape[0]
        self.models = []
        converged_flags = []
        for i in range(self.n_models):
            idx = self.rng.choice(n, size=n, replace=True)
            m = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                alpha=self.alpha,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                random_state=int(self.rng.integers(0, 2**31 - 1)),
                early_stopping=True,
                n_iter_no_change=25,
                validation_fraction=0.15,
            )
            # MLPRegressor can be chatty about max_iter even with early_stopping.
            # We keep early_stopping enabled and suppress repetitive warnings.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                m.fit(X[idx], y[idx])
            self.models.append(m)
            # Not a strict guarantee, but if it stopped before max_iter it likely converged.
            converged_flags.append(bool(getattr(m, "n_iter_", self.max_iter) < self.max_iter))
        return FitResult(
            info={
                "n_models": self.n_models,
                "hidden_layer_sizes": list(self.hidden_layer_sizes),
                "alpha": self.alpha,
                "learning_rate_init": self.learning_rate_init,
                "max_iter": self.max_iter,
                "converged_fraction": float(np.mean(converged_flags)) if converged_flags else None,
            }
        )

    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.models:
            raise RuntimeError("Model is not fit")
        preds = np.vstack([m.predict(X) for m in self.models])
        mu = np.mean(preds, axis=0)
        if not return_std:
            return np.asarray(mu).ravel(), None
        sd = np.std(preds, axis=0)
        return np.asarray(mu).ravel(), np.asarray(sd).ravel()

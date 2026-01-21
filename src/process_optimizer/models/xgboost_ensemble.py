from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .base import FitResult, ProbabilisticRegressor


class XGBoostEnsembleModel(ProbabilisticRegressor):
    """Bootstrap ensemble of XGBoost models.

    Uncertainty: std across ensemble members.

    If xgboost is not installed, constructing this class raises ImportError.
    """

    name = "xgboost"

    def __init__(
        self,
        n_models: int = 25,
        n_estimators: int = 700,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        reg_lambda: float = 1.0,
        random_state: int = 42,
    ):
        try:
            from xgboost import XGBRegressor  # type: ignore
        except Exception as e:
            raise ImportError("xgboost is not installed") from e

        self._XGBRegressor = XGBRegressor
        self.n_models = int(n_models)
        self.params = {
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
            "learning_rate": float(learning_rate),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample_bytree),
            "reg_lambda": float(reg_lambda),
            "objective": "reg:squarederror",
            "n_jobs": -1,
        }
        self.rng = np.random.default_rng(int(random_state))
        self.models = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        n = X.shape[0]
        self.models = []
        for _ in range(self.n_models):
            idx = self.rng.choice(n, size=n, replace=True)
            m = self._XGBRegressor(**self.params)
            m.fit(X[idx], y[idx])
            self.models.append(m)
        return FitResult(info={"n_models": self.n_models, **self.params})

    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.models:
            raise RuntimeError("Model is not fit")
        preds = np.vstack([m.predict(X) for m in self.models])
        mu = np.mean(preds, axis=0)
        if not return_std:
            return np.asarray(mu).ravel(), None
        sd = np.std(preds, axis=0)
        return np.asarray(mu).ravel(), np.asarray(sd).ravel()

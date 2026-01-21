from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base import FitResult, ProbabilisticRegressor


class RandomForestModel(ProbabilisticRegressor):
    name = "random_forest"

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        random_state: int = 42,
    ):
        self.model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            min_samples_leaf=int(min_samples_leaf),
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        self.model.fit(X, y)
        return FitResult(info={"n_estimators": int(self.model.n_estimators)})

    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        mu = self.model.predict(X)
        mu = np.asarray(mu).ravel()
        if not return_std:
            return mu, None
        # Use empirical std across tree predictions as a proxy uncertainty.
        preds = np.vstack([t.predict(X) for t in self.model.estimators_])
        sd = np.std(preds, axis=0)
        return mu, np.asarray(sd).ravel()

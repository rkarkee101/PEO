from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .base import FitResult, ProbabilisticRegressor


class UncertaintyCalibratedRegressor(ProbabilisticRegressor):
    """Wrap a regressor and apply a multiplicative calibration to its predicted std.

    Many surrogate models produce miscalibrated epistemic uncertainty on small datasets.
    A simple and robust fix is to rescale std by a scalar factor computed on held-out
    residuals (or CV residuals).

    This wrapper is intentionally simple: it does not alter the mean prediction.
    """

    def __init__(self, base: ProbabilisticRegressor, scale: float = 1.0, min_std: float = 0.0):
        self.base = base
        self.scale = float(scale)
        self.min_std = float(min_std)
        # Keep the base name so downstream code doesn't need to special-case it.
        self.name = getattr(base, "name", base.__class__.__name__)

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        return self.base.fit(X, y)

    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        mu, sd = self.base.predict(X, return_std=return_std)
        if not return_std or sd is None:
            return mu, sd
        sd = np.asarray(sd, dtype=float).ravel()
        sd = sd * self.scale
        if self.min_std > 0:
            sd = np.maximum(sd, self.min_std)
        return np.asarray(mu).ravel(), sd

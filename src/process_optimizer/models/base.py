from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class FitResult:
    info: Dict[str, Any]


class ProbabilisticRegressor:
    """Thin interface for regressors that can return uncertainty."""

    name: str = "base"

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        raise NotImplementedError

    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError

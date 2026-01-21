from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from process_optimizer.feature_engineering import FeatureSpec

from .base import FitResult, ProbabilisticRegressor


class FeatureTransformedRegressor(ProbabilisticRegressor):
    """Wrap a regressor so it always sees the engineered feature space.

    This is used to integrate DOE-informed feature selection and interaction
    terms into *any* underlying model while keeping downstream code unchanged.
    Inverse design and query paths only need to pass the base preprocessed X;
    this wrapper applies the FeatureSpec transform internally.
    """

    def __init__(self, base: ProbabilisticRegressor, spec: FeatureSpec, name_suffix: str = "+doe"):
        self.base = base
        self.spec = spec
        self.name = f"{getattr(base, 'name', base.__class__.__name__)}{name_suffix}"

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        Xt = self.spec.transform(X)
        res = self.base.fit(Xt, y)
        info = dict(res.info or {})
        info["feature_spec"] = self.spec.to_dict()
        info["engineered_n_features"] = int(Xt.shape[1])
        info["engineered_feature_names"] = list(self.spec.feature_names)
        return FitResult(info=info)

    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        Xt = self.spec.transform(X)
        return self.base.predict(Xt, return_std=return_std)

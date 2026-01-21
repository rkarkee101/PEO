from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class CandidateSolution:
    params: Dict[str, Any]
    score: float
    pred: float
    std: Optional[float]
    model_name: str


def _sample_candidates(bounds: Dict[str, Tuple[float, float]], categories: Dict[str, List[str]], n: int, rng) -> pd.DataFrame:
    cols = {}
    for k, (lo, hi) in bounds.items():
        cols[k] = rng.uniform(float(lo), float(hi), size=n)
    for k, vals in categories.items():
        if not vals:
            continue
        cols[k] = rng.choice(vals, size=n, replace=True)
    return pd.DataFrame(cols)


class InverseDesigner:
    def __init__(
        self,
        preprocessor,
        tool_params: List[str],
        bounds: Dict[str, Tuple[float, float]],
        categories: Dict[str, List[str]],
        lambda_uncertainty: float = 0.35,
        random_state: int = 42,
    ):
        self.preprocessor = preprocessor
        self.tool_params = list(tool_params)
        self.bounds = dict(bounds)
        self.categories = dict(categories)
        self.lambda_uncertainty = float(lambda_uncertainty)
        self.rng = np.random.default_rng(int(random_state))

    def propose(
        self,
        model_wrappers: List[Any],
        target_name: str,
        target_value: float,
        search_budget: int = 6000,
        top_k: int = 10,
        method: str = "random",
    ) -> List[CandidateSolution]:
        if search_budget <= 0:
            return []

        method = str(method).lower().strip()
        if method in {"bayes", "bayesopt", "bo", "bayesian"}:
            try:
                return self._propose_bayesopt(
                    model_wrappers=model_wrappers,
                    target_name=target_name,
                    target_value=float(target_value),
                    search_budget=int(search_budget),
                    top_k=int(top_k),
                )
            except Exception:
                # Fallback to random if bayesopt backend isn't available or fails.
                pass

        return self._propose_random(
            model_wrappers=model_wrappers,
            target_name=target_name,
            target_value=float(target_value),
            search_budget=int(search_budget),
            top_k=int(top_k),
        )

    def _propose_random(
        self,
        model_wrappers: List[Any],
        target_name: str,
        target_value: float,
        search_budget: int,
        top_k: int,
    ) -> List[CandidateSolution]:

        df_cand = _sample_candidates(self.bounds, self.categories, int(search_budget), self.rng)
        # Ensure all tool params exist
        for p in self.tool_params:
            if p not in df_cand.columns:
                # categorical with no levels or numeric missing
                if p in self.categories and self.categories.get(p):
                    df_cand[p] = self.rng.choice(self.categories[p], size=len(df_cand), replace=True)
                else:
                    lo, hi = self.bounds.get(p, (0.0, 1.0))
                    df_cand[p] = self.rng.uniform(float(lo), float(hi), size=len(df_cand))

        Xcand = self.preprocessor.transform(df_cand[self.tool_params])
        if hasattr(Xcand, "toarray"):
            Xcand = Xcand.toarray()
        Xcand = np.asarray(Xcand, dtype=float)

        sols: List[CandidateSolution] = []
        for m in model_wrappers:
            pred, std = m.predict(Xcand, return_std=True)
            pred = np.asarray(pred).ravel()
            if std is not None:
                std = np.asarray(std).ravel()

            # objective: squared error plus uncertainty penalty
            err = (pred - float(target_value)) ** 2
            if std is None:
                score = err
                std_used = None
            else:
                score = err + float(self.lambda_uncertainty) * (std**2)
                std_used = std

            best_idx = np.argsort(score)[: int(top_k)]
            for i in best_idx:
                params: Dict[str, Any] = {}
                for p in self.tool_params:
                    v = df_cand.loc[i, p]
                    if p in self.bounds:
                        params[p] = float(v)
                    else:
                        params[p] = str(v)
                sols.append(
                    CandidateSolution(
                        params=params,
                        score=float(score[i]),
                        pred=float(pred[i]),
                        std=(float(std_used[i]) if std_used is not None else None),
                        model_name=getattr(m, "name", m.__class__.__name__),
                    )
                )

        # Global rank across models
        sols.sort(key=lambda s: s.score)
        return sols[: int(top_k)]

    def _propose_bayesopt(
        self,
        model_wrappers: List[Any],
        target_name: str,
        target_value: float,
        search_budget: int,
        top_k: int,
    ) -> List[CandidateSolution]:
        """Bayesian optimization for inverse design.

        Uses scikit-optimize (skopt) when installed. Optimizes the objective:
        (pred-target)^2 + lambda_uncertainty*std^2

        This is still *model-based* and purely offline: it optimizes the trained
        surrogate, not the physical process.
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Categorical
        except Exception as e:
            raise RuntimeError("Bayesian optimization requires scikit-optimize. Install with: pip install -e '.[bo]' ") from e

        # Build skopt search space in tool_params order
        dims = []
        for p in self.tool_params:
            if p in self.bounds:
                lo, hi = self.bounds[p]
                dims.append(Real(float(lo), float(hi), name=p))
            elif p in self.categories and self.categories.get(p):
                dims.append(Categorical(list(self.categories[p]), name=p))
            else:
                # default to [0,1] if missing
                lo, hi = self.bounds.get(p, (0.0, 1.0))
                dims.append(Real(float(lo), float(hi), name=p))

        # Choose a budget for BO; thousands of calls are unnecessary because each evaluation is cheap.
        n_calls = int(min(120, max(25, search_budget // 80)))
        n_initial = int(min(20, max(8, n_calls // 4)))

        sols: List[CandidateSolution] = []
        for m in model_wrappers:
            def objective(x_list):
                row = {d.name: v for d, v in zip(dims, x_list)}
                df_row = pd.DataFrame([row])
                Xrow = self.preprocessor.transform(df_row[self.tool_params])
                if hasattr(Xrow, "toarray"):
                    Xrow = Xrow.toarray()
                Xrow = np.asarray(Xrow, dtype=float)
                pred, std = m.predict(Xrow, return_std=True)
                pred = float(np.asarray(pred).ravel()[0])
                if std is None:
                    return float((pred - target_value) ** 2)
                sd = float(np.asarray(std).ravel()[0])
                return float((pred - target_value) ** 2 + self.lambda_uncertainty * (sd**2))

            res = gp_minimize(
                objective,
                dimensions=dims,
                n_calls=n_calls,
                n_initial_points=n_initial,
                random_state=int(self.rng.integers(0, 2**31 - 1)),
                acq_func="EI",
            )

            # Collect best few observed points
            scored = list(zip(res.x_iters, res.func_vals))
            scored.sort(key=lambda t: float(t[1]))
            for x_list, score in scored[: int(top_k)]:
                row = {d.name: v for d, v in zip(dims, x_list)}
                df_row = pd.DataFrame([row])
                Xrow = self.preprocessor.transform(df_row[self.tool_params])
                if hasattr(Xrow, "toarray"):
                    Xrow = Xrow.toarray()
                Xrow = np.asarray(Xrow, dtype=float)
                pred, std = m.predict(Xrow, return_std=True)
                pred = float(np.asarray(pred).ravel()[0])
                sd = None if std is None else float(np.asarray(std).ravel()[0])
                params: Dict[str, Any] = {}
                for p in self.tool_params:
                    v = row.get(p)
                    if p in self.bounds:
                        params[p] = float(v)
                    else:
                        params[p] = str(v)
                sols.append(
                    CandidateSolution(
                        params=params,
                        score=float(score),
                        pred=pred,
                        std=sd,
                        model_name=getattr(m, "name", m.__class__.__name__),
                    )
                )

        sols.sort(key=lambda s: s.score)
        return sols[: int(top_k)]

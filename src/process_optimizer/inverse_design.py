from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class CandidateSolution:
    """A candidate recipe from inverse design.

    This dataclass supports both single-target and multi-target inverse design:
      - Single-target: `target`, `pred`, `std` are populated.
      - Multi-target: `preds`/`stds` mappings are populated.
    """

    params: Dict[str, Any]
    score: float
    model_name: str

    # Single-target fields
    target: Optional[str] = None
    pred: Optional[float] = None
    std: Optional[float] = None

    # Multi-target fields
    preds: Optional[Dict[str, float]] = None
    stds: Optional[Dict[str, Optional[float]]] = None

    # Additional diagnostics
    ood_distance: Optional[float] = None
    penalty: Optional[Dict[str, float]] = None


def _sample_candidates(
    bounds: Mapping[str, Tuple[float, float]],
    categories: Mapping[str, List[str]],
    n: int,
    rng: np.random.Generator,
    *,
    fixed_params: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Uniformly sample candidate recipes inside bounds + categorical levels."""

    n = int(max(0, n))
    fixed_params = dict(fixed_params or {})

    cols: Dict[str, Any] = {}
    for k, (lo, hi) in (bounds or {}).items():
        if k in fixed_params:
            cols[k] = np.full(n, float(fixed_params[k]), dtype=float)
        else:
            cols[k] = rng.uniform(float(lo), float(hi), size=n)

    for k, vals in (categories or {}).items():
        if not vals:
            continue
        if k in fixed_params:
            cols[k] = np.array([str(fixed_params[k])] * n, dtype=object)
        else:
            cols[k] = rng.choice(list(vals), size=n, replace=True)

    # Add any fixed params that are outside provided bounds/categories.
    for k, v in fixed_params.items():
        if k not in cols:
            cols[k] = np.array([v] * n, dtype=object)

    return pd.DataFrame(cols)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


def _greedy_diversify(
    candidates: Sequence[CandidateSolution],
    *,
    tool_params: Sequence[str],
    bounds: Mapping[str, Tuple[float, float]],
    categories: Mapping[str, List[str]],
    top_k: int,
    min_distance: float,
) -> List[CandidateSolution]:
    """Pick a diverse subset of candidates in normalized parameter space.

    This is a pragmatic non-uniqueness handler: inverse problems often admit many
    near-equivalent solutions. We avoid returning trivial near-duplicates.
    """

    top_k = int(max(0, top_k))
    if top_k <= 0:
        return []
    if not candidates:
        return []

    # --- Build a normalized embedding for each candidate ---
    cat_levels = {k: list(v) for k, v in (categories or {}).items() if v}
    cat_index: Dict[str, Dict[str, int]] = {
        k: {str(v): i for i, v in enumerate(vals)} for k, vals in cat_levels.items()
    }

    emb: List[np.ndarray] = []
    for c in candidates:
        vec_parts: List[float] = []
        # numeric: scale to [0,1]
        for p in tool_params:
            if p in bounds:
                lo, hi = bounds[p]
                span = float(hi) - float(lo)
                span = span if span > 0 else 1.0
                v = float(c.params.get(p, lo))
                vec_parts.append((v - float(lo)) / span)
        # categorical: one-hot
        for p, vals in cat_levels.items():
            oh = [0.0] * len(vals)
            v = str(c.params.get(p, vals[0]))
            j = cat_index[p].get(v)
            if j is not None:
                oh[int(j)] = 1.0
            vec_parts.extend(oh)
        emb.append(np.asarray(vec_parts, dtype=float))

    # If embedding degenerate, just return top_k.
    if not emb or emb[0].size == 0:
        return list(candidates[:top_k])

    E = np.stack(emb, axis=0)
    # --- Greedy farthest-first selection with a minimum distance threshold ---
    selected_idx: List[int] = [0]
    if top_k == 1:
        return [candidates[0]]

    # Track min-distance to selected set for each point
    dmin = np.linalg.norm(E - E[0:1], axis=1)

    while len(selected_idx) < top_k:
        # Find farthest point from selected set
        i = int(np.argmax(dmin))
        if float(dmin[i]) < float(min_distance):
            break
        selected_idx.append(i)
        dnew = np.linalg.norm(E - E[i : i + 1], axis=1)
        dmin = np.minimum(dmin, dnew)

    return [candidates[i] for i in selected_idx]


class InverseDesigner:
    def __init__(
        self,
        preprocessor,
        tool_params: List[str],
        bounds: Dict[str, Tuple[float, float]],
        categories: Dict[str, List[str]],
        *,
        lambda_uncertainty: float = 0.35,
        lambda_ood: float = 0.0,
        ood_model: Any = None,
        ood_scale: Optional[float] = None,
        diversity_enabled: bool = True,
        diversity_min_distance: float = 0.12,
        diversity_max_candidates: int = 500,
        random_state: int = 42,
    ):
        self.preprocessor = preprocessor
        self.tool_params = list(tool_params)
        self.bounds = dict(bounds)
        self.categories = dict(categories)

        self.lambda_uncertainty = float(lambda_uncertainty)
        self.lambda_ood = float(lambda_ood)

        self.ood_model = ood_model
        self.ood_scale = None if ood_scale is None else float(ood_scale)

        self.diversity_enabled = bool(diversity_enabled)
        self.diversity_min_distance = float(diversity_min_distance)
        self.diversity_max_candidates = int(max(50, diversity_max_candidates))

        self.rng = np.random.default_rng(int(random_state))

    # ---------------------------------------------------------------------
    # Public APIs
    # ---------------------------------------------------------------------
    def propose(
        self,
        model_wrappers: List[Any],
        target_name: str,
        target_value: float,
        *,
        search_budget: int = 6000,
        top_k: int = 10,
        method: str = "random",
        fixed_params: Optional[Mapping[str, Any]] = None,
    ) -> List[CandidateSolution]:
        """Single-target inverse design (backward compatible)."""

        if int(search_budget) <= 0 or not model_wrappers:
            return []

        method = str(method).lower().strip()
        if method in {"bayes", "bayesopt", "bo", "bayesian"}:
            try:
                return self._propose_bayesopt_single(
                    model_wrappers=model_wrappers,
                    target_name=str(target_name),
                    target_value=float(target_value),
                    search_budget=int(search_budget),
                    top_k=int(top_k),
                    fixed_params=fixed_params,
                )
            except Exception:
                # Fallback to random if bayesopt backend isn't available or fails.
                pass

        return self._propose_random_single(
            model_wrappers=model_wrappers,
            target_name=str(target_name),
            target_value=float(target_value),
            search_budget=int(search_budget),
            top_k=int(top_k),
            fixed_params=fixed_params,
        )

    def propose_multi(
        self,
        models_by_target: Mapping[str, Any],
        targets: Mapping[str, float],
        *,
        search_budget: int = 6000,
        top_k: int = 10,
        method: str = "random",
        fixed_params: Optional[Mapping[str, Any]] = None,
        weights: Optional[Mapping[str, float]] = None,
        tolerances: Optional[Mapping[str, float]] = None,
        scales: Optional[Mapping[str, float]] = None,
    ) -> List[CandidateSolution]:
        """Multi-target inverse design.

        Parameters
        - models_by_target: mapping {target_name: model_wrapper}
        - targets: mapping {target_name: desired_value}
        - weights: optional objective weights per target
        - tolerances: optional tolerance per target (activates hinge penalty)
        - scales: optional normalization scale per target (fallback when tolerance absent)
        """

        targets = {str(k): float(v) for k, v in (targets or {}).items()}
        if int(search_budget) <= 0 or not targets or not models_by_target:
            return []

        # Keep only the intersection.
        models = {str(k): models_by_target[str(k)] for k in targets.keys() if str(k) in models_by_target}
        if not models:
            return []

        method = str(method).lower().strip()
        if method in {"bayes", "bayesopt", "bo", "bayesian"}:
            try:
                return self._propose_bayesopt_multi(
                    models_by_target=models,
                    targets=targets,
                    search_budget=int(search_budget),
                    top_k=int(top_k),
                    fixed_params=fixed_params,
                    weights=weights,
                    tolerances=tolerances,
                    scales=scales,
                )
            except Exception:
                pass

        return self._propose_random_multi(
            models_by_target=models,
            targets=targets,
            search_budget=int(search_budget),
            top_k=int(top_k),
            fixed_params=fixed_params,
            weights=weights,
            tolerances=tolerances,
            scales=scales,
        )

    # ---------------------------------------------------------------------
    # Internals: utilities
    # ---------------------------------------------------------------------
    def _transform_candidates(self, df_cand: pd.DataFrame) -> np.ndarray:
        Xcand = self.preprocessor.transform(df_cand[self.tool_params])
        if hasattr(Xcand, "toarray"):
            Xcand = Xcand.toarray()
        return np.asarray(Xcand, dtype=float)

    def _ood_distance(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self.ood_model is None:
            return None
        try:
            dist, _ = self.ood_model.kneighbors(np.asarray(X, dtype=float), n_neighbors=1, return_distance=True)
            d = np.asarray(dist).reshape(-1)
            return d
        except Exception:
            return None

    def _ensure_all_params_present(self, df: pd.DataFrame, fixed_params: Optional[Mapping[str, Any]]) -> pd.DataFrame:
        df2 = df.copy()
        fixed_params = dict(fixed_params or {})
        for p in self.tool_params:
            if p in df2.columns:
                continue
            if p in fixed_params:
                df2[p] = fixed_params[p]
                continue
            if p in self.categories and self.categories.get(p):
                df2[p] = self.rng.choice(self.categories[p], size=len(df2), replace=True)
            else:
                lo, hi = self.bounds.get(p, (0.0, 1.0))
                df2[p] = self.rng.uniform(float(lo), float(hi), size=len(df2))
        # Enforce fixed params (override any sampled values)
        for k, v in fixed_params.items():
            if k in df2.columns:
                df2[k] = v
        return df2

    def _pack_params_from_df(self, df_cand: pd.DataFrame, idx: int) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for p in self.tool_params:
            v = df_cand.loc[idx, p]
            if p in self.bounds:
                params[p] = float(v)
            else:
                params[p] = str(v)
        return params

    def _normalize_scale_map(self, keys: Iterable[str], scales: Optional[Mapping[str, float]]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k in keys:
            s = 1.0
            if scales and k in scales:
                try:
                    s = float(scales[k])
                except Exception:
                    s = 1.0
            if not np.isfinite(s) or s <= 0:
                s = 1.0
            out[str(k)] = float(s)
        return out

    # ---------------------------------------------------------------------
    # Random search: single target
    # ---------------------------------------------------------------------
    def _propose_random_single(
        self,
        *,
        model_wrappers: List[Any],
        target_name: str,
        target_value: float,
        search_budget: int,
        top_k: int,
        fixed_params: Optional[Mapping[str, Any]],
    ) -> List[CandidateSolution]:

        df_cand = _sample_candidates(self.bounds, self.categories, int(search_budget), self.rng, fixed_params=fixed_params)
        df_cand = self._ensure_all_params_present(df_cand, fixed_params)

        Xcand = self._transform_candidates(df_cand)
        ood_d = self._ood_distance(Xcand)

        # Normalize OOD penalty by a characteristic scale.
        ood_scale = self.ood_scale
        if ood_scale is not None and (not np.isfinite(ood_scale) or ood_scale <= 0):
            ood_scale = None

        sols: List[CandidateSolution] = []
        for m in model_wrappers:
            pred, std = m.predict(Xcand, return_std=True)
            pred = np.asarray(pred).ravel()
            std_arr = None if std is None else np.asarray(std).ravel()

            err = (pred - float(target_value)) ** 2
            score = err.copy()
            penalty: Dict[str, np.ndarray] = {}

            if std_arr is not None:
                score = score + float(self.lambda_uncertainty) * (std_arr**2)
                penalty["uncertainty"] = float(self.lambda_uncertainty) * (std_arr**2)

            if ood_d is not None and float(self.lambda_ood) > 0:
                d = np.asarray(ood_d).ravel()
                denom = float(ood_scale) if ood_scale is not None else float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
                denom = denom if np.isfinite(denom) and denom > 0 else 1.0
                score = score + float(self.lambda_ood) * (d / denom) ** 2
                penalty["ood"] = float(self.lambda_ood) * (d / denom) ** 2

            # Keep more than top_k to enable diversification
            keep_n = int(min(len(score), max(int(top_k), 1) * 25))
            best_idx = np.argsort(score)[:keep_n]
            for i in best_idx:
                params = self._pack_params_from_df(df_cand, int(i))
                sols.append(
                    CandidateSolution(
                        params=params,
                        score=float(score[i]),
                        model_name=getattr(m, "name", m.__class__.__name__),
                        target=str(target_name),
                        pred=float(pred[i]),
                        std=(float(std_arr[i]) if std_arr is not None else None),
                        ood_distance=(float(ood_d[i]) if ood_d is not None else None),
                        penalty={k: float(np.asarray(v)[i]) for k, v in penalty.items()} if penalty else None,
                    )
                )

        sols.sort(key=lambda s: float(s.score))
        sols = sols[: int(min(len(sols), self.diversity_max_candidates))]
        if self.diversity_enabled:
            return _greedy_diversify(
                sols,
                tool_params=self.tool_params,
                bounds=self.bounds,
                categories=self.categories,
                top_k=int(top_k),
                min_distance=float(self.diversity_min_distance),
            )
        return sols[: int(top_k)]

    # ---------------------------------------------------------------------
    # Random search: multi target
    # ---------------------------------------------------------------------
    def _propose_random_multi(
        self,
        *,
        models_by_target: Mapping[str, Any],
        targets: Mapping[str, float],
        search_budget: int,
        top_k: int,
        fixed_params: Optional[Mapping[str, Any]],
        weights: Optional[Mapping[str, float]],
        tolerances: Optional[Mapping[str, float]],
        scales: Optional[Mapping[str, float]],
    ) -> List[CandidateSolution]:

        tnames = [str(t) for t in targets.keys()]

        wmap = {t: 1.0 for t in tnames}
        if weights:
            for k, v in weights.items():
                if str(k) in wmap:
                    try:
                        wmap[str(k)] = float(v)
                    except Exception:
                        pass
        # Normalize weights (avoid extreme scale explosions)
        sw = float(sum(max(0.0, float(v)) for v in wmap.values()))
        if sw <= 0:
            sw = 1.0
        wmap = {k: float(max(0.0, v) / sw) for k, v in wmap.items()}

        tolmap: Dict[str, Optional[float]] = {t: None for t in tnames}
        if tolerances:
            for k, v in tolerances.items():
                kk = str(k)
                if kk in tolmap:
                    try:
                        fv = float(v)
                        tolmap[kk] = fv if np.isfinite(fv) and fv > 0 else None
                    except Exception:
                        tolmap[kk] = None

        scmap = self._normalize_scale_map(tnames, scales)

        df_cand = _sample_candidates(self.bounds, self.categories, int(search_budget), self.rng, fixed_params=fixed_params)
        df_cand = self._ensure_all_params_present(df_cand, fixed_params)
        Xcand = self._transform_candidates(df_cand)

        ood_d = self._ood_distance(Xcand)
        ood_scale = self.ood_scale
        if ood_scale is not None and (not np.isfinite(ood_scale) or ood_scale <= 0):
            ood_scale = None

        preds: Dict[str, np.ndarray] = {}
        stds: Dict[str, Optional[np.ndarray]] = {}
        for t in tnames:
            m = models_by_target[t]
            mu, sd = m.predict(Xcand, return_std=True)
            preds[t] = np.asarray(mu).ravel()
            stds[t] = None if sd is None else np.asarray(sd).ravel()

        # Objective aggregation
        score = np.zeros(Xcand.shape[0], dtype=float)
        penalty_terms: Dict[str, np.ndarray] = {}

        for t in tnames:
            mu = preds[t]
            err = mu - float(targets[t])

            tol = tolmap.get(t)
            if tol is not None:
                # hinge penalty: 0 inside tolerance, quadratic outside
                mis = np.maximum(0.0, np.abs(err) - float(tol)) / float(tol)
                mis2 = mis**2
            else:
                s = float(scmap.get(t, 1.0))
                mis2 = (err / max(s, 1e-12)) ** 2

            score = score + float(wmap[t]) * mis2

            sd = stds.get(t)
            if sd is not None:
                # Normalize uncertainty on the same scale as the misfit.
                s = float(tol) if tol is not None else float(scmap.get(t, 1.0))
                up = (sd / max(s, 1e-12)) ** 2
                score = score + float(self.lambda_uncertainty) * float(wmap[t]) * up
                penalty_terms.setdefault("uncertainty", np.zeros_like(score))
                penalty_terms["uncertainty"] += float(self.lambda_uncertainty) * float(wmap[t]) * up

        if ood_d is not None and float(self.lambda_ood) > 0:
            d = np.asarray(ood_d).ravel()
            denom = float(ood_scale) if ood_scale is not None else float(np.median(d[d > 0])) if np.any(d > 0) else 1.0
            denom = denom if np.isfinite(denom) and denom > 0 else 1.0
            op = float(self.lambda_ood) * (d / denom) ** 2
            score = score + op
            penalty_terms["ood"] = op

        # Keep a decent pool before diversity selection
        keep_n = int(min(len(score), max(int(top_k), 1) * 50))
        best_idx = np.argsort(score)[:keep_n]

        sols: List[CandidateSolution] = []
        model_name = "+".join([f"{t}:{getattr(models_by_target[t], 'name', models_by_target[t].__class__.__name__)}" for t in tnames])
        for i in best_idx:
            i = int(i)
            params = self._pack_params_from_df(df_cand, i)
            preds_i = {t: float(preds[t][i]) for t in tnames}
            stds_i = {t: (None if stds[t] is None else float(np.asarray(stds[t])[i])) for t in tnames}
            sols.append(
                CandidateSolution(
                    params=params,
                    score=float(score[i]),
                    model_name=model_name,
                    preds=preds_i,
                    stds=stds_i,
                    ood_distance=(float(ood_d[i]) if ood_d is not None else None),
                    penalty={k: float(np.asarray(v)[i]) for k, v in penalty_terms.items()} if penalty_terms else None,
                )
            )

        sols.sort(key=lambda s: float(s.score))
        sols = sols[: int(min(len(sols), self.diversity_max_candidates))]
        if self.diversity_enabled:
            return _greedy_diversify(
                sols,
                tool_params=self.tool_params,
                bounds=self.bounds,
                categories=self.categories,
                top_k=int(top_k),
                min_distance=float(self.diversity_min_distance),
            )
        return sols[: int(top_k)]

    # ---------------------------------------------------------------------
    # BO: single target
    # ---------------------------------------------------------------------
    def _propose_bayesopt_single(
        self,
        *,
        model_wrappers: List[Any],
        target_name: str,
        target_value: float,
        search_budget: int,
        top_k: int,
        fixed_params: Optional[Mapping[str, Any]],
    ) -> List[CandidateSolution]:
        """Bayesian optimization for inverse design.

        Uses scikit-optimize (skopt) when installed. Optimizes the objective:
          (pred-target)^2 + lambda_uncertainty*std^2 + lambda_ood*ood^2
        """

        try:
            from skopt import gp_minimize
            from skopt.space import Categorical, Real
        except Exception as e:
            raise RuntimeError(
                "Bayesian optimization requires scikit-optimize. Install with: pip install -e '.[bo]'"
            ) from e

        fixed_params = dict(fixed_params or {})

        # Build skopt search space in tool_params order (excluding fixed params)
        dims = []
        var_params: List[str] = []
        for p in self.tool_params:
            if p in fixed_params:
                continue
            if p in self.bounds:
                lo, hi = self.bounds[p]
                dims.append(Real(float(lo), float(hi), name=p))
                var_params.append(p)
            elif p in self.categories and self.categories.get(p):
                dims.append(Categorical(list(self.categories[p]), name=p))
                var_params.append(p)
            else:
                lo, hi = self.bounds.get(p, (0.0, 1.0))
                dims.append(Real(float(lo), float(hi), name=p))
                var_params.append(p)

        # If everything is fixed, fall back to evaluating the single fixed point.
        if not dims:
            df_row = pd.DataFrame([{p: fixed_params.get(p) for p in self.tool_params}])
            df_row = self._ensure_all_params_present(df_row, fixed_params)
            Xrow = self._transform_candidates(df_row)
            ood_d = self._ood_distance(Xrow)
            sols: List[CandidateSolution] = []
            for m in model_wrappers:
                pred, std = m.predict(Xrow, return_std=True)
                mu = float(np.asarray(pred).ravel()[0])
                sd = None if std is None else float(np.asarray(std).ravel()[0])
                score = float((mu - target_value) ** 2)
                if sd is not None:
                    score += float(self.lambda_uncertainty) * (sd**2)
                if ood_d is not None and float(self.lambda_ood) > 0:
                    denom = float(self.ood_scale) if self.ood_scale is not None else 1.0
                    denom = denom if np.isfinite(denom) and denom > 0 else 1.0
                    score += float(self.lambda_ood) * (float(ood_d[0]) / denom) ** 2
                sols.append(
                    CandidateSolution(
                        params={p: (float(df_row.loc[0, p]) if p in self.bounds else str(df_row.loc[0, p])) for p in self.tool_params},
                        score=score,
                        model_name=getattr(m, "name", m.__class__.__name__),
                        target=str(target_name),
                        pred=mu,
                        std=sd,
                        ood_distance=(float(ood_d[0]) if ood_d is not None else None),
                    )
                )
            sols.sort(key=lambda s: float(s.score))
            return sols[: int(top_k)]

        # Choose a budget for BO.
        n_calls = int(min(160, max(30, search_budget // 60)))
        n_initial = int(min(30, max(10, n_calls // 4)))

        sols: List[CandidateSolution] = []
        for m in model_wrappers:

            def objective(x_list):
                row = dict(fixed_params)
                for d, v in zip(dims, x_list):
                    row[d.name] = v
                df_row = pd.DataFrame([row])
                df_row = self._ensure_all_params_present(df_row, fixed_params)
                Xrow = self._transform_candidates(df_row)

                mu, sd = m.predict(Xrow, return_std=True)
                mu = float(np.asarray(mu).ravel()[0])
                loss = float((mu - target_value) ** 2)
                if sd is not None:
                    s = float(np.asarray(sd).ravel()[0])
                    loss += float(self.lambda_uncertainty) * (s**2)

                if self.ood_model is not None and float(self.lambda_ood) > 0:
                    d = self._ood_distance(Xrow)
                    if d is not None:
                        denom = float(self.ood_scale) if self.ood_scale is not None else 1.0
                        denom = denom if np.isfinite(denom) and denom > 0 else 1.0
                        loss += float(self.lambda_ood) * (float(d[0]) / denom) ** 2
                return float(loss)

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
            keep_n = int(min(len(scored), max(int(top_k), 1) * 25))
            for x_list, sc in scored[:keep_n]:
                row = dict(fixed_params)
                for d, v in zip(dims, x_list):
                    row[d.name] = v
                df_row = pd.DataFrame([row])
                df_row = self._ensure_all_params_present(df_row, fixed_params)
                Xrow = self._transform_candidates(df_row)
                mu, sd = m.predict(Xrow, return_std=True)
                mu = float(np.asarray(mu).ravel()[0])
                sdv = None if sd is None else float(np.asarray(sd).ravel()[0])
                ood_d = self._ood_distance(Xrow)
                sols.append(
                    CandidateSolution(
                        params={p: (float(df_row.loc[0, p]) if p in self.bounds else str(df_row.loc[0, p])) for p in self.tool_params},
                        score=float(sc),
                        model_name=getattr(m, "name", m.__class__.__name__),
                        target=str(target_name),
                        pred=mu,
                        std=sdv,
                        ood_distance=(float(ood_d[0]) if ood_d is not None else None),
                    )
                )

        sols.sort(key=lambda s: float(s.score))
        sols = sols[: int(min(len(sols), self.diversity_max_candidates))]
        if self.diversity_enabled:
            return _greedy_diversify(
                sols,
                tool_params=self.tool_params,
                bounds=self.bounds,
                categories=self.categories,
                top_k=int(top_k),
                min_distance=float(self.diversity_min_distance),
            )
        return sols[: int(top_k)]

    # ---------------------------------------------------------------------
    # BO: multi target
    # ---------------------------------------------------------------------
    def _propose_bayesopt_multi(
        self,
        *,
        models_by_target: Mapping[str, Any],
        targets: Mapping[str, float],
        search_budget: int,
        top_k: int,
        fixed_params: Optional[Mapping[str, Any]],
        weights: Optional[Mapping[str, float]],
        tolerances: Optional[Mapping[str, float]],
        scales: Optional[Mapping[str, float]],
    ) -> List[CandidateSolution]:

        try:
            from skopt import gp_minimize
            from skopt.space import Categorical, Real
        except Exception as e:
            raise RuntimeError(
                "Bayesian optimization requires scikit-optimize. Install with: pip install -e '.[bo]'"
            ) from e

        fixed_params = dict(fixed_params or {})
        tnames = [str(t) for t in targets.keys()]

        # weights / tolerances / scales match random implementation
        wmap = {t: 1.0 for t in tnames}
        if weights:
            for k, v in weights.items():
                kk = str(k)
                if kk in wmap:
                    try:
                        wmap[kk] = float(v)
                    except Exception:
                        pass
        sw = float(sum(max(0.0, float(v)) for v in wmap.values()))
        if sw <= 0:
            sw = 1.0
        wmap = {k: float(max(0.0, v) / sw) for k, v in wmap.items()}

        tolmap: Dict[str, Optional[float]] = {t: None for t in tnames}
        if tolerances:
            for k, v in tolerances.items():
                kk = str(k)
                if kk in tolmap:
                    try:
                        fv = float(v)
                        tolmap[kk] = fv if np.isfinite(fv) and fv > 0 else None
                    except Exception:
                        tolmap[kk] = None

        scmap = self._normalize_scale_map(tnames, scales)

        # Build search dimensions excluding fixed
        dims = []
        for p in self.tool_params:
            if p in fixed_params:
                continue
            if p in self.bounds:
                lo, hi = self.bounds[p]
                dims.append(Real(float(lo), float(hi), name=p))
            elif p in self.categories and self.categories.get(p):
                dims.append(Categorical(list(self.categories[p]), name=p))
            else:
                lo, hi = self.bounds.get(p, (0.0, 1.0))
                dims.append(Real(float(lo), float(hi), name=p))

        # Handle fully fixed as a special case
        if not dims:
            return self._propose_random_multi(
                models_by_target=models_by_target,
                targets=targets,
                search_budget=1,
                top_k=top_k,
                fixed_params=fixed_params,
                weights=wmap,
                tolerances=tolmap,
                scales=scmap,
            )

        # Choose a budget for BO.
        n_calls = int(min(220, max(40, search_budget // 55)))
        n_initial = int(min(40, max(12, n_calls // 4)))

        def objective(x_list):
            row = dict(fixed_params)
            for d, v in zip(dims, x_list):
                row[d.name] = v
            df_row = pd.DataFrame([row])
            df_row = self._ensure_all_params_present(df_row, fixed_params)
            Xrow = self._transform_candidates(df_row)

            loss = 0.0
            for t in tnames:
                m = models_by_target[t]
                mu, sd = m.predict(Xrow, return_std=True)
                mu = float(np.asarray(mu).ravel()[0])
                err = mu - float(targets[t])

                tol = tolmap.get(t)
                if tol is not None:
                    mis = max(0.0, abs(err) - float(tol)) / float(tol)
                    loss += float(wmap[t]) * (mis**2)
                else:
                    s = float(scmap.get(t, 1.0))
                    loss += float(wmap[t]) * ((err / max(s, 1e-12)) ** 2)

                if sd is not None:
                    sdv = float(np.asarray(sd).ravel()[0])
                    s = float(tol) if tol is not None else float(scmap.get(t, 1.0))
                    loss += float(self.lambda_uncertainty) * float(wmap[t]) * ((sdv / max(s, 1e-12)) ** 2)

            if self.ood_model is not None and float(self.lambda_ood) > 0:
                d = self._ood_distance(Xrow)
                if d is not None:
                    denom = float(self.ood_scale) if self.ood_scale is not None else 1.0
                    denom = denom if np.isfinite(denom) and denom > 0 else 1.0
                    loss += float(self.lambda_ood) * (float(d[0]) / denom) ** 2

            return float(loss)

        res = gp_minimize(
            objective,
            dimensions=dims,
            n_calls=n_calls,
            n_initial_points=n_initial,
            random_state=int(self.rng.integers(0, 2**31 - 1)),
            acq_func="EI",
        )

        scored = list(zip(res.x_iters, res.func_vals))
        scored.sort(key=lambda t: float(t[1]))
        keep_n = int(min(len(scored), max(int(top_k), 1) * 60))

        sols: List[CandidateSolution] = []
        model_name = "+".join([f"{t}:{getattr(models_by_target[t], 'name', models_by_target[t].__class__.__name__)}" for t in tnames])

        for x_list, sc in scored[:keep_n]:
            row = dict(fixed_params)
            for d, v in zip(dims, x_list):
                row[d.name] = v
            df_row = pd.DataFrame([row])
            df_row = self._ensure_all_params_present(df_row, fixed_params)
            Xrow = self._transform_candidates(df_row)
            ood_d = self._ood_distance(Xrow)

            preds_i: Dict[str, float] = {}
            stds_i: Dict[str, Optional[float]] = {}
            for t in tnames:
                m = models_by_target[t]
                mu, sd = m.predict(Xrow, return_std=True)
                preds_i[t] = float(np.asarray(mu).ravel()[0])
                stds_i[t] = None if sd is None else float(np.asarray(sd).ravel()[0])

            params = {p: (float(df_row.loc[0, p]) if p in self.bounds else str(df_row.loc[0, p])) for p in self.tool_params}
            sols.append(
                CandidateSolution(
                    params=params,
                    score=float(sc),
                    model_name=model_name,
                    preds=preds_i,
                    stds=stds_i,
                    ood_distance=(float(ood_d[0]) if ood_d is not None else None),
                )
            )

        sols.sort(key=lambda s: float(s.score))
        sols = sols[: int(min(len(sols), self.diversity_max_candidates))]
        if self.diversity_enabled:
            return _greedy_diversify(
                sols,
                tool_params=self.tool_params,
                bounds=self.bounds,
                categories=self.categories,
                top_k=int(top_k),
                min_distance=float(self.diversity_min_distance),
            )
        return sols[: int(top_k)]

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class FeatureSpec:
    """Feature selection + DOE-informed augmentation spec.

    The pipeline first converts raw tool parameters into a base feature matrix X
    using the saved sklearn preprocessor (scaling + one-hot encoding). FeatureSpec
    then optionally:
      1) selects a subset of base features (by indices)
      2) appends interaction features (products of selected numeric base features)
      3) appends power terms (e.g., quadratic x^2) for selected numeric base features

    This spec is saved with each trained model so inverse design and queries can
    apply the exact same feature transform.
    """

    base_indices: Optional[List[int]]  # None means keep all
    interaction_pairs: List[Tuple[int, int]]
    power_terms: List[Tuple[int, int]]  # (base_feature_index, power), e.g. (i,2) for x_i^2
    base_feature_names: List[str]
    interaction_feature_names: List[str]
    power_feature_names: List[str]

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.base_indices is None:
            Xb = X
        else:
            Xb = X[:, self.base_indices]

        feats = [Xb]

        # Interactions
        if self.interaction_pairs:
            inter = []
            for i, j in self.interaction_pairs:
                inter.append((X[:, i] * X[:, j]).reshape(-1, 1))
            Xi = np.concatenate(inter, axis=1) if inter else np.empty((X.shape[0], 0))
            feats.append(Xi)

        # Power terms
        if self.power_terms:
            pw = []
            for i, p in self.power_terms:
                pw.append((X[:, i] ** int(p)).reshape(-1, 1))
            Xp = np.concatenate(pw, axis=1) if pw else np.empty((X.shape[0], 0))
            feats.append(Xp)

        if len(feats) == 1:
            return feats[0]
        return np.concatenate(feats, axis=1)

    @property
    def feature_names(self) -> List[str]:
        return list(self.base_feature_names) + list(self.interaction_feature_names) + list(self.power_feature_names)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_indices": self.base_indices,
            "interaction_pairs": [list(p) for p in self.interaction_pairs],
            "power_terms": [list(p) for p in self.power_terms],
            "base_feature_names": list(self.base_feature_names),
            "interaction_feature_names": list(self.interaction_feature_names),
            "power_feature_names": list(self.power_feature_names),
        }


def _factor_to_feature_indices(feature_names: Sequence[str], factors: Sequence[str]) -> List[int]:
    """Map original factor names to indices in the preprocessed feature matrix.

    - Numeric factor: exact match
    - Categorical factor: include all one-hot columns that start with "{factor}_"
    """
    idx: List[int] = []
    fset = [str(f) for f in factors]
    for i, fn in enumerate(feature_names):
        for f in fset:
            if fn == f or fn.startswith(f + "_"):
                idx.append(i)
                break
    # unique, preserve order
    seen = set()
    out = []
    for i in idx:
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out


def build_doe_informed_spec(
    cfg: Dict[str, Any],
    feature_names: Sequence[str],
    doe_analysis: Dict[str, Any],
) -> FeatureSpec:
    """Create a FeatureSpec from DOE analysis results for one target.

    The current implementation uses DOE analysis p-values to:
      - select influential factors (main effects)
      - add significant 2-factor numeric interactions
      - optionally add significant quadratic curvature terms for numeric factors

    This is intentionally conservative for DOE-sized datasets: we prefer a smaller,
    hierarchy-respecting feature set to reduce overfitting risk.
    """

    doe2ml = (cfg.get("doe_to_ml") or {})
    sel_cfg = (doe2ml.get("selection") or {})
    int_cfg = (doe2ml.get("interactions") or {})
    quad_cfg = (doe2ml.get("quadratic") or {})

    pthr = float(sel_cfg.get("p_value_threshold", 0.15))
    top_k = int(sel_cfg.get("top_k", 8))
    keep_at_least = int(sel_cfg.get("keep_at_least", 3))
    always_keep = [str(x) for x in (sel_cfg.get("always_keep") or [])]

    effects = doe_analysis.get("effects", {}) or {}
    ranked = sorted(effects.items(), key=lambda kv: float((kv[1] or {}).get("p", 1.0)))
    selected_factors = [k for k, v in ranked if float((v or {}).get("p", 1.0)) <= pthr]

    if not selected_factors:
        selected_factors = [k for k, _ in ranked[: max(0, top_k)]]

    for f in always_keep:
        if f not in selected_factors:
            selected_factors.append(f)

    if len(selected_factors) < keep_at_least:
        for k, _ in ranked:
            if k not in selected_factors:
                selected_factors.append(k)
            if len(selected_factors) >= keep_at_least:
                break

    base_indices = _factor_to_feature_indices(feature_names, selected_factors)
    if not base_indices:
        base_indices = list(range(len(feature_names)))
        base_feature_names = list(map(str, feature_names))
    else:
        base_feature_names = [str(feature_names[i]) for i in base_indices]

    # Interactions (numeric-numeric only; DOEAnalyzer only emits numeric interactions)
    interaction_pairs: List[Tuple[int, int]] = []
    interaction_names: List[str] = []
    if bool(int_cfg.get("enabled", True)):
        ipthr = float(int_cfg.get("p_value_threshold", 0.15))
        itop = int(int_cfg.get("top_k", 10))
        ints = doe_analysis.get("interactions", {}) or {}
        ranked_int = sorted(ints.items(), key=lambda kv: float((kv[1] or {}).get("p", 1.0)))
        chosen = [(k, v) for k, v in ranked_int if float((v or {}).get("p", 1.0)) <= ipthr]
        if not chosen:
            chosen = ranked_int[: max(0, itop)]

        for name, _meta in chosen:
            if not name or ":" not in str(name):
                continue
            a, b = str(name).split(":", 1)
            a = a.strip()
            b = b.strip()
            try:
                ia = list(feature_names).index(a)
                ib = list(feature_names).index(b)
            except ValueError:
                continue
            interaction_pairs.append((int(ia), int(ib)))
            interaction_names.append(f"{a}*{b}")

            # Strong hierarchy: include main effects whenever we add an interaction
            if ia not in base_indices:
                base_indices.append(ia)
                base_feature_names.append(str(feature_names[ia]))
            if ib not in base_indices:
                base_indices.append(ib)
                base_feature_names.append(str(feature_names[ib]))

    # Quadratic curvature terms (numeric only; DOEAnalyzer emits per-factor p-values).
    power_terms: List[Tuple[int, int]] = []
    power_names: List[str] = []
    if bool(quad_cfg.get("enabled", True)):
        qpthr = float(quad_cfg.get("p_value_threshold", 0.20))
        qtop = int(quad_cfg.get("top_k", 6))
        quad = doe_analysis.get("quadratic", {}) or {}
        ranked_q = sorted(quad.items(), key=lambda kv: float((kv[1] or {}).get("p", 1.0)))
        chosen_q = [(k, v) for k, v in ranked_q if float((v or {}).get("p", 1.0)) <= qpthr]
        if not chosen_q:
            chosen_q = ranked_q[: max(0, qtop)]

        for fac, _meta in chosen_q:
            f = str(fac)
            try:
                idx = list(feature_names).index(f)
            except ValueError:
                continue
            power_terms.append((int(idx), 2))
            power_names.append(f"{f}^2")
            # hierarchy: include main effect for the squared term
            if idx not in base_indices:
                base_indices.append(int(idx))
                base_feature_names.append(str(feature_names[idx]))

    # De-duplicate base indices in original order
    seen = set()
    base_idx_unique: List[int] = []
    base_names_unique: List[str] = []
    for i, nm in zip(base_indices, base_feature_names):
        if i in seen:
            continue
        seen.add(i)
        base_idx_unique.append(int(i))
        base_names_unique.append(str(nm))

    return FeatureSpec(
        base_indices=base_idx_unique,
        interaction_pairs=interaction_pairs,
        power_terms=power_terms,
        base_feature_names=base_names_unique,
        interaction_feature_names=interaction_names,
        power_feature_names=power_names,
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class FeatureSpec:
    """Feature selection + interaction augmentation spec.

    The pipeline first converts raw tool parameters into a base feature matrix X
    using the saved sklearn preprocessor (scaling + one-hot encoding). FeatureSpec
    then optionally:
      1) selects a subset of base features (by indices)
      2) appends interaction features (products of selected numeric base features)

    This spec is saved with each trained model so inverse design and queries can
    apply the exact same feature transform.
    """

    base_indices: Optional[List[int]]  # None means keep all
    interaction_pairs: List[Tuple[int, int]]
    base_feature_names: List[str]
    interaction_feature_names: List[str]

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.base_indices is None:
            Xb = X
        else:
            Xb = X[:, self.base_indices]

        if not self.interaction_pairs:
            return Xb

        inter = []
        for i, j in self.interaction_pairs:
            inter.append((X[:, i] * X[:, j]).reshape(-1, 1))
        Xi = np.concatenate(inter, axis=1) if inter else np.empty((X.shape[0], 0))
        return np.concatenate([Xb, Xi], axis=1)

    @property
    def feature_names(self) -> List[str]:
        return list(self.base_feature_names) + list(self.interaction_feature_names)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_indices": self.base_indices,
            "interaction_pairs": [list(p) for p in self.interaction_pairs],
            "base_feature_names": list(self.base_feature_names),
            "interaction_feature_names": list(self.interaction_feature_names),
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
    """Create a FeatureSpec from DOE analysis results for one target."""

    # Defaults
    doe2ml = (cfg.get("doe_to_ml") or {})
    sel_cfg = (doe2ml.get("selection") or {})
    int_cfg = (doe2ml.get("interactions") or {})

    pthr = float(sel_cfg.get("p_value_threshold", 0.15))
    top_k = int(sel_cfg.get("top_k", 8))
    keep_at_least = int(sel_cfg.get("keep_at_least", 3))
    always_keep = [str(x) for x in (sel_cfg.get("always_keep") or [])]

    effects = doe_analysis.get("effects", {}) or {}
    # Sort by p-value
    ranked = sorted(effects.items(), key=lambda kv: float((kv[1] or {}).get("p", 1.0)))
    selected_factors = [k for k, v in ranked if float((v or {}).get("p", 1.0)) <= pthr]

    # top_k fallback
    if not selected_factors:
        selected_factors = [k for k, _ in ranked[: max(0, top_k)]]

    # enforce minimum + always_keep
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
        base_feature_names = list(feature_names)
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
        chosen = [
            (k, v)
            for k, v in ranked_int
            if float((v or {}).get("p", 1.0)) <= ipthr
        ]
        if not chosen:
            chosen = ranked_int[: max(0, itop)]

        for name, meta in chosen:
            if not name or ":" not in name:
                continue
            a, b = name.split(":", 1)
            a = a.strip()
            b = b.strip()
            # Find numeric feature index (exact match for scaled numeric)
            try:
                ia = list(feature_names).index(a)
                ib = list(feature_names).index(b)
            except ValueError:
                continue
            interaction_pairs.append((int(ia), int(ib)))
            interaction_names.append(f"{a}*{b}")

            # Ensure base includes factors used by interactions
            if ia not in base_indices:
                base_indices.append(ia)
                base_feature_names.append(str(feature_names[ia]))
            if ib not in base_indices:
                base_indices.append(ib)
                base_feature_names.append(str(feature_names[ib]))

    # De-duplicate base indices in original order
    seen = set()
    base_idx_unique: List[int] = []
    base_names_unique: List[str] = []
    for i, nm in zip(base_indices, base_feature_names):
        if i in seen:
            continue
        seen.add(i)
        base_idx_unique.append(i)
        base_names_unique.append(nm)

    return FeatureSpec(
        base_indices=base_idx_unique,
        interaction_pairs=interaction_pairs,
        base_feature_names=base_names_unique,
        interaction_feature_names=interaction_names,
    )

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML config into a plain dict."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {p} must parse to a mapping (dict)")
    return cfg


def validate_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight validation plus defaults.

    The config stays flexible by design. This function normalizes a few common
    keys and applies safe defaults.
    """
    cfg = dict(cfg)

    cfg.setdefault("project", {})
    cfg["project"].setdefault("name", "process-engineering-optimizer")

    if "data" not in cfg:
        raise ValueError("config.yaml must include a 'data' section")
    data = cfg["data"]
    for k in ["tool_parameters", "target_properties"]:
        if k not in data or not isinstance(data[k], list) or not data[k]:
            raise ValueError(f"data.{k} must be a non-empty list")

    data.setdefault("delimiter", ",")
    data.setdefault("encoding", "utf-8")

    # Backward-compatible aliases
    # - categorical_parameters: ["gas", ...]
    # - categorical_params: dict (name -> levels) or list of names
    if "categorical_params" not in data and "categorical_parameters" in data:
        data["categorical_params"] = data.get("categorical_parameters")

    cat = data.get("categorical_params", {})
    if cat is None:
        cat = {}
    if isinstance(cat, list):
        cat = {str(k): [] for k in cat}
    if not isinstance(cat, dict):
        raise ValueError("data.categorical_params must be a dict or a list")
    data["categorical_params"] = cat

    # Units aliases
    if "units" not in data and "target_units" in data:
        data["units"] = data.get("target_units") or {}
    data.setdefault("units", {})

    # Factor space (optional). If absent, inferred from the measured CSV.
    cfg.setdefault("factor_space", {})
    cfg["factor_space"].setdefault("bounds", {})
    cfg["factor_space"].setdefault("categories", {})

    # DOE
    cfg.setdefault("doe", {})
    doe = cfg["doe"]
    doe.setdefault(
        "methods",
        [
            "full_factorial",
            "fractional_factorial",
            "plackett_burman",
            "latin_hypercube",
            "central_composite",
            "box_behnken",
        ],
    )
    doe.setdefault("n_samples", 24)
    doe.setdefault("fractional_resolution", 3)
    doe.setdefault("interaction_depth", 2)

    # Training
    cfg.setdefault("training", {})
    tr = cfg["training"]
    tr.setdefault("models", ["gp", "random_forest", "xgboost", "mlp"])
    tr.setdefault("test_size", 0.2)
    tr.setdefault("cv_folds", 5)
    tr.setdefault("random_state", 42)
    tr.setdefault("autotune", True)
    tr.setdefault("max_tuning_trials", 40)
    tr.setdefault("tuning_timeout_s", 0)
    tr.setdefault("overfit_guard", {"max_train_test_r2_gap": 0.15})

    # DOE -> ML feature engineering (optional)
    cfg.setdefault("doe_to_ml", {})
    d2m = cfg["doe_to_ml"]
    d2m.setdefault("enabled", False)
    d2m.setdefault("selection", {})
    d2m["selection"].setdefault("p_value_threshold", 0.15)
    d2m["selection"].setdefault("top_k", 8)
    d2m["selection"].setdefault("keep_at_least", 3)
    d2m["selection"].setdefault("always_keep", [])
    d2m.setdefault("interactions", {})
    d2m["interactions"].setdefault("enabled", True)
    d2m["interactions"].setdefault("p_value_threshold", 0.15)
    d2m["interactions"].setdefault("top_k", 10)

    # Inverse design
    cfg.setdefault("inverse_design", {})
    inv = cfg["inverse_design"]
    inv.setdefault("search_budget", 6000)
    inv.setdefault("top_k", 10)
    if "lambda_uncertainty" not in inv and "uncertainty_weight" in inv:
        inv["lambda_uncertainty"] = inv.get("uncertainty_weight")
    inv.setdefault("lambda_uncertainty", 0.35)

    # Storage
    cfg.setdefault("storage", {})
    cfg["storage"].setdefault("root", "./storage")

    # RAG
    cfg.setdefault("rag", {})
    rag = cfg["rag"]
    rag.setdefault("retriever", "tfidf")
    rag.setdefault("top_k", 6)
    rag.setdefault("st_model", "all-MiniLM-L6-v2")

    # Logging
    cfg.setdefault("logging", {})
    cfg["logging"].setdefault("level", "INFO")

    return cfg


def get_factor_bounds(cfg: Dict[str, Any], df) -> Dict[str, Tuple[float, float]]:
    """Numeric bounds for each numeric tool parameter.

    Precedence:
    1) cfg.factor_space.bounds
    2) inferred from df with 10 percent padding
    """
    bounds: Dict[str, Tuple[float, float]] = {}
    tool_params: List[str] = cfg["data"]["tool_parameters"]
    provided = cfg.get("factor_space", {}).get("bounds", {}) or {}

    categorical = set((cfg.get("factor_space", {}).get("categories", {}) or {}).keys())
    categorical |= set((cfg["data"].get("categorical_params", {}) or {}).keys())

    for p in tool_params:
        if p in categorical:
            continue
        if p in provided:
            lo, hi = provided[p]
            bounds[p] = (float(lo), float(hi))
            continue
        if df is None or p not in df.columns:
            continue
        lo = float(df[p].min())
        hi = float(df[p].max())
        span = hi - lo
        pad = 0.1 * span if span > 0 else 1.0
        bounds[p] = (lo - pad, hi + pad)

    return bounds


def get_categorical_levels(cfg: Dict[str, Any], df) -> Dict[str, List[str]]:
    """Categorical levels for each categorical tool parameter."""
    levels: Dict[str, List[str]] = {}

    provided = cfg.get("factor_space", {}).get("categories", {}) or {}
    data_cat = cfg.get("data", {}).get("categorical_params", {}) or {}

    merged = dict(data_cat)
    merged.update(provided)

    for p, vals in merged.items():
        if vals:
            levels[p] = [str(x) for x in vals]
        elif df is not None and p in df.columns:
            levels[p] = sorted([str(x) for x in df[p].dropna().unique().tolist()])

    return levels

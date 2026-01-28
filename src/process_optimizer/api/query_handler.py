from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from process_optimizer.inverse_design import InverseDesigner
from process_optimizer.storage.run_loader import RunLoader
from process_optimizer.storage.vector_store import VectorStore


def _load_training_results(run_root: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    p = None
    try:
        p = manifest.get("training_results")
    except Exception:
        p = None
    if not p:
        p = run_root / "reports" / "training_results.json"
    try:
        return json_load(Path(p))
    except Exception:
        return {}


def json_load(path: Path) -> Dict[str, Any]:
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pick_model_rank(training_results: Dict[str, Any], target: str) -> List[str]:
    """Return model rank list for a target (best first)."""
    try:
        pack = (training_results.get("targets") or {}).get(str(target)) or {}
        rank = pack.get("model_rank") or []
        if isinstance(rank, list):
            return [str(x) for x in rank]
    except Exception:
        pass
    # legacy-ish fallback
    try:
        rank = (training_results.get("model_rank") or {}).get(str(target)) or []
        if isinstance(rank, list):
            out = []
            for x in rank:
                if isinstance(x, dict) and "model" in x:
                    out.append(str(x["model"]))
                else:
                    out.append(str(x))
            return out
    except Exception:
        pass
    return []


def _is_overfit(training_results: Dict[str, Any], target: str, model_name: str) -> bool:
    try:
        pack = (training_results.get("targets") or {}).get(str(target)) or {}
        per_model = (pack.get("models") or {})
        m = per_model.get(str(model_name)) or {}
        return bool(m.get("overfit_flag", False))
    except Exception:
        return False


def _filter_models_for_target(
    loaded_models,
    *,
    target: str,
    training_results: Dict[str, Any],
    use_top: int,
    exclude_overfit: bool,
) -> List[Any]:
    """Return wrappers (ordered best-first) for a target."""

    target = str(target)
    all_wrappers = [m.wrapper for m in loaded_models if str(m.target) == target]
    if not all_wrappers:
        return []

    # If we have training results, rank models and optionally exclude overfit.
    rank = _pick_model_rank(training_results, target)
    if not rank:
        return all_wrappers[: max(1, int(use_top))]

    name_to_wrapper: Dict[str, Any] = {}
    for m in loaded_models:
        if str(m.target) != target:
            continue
        name_to_wrapper[str(m.model_name)] = m.wrapper
        # wrapper may have .name; keep that too
        wname = getattr(m.wrapper, "name", None)
        if wname is not None:
            name_to_wrapper[str(wname)] = m.wrapper

    chosen: List[Any] = []
    for model_name in rank:
        w = name_to_wrapper.get(str(model_name))
        if w is None:
            continue
        if exclude_overfit and _is_overfit(training_results, target, str(model_name)):
            continue
        chosen.append(w)
        if len(chosen) >= max(1, int(use_top)):
            break

    # If all models were excluded, fall back to the best-ranked model anyway.
    if not chosen:
        for model_name in rank:
            w = name_to_wrapper.get(str(model_name))
            if w is None:
                continue
            chosen.append(w)
            if len(chosen) >= max(1, int(use_top)):
                break

    return chosen or all_wrappers[: max(1, int(use_top))]


def _load_ood(run_root: Path, manifest: Dict[str, Any]) -> Tuple[Optional[Any], Optional[float]]:
    try:
        ood = manifest.get("ood") or {}
        model_path = ood.get("model")
        scale = ood.get("scale")
        if model_path:
            return joblib.load(model_path), (None if scale is None else float(scale))
    except Exception:
        pass

    # Legacy key
    try:
        model_path = manifest.get("ood_model")
        if model_path:
            return joblib.load(model_path), None
    except Exception:
        pass
    return None, None


def _target_scales_from_csv(csv_path: str, targets: List[str]) -> Dict[str, float]:
    scales: Dict[str, float] = {}
    try:
        df = pd.read_csv(csv_path)
        for t in targets:
            if t not in df.columns:
                continue
            y = pd.to_numeric(df[t], errors="coerce").astype(float)
            s = float(np.nanstd(y.values))
            if not np.isfinite(s) or s <= 0:
                # fallback to IQR-ish scale
                q25 = float(np.nanpercentile(y.values, 25))
                q75 = float(np.nanpercentile(y.values, 75))
                s = float(max(1e-12, q75 - q25))
            scales[str(t)] = float(s)
    except Exception:
        pass
    # Ensure no missing/invalid values.
    for t in targets:
        s = float(scales.get(str(t), 1.0))
        if not np.isfinite(s) or s <= 0:
            s = 1.0
        scales[str(t)] = float(s)
    return scales


def _retrieve_docs(run_root: Path, manifest: Dict[str, Any], query: str, top_k_docs: int) -> List[Dict[str, Any]]:
    rag_db = manifest.get("rag_db") or (run_root / "rag" / "vector_store.joblib")
    try:
        vs = VectorStore.load(rag_db)
        retrieved = vs.query(query, top_k=top_k_docs)
        return [
            {
                "doc_id": d.doc_id,
                "meta": d.meta,
                "snippet": (d.text[:800] + "..." if len(d.text) > 800 else d.text),
            }
            for d in retrieved
        ]
    except Exception:
        return []


def handle_query(
    run_root: str | Path,
    manifest: Dict[str, Any],
    target: str,
    value: float,
    question: str | None = None,
    top_k_docs: int = 6,
    *,
    fixed_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Retrieve relevant run artifacts and propose tool settings for a single target value."""

    run_root = Path(run_root)

    # Retrieval
    retrieved_docs = _retrieve_docs(run_root, manifest, question or f"{target} {value}", top_k_docs=top_k_docs)

    # Load models and preprocessor
    loader = RunLoader(run_root)
    pre = loader.load_preprocessor()
    loaded_models = loader.load_models()

    cfg = manifest.get("config", {}) or {}
    inv_cfg = (cfg.get("inverse_design") or {})

    training_results = _load_training_results(run_root, manifest)
    use_top = int(inv_cfg.get("use_top_models", inv_cfg.get("top_models", 3)))
    exclude_overfit = bool(inv_cfg.get("exclude_overfit_models", True))

    models_for_target = _filter_models_for_target(
        loaded_models,
        target=str(target),
        training_results=training_results,
        use_top=use_top,
        exclude_overfit=exclude_overfit,
    )
    if not models_for_target:
        raise ValueError(f"No models found for target '{target}' in run {manifest.get('run_id')}")

    space = manifest.get("space", {}) or {}
    bounds = space.get("bounds", {}) or {}
    categories = space.get("categorical_levels", {}) or {}

    tool_params = (cfg.get("data", {}) or {}).get("tool_parameters") or []

    lam_u = float(inv_cfg.get("lambda_uncertainty", inv_cfg.get("uncertainty_weight", 0.35)))
    lam_ood = float(inv_cfg.get("lambda_ood", inv_cfg.get("ood_weight", 0.0)))
    budget = int(inv_cfg.get("search_budget", 6000))
    top_k = int(inv_cfg.get("top_k", 10))
    method = str(inv_cfg.get("method", "random"))

    ood_model, ood_scale = _load_ood(run_root, manifest)

    inv = InverseDesigner(
        preprocessor=pre,
        tool_params=tool_params,
        bounds=bounds,
        categories=categories,
        lambda_uncertainty=lam_u,
        lambda_ood=lam_ood,
        ood_model=ood_model,
        ood_scale=ood_scale,
        diversity_enabled=bool(inv_cfg.get("diversity", {}).get("enabled", True))
        if isinstance(inv_cfg.get("diversity"), dict)
        else bool(inv_cfg.get("diversity_enabled", True)),
        diversity_min_distance=float(
            (inv_cfg.get("diversity") or {}).get("min_distance", inv_cfg.get("diversity_min_distance", 0.12))
        ),
        diversity_max_candidates=int(
            (inv_cfg.get("diversity") or {}).get("max_candidates", inv_cfg.get("diversity_max_candidates", 500))
        ),
        random_state=int((cfg.get("training") or {}).get("random_state", 42)),
    )

    sols = inv.propose(
        models_for_target,
        target_name=str(target),
        target_value=float(value),
        search_budget=budget,
        top_k=top_k,
        method=method,
        fixed_params=fixed_params,
    )

    return {
        "run_id": manifest.get("run_id"),
        "target": str(target),
        "target_value": float(value),
        "fixed_params": dict(fixed_params or {}),
        "suggestions": [asdict(s) for s in sols],
        "retrieved": retrieved_docs,
    }


def handle_query_multi(
    run_root: str | Path,
    manifest: Dict[str, Any],
    targets: Mapping[str, float],
    *,
    question: str | None = None,
    top_k_docs: int = 6,
    fixed_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Multi-target inverse design."""

    run_root = Path(run_root)
    targets = {str(k): float(v) for k, v in (targets or {}).items()}
    if not targets:
        raise ValueError("No targets specified.")

    retrieved_docs = _retrieve_docs(run_root, manifest, question or " ".join(f"{k} {v}" for k, v in targets.items()), top_k_docs=top_k_docs)

    loader = RunLoader(run_root)
    pre = loader.load_preprocessor()
    loaded_models = loader.load_models()

    cfg = manifest.get("config", {}) or {}
    inv_cfg = (cfg.get("inverse_design") or {})
    mt_cfg = (inv_cfg.get("multi_target") or {})

    training_results = _load_training_results(run_root, manifest)

    # For multi-target, default to the single best model per target.
    exclude_overfit = bool(inv_cfg.get("exclude_overfit_models", True))
    models_by_target: Dict[str, Any] = {}
    for t in targets.keys():
        wrps = _filter_models_for_target(
            loaded_models,
            target=str(t),
            training_results=training_results,
            use_top=1,
            exclude_overfit=exclude_overfit,
        )
        if wrps:
            models_by_target[str(t)] = wrps[0]

    if not models_by_target:
        raise ValueError("No models found for the requested targets in this run.")

    space = manifest.get("space", {}) or {}
    bounds = space.get("bounds", {}) or {}
    categories = space.get("categorical_levels", {}) or {}
    tool_params = (cfg.get("data", {}) or {}).get("tool_parameters") or []

    lam_u = float(inv_cfg.get("lambda_uncertainty", inv_cfg.get("uncertainty_weight", 0.35)))
    lam_ood = float(inv_cfg.get("lambda_ood", inv_cfg.get("ood_weight", 0.0)))
    budget = int(inv_cfg.get("search_budget", 6000))
    top_k = int(inv_cfg.get("top_k", 10))
    method = str(inv_cfg.get("method", "random"))

    ood_model, ood_scale = _load_ood(run_root, manifest)

    csv_path = (manifest.get("inputs", {}) or {}).get("csv")
    scales = _target_scales_from_csv(str(csv_path), list(targets.keys())) if csv_path else {t: 1.0 for t in targets.keys()}

    # Multi-target weights / tolerances from config
    weights = (mt_cfg.get("weights") or {}) if isinstance(mt_cfg.get("weights"), dict) else None
    tolerances = (mt_cfg.get("tolerances") or {}) if isinstance(mt_cfg.get("tolerances"), dict) else None

    inv = InverseDesigner(
        preprocessor=pre,
        tool_params=tool_params,
        bounds=bounds,
        categories=categories,
        lambda_uncertainty=lam_u,
        lambda_ood=lam_ood,
        ood_model=ood_model,
        ood_scale=ood_scale,
        diversity_enabled=bool(inv_cfg.get("diversity", {}).get("enabled", True))
        if isinstance(inv_cfg.get("diversity"), dict)
        else bool(inv_cfg.get("diversity_enabled", True)),
        diversity_min_distance=float(
            (inv_cfg.get("diversity") or {}).get("min_distance", inv_cfg.get("diversity_min_distance", 0.12))
        ),
        diversity_max_candidates=int(
            (inv_cfg.get("diversity") or {}).get("max_candidates", inv_cfg.get("diversity_max_candidates", 500))
        ),
        random_state=int((cfg.get("training") or {}).get("random_state", 42)),
    )

    sols = inv.propose_multi(
        models_by_target=models_by_target,
        targets=targets,
        search_budget=budget,
        top_k=top_k,
        method=method,
        fixed_params=fixed_params,
        weights=weights,
        tolerances=tolerances,
        scales=scales,
    )

    return {
        "run_id": manifest.get("run_id"),
        "targets": dict(targets),
        "fixed_params": dict(fixed_params or {}),
        "suggestions": [asdict(s) for s in sols],
        "retrieved": retrieved_docs,
    }


def handle_forward_predict(
    run_root: str | Path,
    manifest: Dict[str, Any],
    params: Mapping[str, Any],
    *,
    targets: Optional[List[str]] = None,
    question: str | None = None,
    top_k_docs: int = 6,
) -> Dict[str, Any]:
    """Forward prediction: tool parameters -> predicted properties."""

    run_root = Path(run_root)
    params = dict(params or {})

    cfg = manifest.get("config", {}) or {}
    tool_params = (cfg.get("data", {}) or {}).get("tool_parameters") or []
    all_targets = (cfg.get("data", {}) or {}).get("target_properties") or []

    if targets is None:
        targets = list(all_targets)
    else:
        targets = [t for t in targets if t in all_targets]

    if not targets:
        raise ValueError("No valid target properties specified for forward prediction.")

    retrieved_docs = _retrieve_docs(
        run_root,
        manifest,
        question
        or (
            "predict "
            + ", ".join(f"{k}={v}" for k, v in params.items())
            + " -> "
            + ", ".join(targets)
        ),
        top_k_docs=top_k_docs,
    )

    loader = RunLoader(run_root)
    pre = loader.load_preprocessor()
    loaded_models = loader.load_models()
    training_results = _load_training_results(run_root, manifest)

    inv_cfg = (cfg.get("inverse_design") or {})
    exclude_overfit = bool(inv_cfg.get("exclude_overfit_models", True))

    # Assemble one-row dataframe. Fill missing parameters conservatively.
    space = manifest.get("space", {}) or {}
    bounds = space.get("bounds", {}) or {}
    categories = space.get("categorical_levels", {}) or {}

    filled: Dict[str, Any] = dict(params)
    missing: List[str] = []
    for p in tool_params:
        if p in filled:
            continue
        missing.append(p)
        if p in bounds:
            lo, hi = bounds[p]
            filled[p] = float(lo + 0.5 * (hi - lo))
        elif p in categories and categories.get(p):
            filled[p] = str(categories[p][0])
        else:
            filled[p] = 0.0

    df_row = pd.DataFrame([{p: filled.get(p) for p in tool_params}])
    Xrow = pre.transform(df_row[tool_params])
    if hasattr(Xrow, "toarray"):
        Xrow = Xrow.toarray()
    Xrow = np.asarray(Xrow, dtype=float)

    ood_model, _ood_scale = _load_ood(run_root, manifest)
    ood_distance = None
    if ood_model is not None:
        try:
            dist, _ = ood_model.kneighbors(Xrow, n_neighbors=1, return_distance=True)
            ood_distance = float(np.asarray(dist).reshape(-1)[0])
        except Exception:
            ood_distance = None

    preds: Dict[str, Dict[str, Any]] = {}
    for t in targets:
        wrps = _filter_models_for_target(
            loaded_models,
            target=str(t),
            training_results=training_results,
            use_top=1,
            exclude_overfit=exclude_overfit,
        )
        if not wrps:
            continue
        m = wrps[0]
        mu, sd = m.predict(Xrow, return_std=True)
        mu = float(np.asarray(mu).ravel()[0])
        sdv = None if sd is None else float(np.asarray(sd).ravel()[0])
        preds[str(t)] = {"mean": mu, "std": sdv, "model": getattr(m, "name", m.__class__.__name__)}

    return {
        "run_id": manifest.get("run_id"),
        "params": {k: filled[k] for k in tool_params},
        "missing_filled": missing,
        "ood_distance": ood_distance,
        "predictions": preds,
        "retrieved": retrieved_docs,
    }


def handle_retrieve_only(
    run_root: str | Path,
    manifest: Dict[str, Any],
    question: str,
    top_k_docs: int = 6,
) -> Dict[str, Any]:
    """RAG retrieval only (no target=value)."""
    run_root = Path(run_root)
    return {
        "run_id": manifest.get("run_id"),
        "question": question,
        "retrieved": _retrieve_docs(run_root, manifest, question, top_k_docs=top_k_docs),
    }


def handle_trend(
    run_root: str | Path,
    manifest: Dict[str, Any],
    target: str,
    question: str | None = None,
    top_k_docs: int = 6,
) -> Dict[str, Any]:
    """Compute a simple, data-driven trend summary for a target."""

    run_root = Path(run_root)
    cfg = manifest.get("config", {}) or {}
    tool_params = (cfg.get("data", {}) or {}).get("tool_parameters") or []
    categorical = (cfg.get("data", {}) or {}).get("categorical_params") or {}
    cat_cols = set(categorical.keys())

    csv_path = (manifest.get("inputs", {}) or {}).get("csv")
    if not csv_path:
        raise ValueError("Manifest does not contain the input CSV path.")
    df = pd.read_csv(csv_path)

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in CSV columns")

    # Numeric correlations
    num_cols = [c for c in tool_params if c in df.columns and c not in cat_cols]
    corrs = []
    for c in num_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        y = pd.to_numeric(df[target], errors="coerce")
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 5:
            continue
        rho, p = spearmanr(x[ok], y[ok])
        corrs.append((c, float(rho), float(p)))
    corrs.sort(key=lambda t: abs(t[1]), reverse=True)

    # Categorical means
    cat_summaries = []
    for c in tool_params:
        if c in df.columns and c in cat_cols:
            grp = df.groupby(c)[target].agg(["mean", "count"]).sort_values("mean")
            rows = [(str(idx), float(r["mean"]), int(r["count"])) for idx, r in grp.iterrows() if int(r["count"]) >= 2]
            if rows:
                cat_summaries.append((c, rows[:8]))

    lines = []
    lines.append(f"Trend summary for target '{target}' (based on the measured CSV).")
    if corrs:
        lines.append("Top numeric correlations (Spearman rho, sign indicates direction):")
        for c, rho, p in corrs[:8]:
            lines.append(f"  - {c}: rho={rho:+.3f} (p={p:.2g})")
    else:
        lines.append("No usable numeric correlations found (too few finite data points).")

    if cat_summaries:
        lines.append("\nCategorical effects (mean target by category; count in parentheses):")
        for c, rows in cat_summaries:
            pretty = ", ".join([f"{k}={v:.4g} (n={n})" for k, v, n in rows])
            lines.append(f"  - {c}: {pretty}")

    retrieved_pack = handle_retrieve_only(run_root, manifest, question or f"trend {target}", top_k_docs=top_k_docs)
    return {
        "run_id": manifest.get("run_id"),
        "target": target,
        "trend_text": "\n".join(lines),
        "retrieved": retrieved_pack.get("retrieved", []),
    }

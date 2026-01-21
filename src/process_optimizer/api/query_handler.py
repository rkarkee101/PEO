from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from process_optimizer.inverse_design import InverseDesigner
from process_optimizer.storage.run_loader import RunLoader
from process_optimizer.storage.vector_store import VectorStore


def handle_query(
    run_root: str | Path,
    manifest: Dict[str, Any],
    target: str,
    value: float,
    question: str | None = None,
    top_k_docs: int = 6,
) -> Dict[str, Any]:
    """Retrieve relevant run artifacts and propose tool settings for a target value."""

    run_root = Path(run_root)

    # Load vector store for retrieval
    rag_db = manifest.get("rag_db") or (run_root / "rag" / "vector_store.joblib")
    try:
        vs = VectorStore.load(rag_db)
        retrieved = vs.query(question or f"{target} {value}", top_k=top_k_docs)
        retrieved_docs = [
            {
                "doc_id": d.doc_id,
                "meta": d.meta,
                "snippet": (d.text[:600] + "..." if len(d.text) > 600 else d.text),
            }
            for d in retrieved
        ]
    except Exception:
        retrieved_docs = []

    # Load models and preprocessor
    loader = RunLoader(run_root)
    pre = loader.load_preprocessor()
    models = loader.load_models()
    models_for_target = [m.wrapper for m in models if str(m.target) == str(target)]

    if not models_for_target:
        raise ValueError(f"No models found for target '{target}' in run {manifest.get('run_id')}")

    space = manifest.get("space", {}) or {}
    bounds = space.get("bounds", {}) or {}
    categories = space.get("categorical_levels", {}) or {}

    cfg = manifest.get("config", {}) or {}
    inv_cfg = (cfg.get("inverse_design") or {})
    lam = float(inv_cfg.get("lambda_uncertainty", inv_cfg.get("uncertainty_weight", 0.35)))
    budget = int(inv_cfg.get("search_budget", 6000))
    top_k = int(inv_cfg.get("top_k", 10))
    method = str(inv_cfg.get("method", "random"))

    tool_params = (cfg.get("data", {}) or {}).get("tool_parameters") or []

    inv = InverseDesigner(
        preprocessor=pre,
        tool_params=tool_params,
        bounds=bounds,
        categories=categories,
        lambda_uncertainty=lam,
        random_state=int((cfg.get("training") or {}).get("random_state", 42)),
    )

    sols = inv.propose(
        models_for_target,
        target_name=target,
        target_value=float(value),
        search_budget=budget,
        top_k=top_k,
        method=method,
    )

    return {
        "run_id": manifest.get("run_id"),
        "target": target,
        "target_value": float(value),
        "suggestions": [asdict(s) for s in sols],
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
    rag_db = manifest.get("rag_db") or (run_root / "rag" / "vector_store.joblib")
    try:
        vs = VectorStore.load(rag_db)
        retrieved = vs.query(question, top_k=top_k_docs)
        retrieved_docs = [
            {
                "doc_id": d.doc_id,
                "meta": d.meta,
                "snippet": (d.text[:800] + "..." if len(d.text) > 800 else d.text),
            }
            for d in retrieved
        ]
    except Exception:
        retrieved_docs = []
    return {"run_id": manifest.get("run_id"), "question": question, "retrieved": retrieved_docs}


def handle_trend(
    run_root: str | Path,
    manifest: Dict[str, Any],
    target: str,
    question: str | None = None,
    top_k_docs: int = 6,
) -> Dict[str, Any]:
    """Compute a simple, data-driven trend summary for a target.

    This is meant for questions like: "What is the general trend for sheet resistance?"
    It uses the original CSV referenced in the manifest plus the tool/target columns
    from the stored config.
    """
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
            # NOTE: `r` is a Series. `r.mean` is a method, not the aggregated value.
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

    # Attach retrieved notes too
    retrieved_pack = handle_retrieve_only(run_root, manifest, question or f"trend {target}", top_k_docs=top_k_docs)
    return {
        "run_id": manifest.get("run_id"),
        "target": target,
        "trend_text": "\n".join(lines),
        "retrieved": retrieved_pack.get("retrieved", []),
    }

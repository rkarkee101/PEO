from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ObjectiveSpec:
    name: str
    goal: str  # "minimize" | "maximize"


@dataclass
class ConstraintSpec:
    name: str
    op: str  # "<=", ">="
    value: float


_CONSTRAINT_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*(<=|>=)\s*([-+0-9.eE]+)\s*$")


def _parse_objectives(cfg: Dict[str, Any], default_targets: List[str]) -> List[ObjectiveSpec]:
    objs = cfg.get("objectives")
    if not objs:
        # Sensible fallback: first target maximize, rest minimize.
        out: List[ObjectiveSpec] = []
        for i, t in enumerate(default_targets):
            out.append(ObjectiveSpec(name=t, goal="maximize" if i == 0 else "minimize"))
        return out

    out = []
    for o in objs:
        name = str(o.get("name"))
        goal = str(o.get("goal", "minimize")).lower()
        if goal not in {"minimize", "maximize"}:
            goal = "minimize"
        out.append(ObjectiveSpec(name=name, goal=goal))
    return out


def _parse_constraints(cfg: Dict[str, Any]) -> List[ConstraintSpec]:
    constraints = cfg.get("constraints") or []
    out: List[ConstraintSpec] = []
    for c in constraints:
        if isinstance(c, str):
            m = _CONSTRAINT_RE.match(c)
            if not m:
                continue
            name, op, val = m.group(1), m.group(2), float(m.group(3))
            out.append(ConstraintSpec(name=name, op=op, value=val))
        elif isinstance(c, dict):
            name = str(c.get("name") or c.get("metric") or "")
            op = str(c.get("op") or c.get("operator") or "<=")
            val = float(c.get("value"))
            if name and op in {"<=", ">="}:
                out.append(ConstraintSpec(name=name, op=op, value=val))
    return out


def _pick_best_model_name(training_results: Dict[str, Any], target: str) -> Optional[str]:
    """Pick best model for a target using the same score used for ranking."""
    try:
        rank = training_results.get("model_rank", {}).get(target, [])
        if not rank:
            return None
        return str(rank[0].get("model"))
    except Exception:
        return None


def _safe_attach_trial(ax_client, params: Dict[str, Any]) -> int:
    res = ax_client.attach_trial(params)
    # Ax API has changed across versions. Handle common shapes.
    if isinstance(res, tuple) and len(res) >= 2:
        # (trial_index, metadata)
        return int(res[0])
    if isinstance(res, dict) and "trial_index" in res:
        return int(res["trial_index"])
    if hasattr(res, "trial_index"):
        return int(res.trial_index)
    return int(res)


def _safe_next_trial(ax_client) -> Tuple[Dict[str, Any], int]:
    res = ax_client.get_next_trial()
    if isinstance(res, tuple) and len(res) >= 2:
        return dict(res[0]), int(res[1])
    if isinstance(res, dict) and "parameters" in res and "trial_index" in res:
        return dict(res["parameters"]), int(res["trial_index"])
    # Fallback: some versions return an object.
    params = getattr(res, "parameters", None)
    tid = getattr(res, "trial_index", None)
    if params is not None and tid is not None:
        return dict(params), int(tid)
    raise RuntimeError("Unexpected AxClient.get_next_trial() return type")


def suggest_mobo(
    *,
    df: pd.DataFrame,
    tool_params: List[str],
    targets: List[str],
    space: Dict[str, Any],
    mobo_cfg: Dict[str, Any],
    reports_dir: Path,
    models_dir: Path,
    preprocessor: Any,
    training_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Run EHVI-style multi-objective BO via Ax/BoTorch and export candidate recipes.

    Returns a dict with keys:
      - suggestions_csv
      - objectives
      - constraints
    """

    try:
        from ax.service.ax_client import AxClient
        from ax.service.utils.instantiation import ObjectiveProperties
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "MOBO is enabled but Ax is not installed. Install with: pip install -e \".[mobo]\""
        ) from e

    reports_dir.mkdir(parents=True, exist_ok=True)

    bounds = (space.get("bounds") or {})
    cat_levels = (space.get("categorical_levels") or {})

    objectives = _parse_objectives(mobo_cfg, default_targets=targets)
    constraints = _parse_constraints(mobo_cfg)

    # ---- Simple convergence/status summary (observed data only) ----
    # This lets you track whether new measurements are improving the Pareto tradeoff.
    status: Dict[str, Any] = {
        "n_observations": int(len(df)),
        "n_feasible": None,
        "feasible_fraction": None,
        "best_feasible": {},
        "best_overall": {},
    }
    try:
        # Overall best (ignoring constraints)
        for o in objectives:
            if o.name not in df.columns:
                continue
            s = df[o.name].dropna()
            if len(s) == 0:
                continue
            status["best_overall"][o.name] = float(s.min()) if o.goal == "minimize" else float(s.max())

        # Feasible best (constraints satisfied)
        if constraints:
            feasible = pd.Series(True, index=df.index)
            for c in constraints:
                if c.name not in df.columns:
                    continue
                col = df[c.name]
                if c.op == "<=":
                    feasible &= col <= c.bound
                elif c.op == ">=":
                    feasible &= col >= c.bound

            n_feas = int(feasible.sum())
            status["n_feasible"] = n_feas
            status["feasible_fraction"] = float(n_feas / len(df)) if len(df) else 0.0

            if n_feas > 0:
                df_feas = df.loc[feasible]
                for o in objectives:
                    if o.name not in df_feas.columns:
                        continue
                    s = df_feas[o.name].dropna()
                    if len(s) == 0:
                        continue
                    status["best_feasible"][o.name] = float(s.min()) if o.goal == "minimize" else float(s.max())
    except Exception:
        # Keep status best-effort (never block MOBO).
        pass

    # ---- Build Ax search space ----
    parameters: List[Dict[str, Any]] = []
    for p in tool_params:
        if p in cat_levels:
            values = list(cat_levels[p])
            if not values:
                # infer from df
                values = sorted([str(v) for v in df[p].dropna().unique().tolist()])
            parameters.append({"name": p, "type": "choice", "values": values, "value_type": "str"})
        else:
            lo_hi = bounds.get(p)
            if not lo_hi:
                lo_hi = [float(df[p].min()), float(df[p].max())]
            parameters.append({"name": p, "type": "range", "bounds": [float(lo_hi[0]), float(lo_hi[1])], "value_type": "float"})

    objective_props = {}
    for o in objectives:
        objective_props[o.name] = ObjectiveProperties(minimize=(o.goal == "minimize"))

    outcome_constraints: List[str] = []
    for c in constraints:
        outcome_constraints.append(f"{c.name} {c.op} {c.value}")

    ax_client = AxClient(enforce_sequential_optimization=False)
    ax_client.create_experiment(
        name=str(mobo_cfg.get("experiment_name") or "peo_mobo"),
        parameters=parameters,
        objectives=objective_props,
        outcome_constraints=outcome_constraints or None,
    )

    # ---- Feed historical observations ----
    needed_metrics = {o.name for o in objectives} | {c.name for c in constraints}

    hist = df.dropna(subset=[*tool_params, *list(needed_metrics)]).copy()
    # Downsample if user has a lot of rows and wants faster generation.
    max_hist = int(mobo_cfg.get("max_hist_points", 500))
    if len(hist) > max_hist:
        hist = hist.sample(n=max_hist, random_state=0)

    for _, row in hist.iterrows():
        params = {}
        for p in tool_params:
            if p in cat_levels:
                params[p] = str(row[p])
            else:
                params[p] = float(row[p])
        raw_data = {m: (float(row[m]), 0.0) for m in needed_metrics}
        tid = _safe_attach_trial(ax_client, params)
        ax_client.complete_trial(trial_index=tid, raw_data=raw_data)

    n_suggestions = int(mobo_cfg.get("n_suggestions", 12))

    suggested_params: List[Dict[str, Any]] = []
    for _ in range(n_suggestions):
        params, _tid = _safe_next_trial(ax_client)
        # Normalize categorical values to str for CSV.
        for p in tool_params:
            if p in cat_levels and p in params:
                params[p] = str(params[p])
        suggested_params.append(params)

    cand_df = pd.DataFrame(suggested_params)

    # ---- Predict objectives using best single-objective models (for operator sanity-check) ----
    # Pick best model per objective based on the trained model ranking.
    pred_cols = {}
    try:
        for o in objectives:
            mname = _pick_best_model_name(training_results, o.name)
            if not mname:
                continue
            model_path = models_dir / f"{o.name}__{mname}.joblib"
            if not model_path.exists():
                continue
            import joblib

            wrapper = joblib.load(model_path)
            Xc = preprocessor.transform(cand_df[tool_params])
            yhat, ystd = wrapper.predict(Xc, return_std=True)
            pred_cols[f"pred_{o.name}"] = yhat
            pred_cols[f"std_{o.name}"] = ystd
    except Exception as e:
        logger.warning("MOBO: prediction annotation failed: %s", e)

    for k, v in pred_cols.items():
        cand_df[k] = np.asarray(v)

    out_csv = reports_dir / (mobo_cfg.get("export_filename") or "suggested_recipes_mobo.csv")
    cand_df.to_csv(out_csv, index=False)

    return {
        "suggestions_csv": str(out_csv),
        "objectives": [o.__dict__ for o in objectives],
        "constraints": [c.__dict__ for c in constraints],
        "status": status,
    }

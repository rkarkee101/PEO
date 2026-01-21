from __future__ import annotations

import inspect
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
    threshold: Optional[float] = None  # optional objective threshold


@dataclass
class ConstraintSpec:
    name: str
    op: str  # "<=", ">="
    value: float


_CONSTRAINT_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*(<=|>=)\s*([-+0-9.eE]+)\s*$")


def _parse_objectives(cfg: Dict[str, Any], default_targets: List[str]) -> List[ObjectiveSpec]:
    objs = cfg.get("objectives")
    thresholds_cfg = cfg.get("objective_thresholds") or cfg.get("thresholds") or {}

    # objective_thresholds may be:
    #  - dict: {metric_name: value, ...}
    #  - list: [{"name": "...", "value": ...}, {"metric": "...", "threshold": ...}]
    thr_map: Dict[str, float] = {}
    try:
        if isinstance(thresholds_cfg, dict):
            for k, v in thresholds_cfg.items():
                thr_map[str(k)] = float(v)
        elif isinstance(thresholds_cfg, list):
            for item in thresholds_cfg:
                if not isinstance(item, dict):
                    continue
                name = item.get("name") or item.get("metric") or item.get("metric_name")
                val = item.get("value") or item.get("threshold") or item.get("bound")
                if name is not None and val is not None:
                    thr_map[str(name)] = float(val)
    except Exception:
        thr_map = {}

    if not objs:
        # Fallback: first target maximize, rest minimize.
        out: List[ObjectiveSpec] = []
        for i, t in enumerate(default_targets):
            goal = "maximize" if i == 0 else "minimize"
            out.append(ObjectiveSpec(name=t, goal=goal, threshold=thr_map.get(t)))
        return out

    out: List[ObjectiveSpec] = []
    for o in objs:
        name = str(o.get("name"))
        goal = str(o.get("goal", "minimize")).lower()
        if goal not in {"minimize", "maximize"}:
            goal = "minimize"
        out.append(ObjectiveSpec(name=name, goal=goal, threshold=thr_map.get(name)))
    return out


def _parse_constraints(cfg: Dict[str, Any]) -> List[ConstraintSpec]:
    # Support both:
    #   mobo.constraints: ["y <= 10", ...]  (preferred)
    #   mobo.outcome_constraints: ["y <= 10", ...] (your current yaml)
    constraints = cfg.get("constraints")
    if constraints is None:
        constraints = cfg.get("outcome_constraints")
    constraints = constraints or []

    out: List[ConstraintSpec] = []
    for c in constraints:
        if isinstance(c, str):
            m = _CONSTRAINT_RE.match(c)
            if not m:
                continue
            name, op, val = m.group(1), m.group(2), float(m.group(3))
            out.append(ConstraintSpec(name=name, op=op, value=val))
        elif isinstance(c, dict):
            name = str(c.get("name") or c.get("metric") or c.get("metric_name") or "")
            op = str(c.get("op") or c.get("operator") or "<=")
            val_raw = c.get("value")
            if val_raw is None:
                val_raw = c.get("bound")
            if val_raw is None:
                continue
            try:
                val = float(val_raw)
            except Exception:
                continue
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

    if isinstance(res, tuple) and len(res) >= 1:
        a = res[0]

        if isinstance(a, (int, np.integer)):
            return int(a)

        if isinstance(a, dict) and "trial_index" in a:
            return int(a["trial_index"])

        if hasattr(a, "trial_index"):
            return int(getattr(a, "trial_index"))

        if len(res) >= 2:
            b = res[1]
            if isinstance(b, (int, np.integer)):
                return int(b)
            if isinstance(b, dict) and "trial_index" in b:
                return int(b["trial_index"])
            if hasattr(b, "trial_index"):
                return int(getattr(b, "trial_index"))

        raise RuntimeError(f"Unexpected attach_trial tuple shape: {res!r}")

    if isinstance(res, dict):
        if "trial_index" in res:
            return int(res["trial_index"])
        for k in ("data", "trial", "result", "metadata"):
            v = res.get(k)
            if isinstance(v, dict) and "trial_index" in v:
                return int(v["trial_index"])
        raise RuntimeError(f"Unexpected attach_trial dict shape: {res!r}")

    if hasattr(res, "trial_index"):
        return int(getattr(res, "trial_index"))

    if isinstance(res, (int, np.integer)):
        return int(res)

    raise RuntimeError(f"Unexpected AxClient.attach_trial() return type: {type(res)}; value={res!r}")


def _safe_next_trial(ax_client) -> Tuple[Dict[str, Any], int]:
    res = ax_client.get_next_trial()

    if isinstance(res, tuple) and len(res) >= 2:
        a, b = res[0], res[1]

        if isinstance(a, (int, np.integer)) and isinstance(b, dict):
            return dict(b), int(a)

        if isinstance(a, dict) and isinstance(b, (int, np.integer)):
            return dict(a), int(b)

        for x in (a, b):
            if isinstance(x, dict) and "parameters" in x and "trial_index" in x:
                return dict(x["parameters"]), int(x["trial_index"])

        if isinstance(a, dict) and hasattr(b, "trial_index"):
            return dict(a), int(getattr(b, "trial_index"))
        if isinstance(b, dict) and hasattr(a, "trial_index"):
            return dict(b), int(getattr(a, "trial_index"))

        raise RuntimeError(f"Unhandled get_next_trial tuple shape: {res!r}")

    if isinstance(res, dict):
        if "parameters" in res and "trial_index" in res:
            return dict(res["parameters"]), int(res["trial_index"])
        for k in ("result", "data"):
            v = res.get(k)
            if isinstance(v, dict) and "parameters" in v and "trial_index" in v:
                return dict(v["parameters"]), int(v["trial_index"])
        raise RuntimeError(f"Unexpected get_next_trial dict shape: {res!r}")

    params = getattr(res, "parameters", None)
    tid = getattr(res, "trial_index", None)
    if params is not None and tid is not None:
        return dict(params), int(tid)

    raise RuntimeError(f"Unexpected AxClient.get_next_trial() return type: {type(res)}; value={res!r}")


def _objective_properties_with_threshold(ObjectiveProperties, minimize: bool, threshold: Optional[float]):
    """
    Ax API varies by version:
      - some versions support ObjectiveProperties(minimize=..., threshold=...)
      - some don't. We detect via signature and only pass threshold if supported.
    """
    try:
        sig = inspect.signature(ObjectiveProperties)
        if threshold is not None and "threshold" in sig.parameters:
            return ObjectiveProperties(minimize=minimize, threshold=float(threshold))
    except Exception:
        pass
    return ObjectiveProperties(minimize=minimize)


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
    """
    EHVI-style MOBO via Ax/BoTorch.
    Exports candidate recipes to CSV.

    Returns dict with keys:
      - suggestions_csv
      - objectives
      - constraints
      - status
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

    # ---- Simple observed-data status summary (non-blocking) ----
    status: Dict[str, Any] = {
        "n_observations": int(len(df)),
        "n_feasible": None,
        "feasible_fraction": None,
        "best_feasible": {},
        "best_overall": {},
    }
    try:
        for o in objectives:
            if o.name not in df.columns:
                continue
            s = df[o.name].dropna()
            if len(s) == 0:
                continue
            status["best_overall"][o.name] = float(s.min()) if o.goal == "minimize" else float(s.max())

        if constraints:
            feasible = pd.Series(True, index=df.index)
            for c in constraints:
                if c.name not in df.columns:
                    continue
                col = df[c.name]
                if c.op == "<=":
                    feasible &= col <= c.value
                elif c.op == ">=":
                    feasible &= col >= c.value
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
        pass

    # ---- Build Ax search space ----
    parameters: List[Dict[str, Any]] = []
    for p in tool_params:
        if p in cat_levels:
            values = list(cat_levels[p])
            if not values:
                values = sorted([str(v) for v in df[p].dropna().unique().tolist()])
            parameters.append(
                {
                    "name": p,
                    "type": "choice",
                    "values": values,
                    "value_type": "str",
                    # silence Ax warnings by setting explicitly
                    "is_ordered": False,
                    "sort_values": False,
                }
            )
        else:
            lo_hi = bounds.get(p)
            if not lo_hi:
                lo_hi = [float(df[p].min()), float(df[p].max())]
            parameters.append(
                {
                    "name": p,
                    "type": "range",
                    "bounds": [float(lo_hi[0]), float(lo_hi[1])],
                    "value_type": "float",
                }
            )

    objective_props: Dict[str, Any] = {}
    for o in objectives:
        objective_props[o.name] = _objective_properties_with_threshold(
            ObjectiveProperties,
            minimize=(o.goal == "minimize"),
            threshold=o.threshold,
        )

    outcome_constraints: List[str] = []
    for c in constraints:
        outcome_constraints.append(f"{c.name} {c.op} {c.value}")

    ax_client = AxClient(enforce_sequential_optimization=False)

    # ---- Create experiment with best-effort support for objective thresholds ----
    create_kwargs: Dict[str, Any] = dict(
        name=str(mobo_cfg.get("experiment_name") or "peo_mobo"),
        parameters=parameters,
        objectives=objective_props,
        outcome_constraints=outcome_constraints or None,
    )

    # Some Ax versions accept objective_thresholds kwarg on create_experiment; many don't.
    # We only add it if the method signature supports it.
    try:
        sig = inspect.signature(ax_client.create_experiment)
        if "objective_thresholds" in sig.parameters:
            # Ax expects a mapping metric -> threshold (in many versions), but we keep it simple:
            # pass a dict if all thresholds are present
            thr_dict = {o.name: o.threshold for o in objectives if o.threshold is not None}
            if thr_dict:
                create_kwargs["objective_thresholds"] = thr_dict
    except Exception:
        pass

    try:
        ax_client.create_experiment(**create_kwargs)
    except TypeError as e:
        # Most common: AxClient.create_experiment() got an unexpected keyword argument 'objective_thresholds'
        if "objective_thresholds" in str(e) and "objective_thresholds" in create_kwargs:
            logger.warning(
                "AxClient.create_experiment() in your Ax version does not support `objective_thresholds`. "
                "Proceeding without thresholds (EHVI still works, but Ax may warn)."
            )
            create_kwargs.pop("objective_thresholds", None)
            ax_client.create_experiment(**create_kwargs)
        else:
            raise

    # ---- Feed historical observations ----
    needed_metrics = {o.name for o in objectives} | {c.name for c in constraints}

    hist = df.dropna(subset=[*tool_params, *list(needed_metrics)]).copy()

    max_hist = int(mobo_cfg.get("max_hist_points", 500))
    if len(hist) > max_hist:
        hist = hist.sample(n=max_hist, random_state=0)

    for _, row in hist.iterrows():
        params: Dict[str, Any] = {}
        for p in tool_params:
            if p in cat_levels:
                params[p] = str(row[p])
            else:
                params[p] = float(row[p])

        raw_data = {m: (float(row[m]), 0.0) for m in needed_metrics}
        tid = _safe_attach_trial(ax_client, params)
        ax_client.complete_trial(trial_index=tid, raw_data=raw_data)

    # ---- Generate suggestions ----
    n_suggestions = int(mobo_cfg.get("n_suggestions", 12))

    suggested_params: List[Dict[str, Any]] = []
    for _ in range(n_suggestions):
        params, _tid = _safe_next_trial(ax_client)
        for p in tool_params:
            if p in cat_levels and p in params:
                params[p] = str(params[p])
        suggested_params.append(params)

    cand_df = pd.DataFrame(suggested_params)

    # ---- Annotate with model predictions (optional sanity-check for operator) ----
    pred_cols: Dict[str, np.ndarray] = {}
    try:
        import joblib

        for o in objectives:
            mname = _pick_best_model_name(training_results, o.name)
            if not mname:
                continue
            model_path = models_dir / f"{o.name}__{mname}.joblib"
            if not model_path.exists():
                continue

            wrapper = joblib.load(model_path)
            Xc = preprocessor.transform(cand_df[tool_params])
            yhat, ystd = wrapper.predict(Xc, return_std=True)

            pred_cols[f"pred_{o.name}"] = np.asarray(yhat)
            pred_cols[f"std_{o.name}"] = np.asarray(ystd)
    except Exception as e:
        logger.warning("MOBO: prediction annotation failed: %s", e)

    for k, v in pred_cols.items():
        cand_df[k] = v

    out_csv = reports_dir / (mobo_cfg.get("export_filename") or "suggested_recipes_mobo.csv")
    cand_df.to_csv(out_csv, index=False)

    return {
        "suggestions_csv": str(out_csv),
        "objectives": [o.__dict__ for o in objectives],
        "constraints": [c.__dict__ for c in constraints],
        "status": status,
    }


# Backwards-compatible alias (older pipeline versions import this name)
suggest_mobo_ax = suggest_mobo

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from process_optimizer.config import get_categorical_levels, get_factor_bounds, validate_config
from process_optimizer.data_loader import DataLoader
from process_optimizer.doe.analyzer import DOEAnalyzer
from process_optimizer.doe.designs import DOEDesigner, FactorSpace
from process_optimizer.feature_engineering import build_doe_informed_spec
from process_optimizer.logging_utils import configure_logging
from process_optimizer.storage.run_manager import RunManager
from process_optimizer.storage.vector_store import VectorStore
from process_optimizer.training.trainer import ModelTrainer
from process_optimizer.visualization.plotter import Plotter

logger = logging.getLogger(__name__)


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base (without mutating inputs)."""
    out: Dict[str, Any] = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dict(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _summarize_training_for_rag(training_results: Dict[str, Any]) -> List[str]:
    docs: List[str] = []
    targets = training_results.get("targets", {})
    for t, td in targets.items():
        rank = td.get("model_rank", [])
        docs.append(f"Target: {t}. Ranked models by test RMSE (with overfit penalty): {rank}.")
        for m, rec in (td.get("models", {}) or {}).items():
            te = rec.get("metrics", {}).get("test", {})
            tr = rec.get("metrics", {}).get("train", {})
            gap = rec.get("metrics", {}).get("train_test_r2_gap")
            of = rec.get("metrics", {}).get("overfit_flag")
            docs.append(
                f"Model {m} for {t}: test RMSE={te.get('rmse')}, test R2={te.get('r2')}, "
                f"train R2={tr.get('r2')}, R2 gap={gap}, overfit_flag={of}."
            )
    return docs


def _run_core(
    *,
    cfg: Dict[str, Any],
    csv_path: str,
    rm: RunManager,
    run_id: str,
    paths,
    parent_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Core pipeline for an already-created run."""

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "config": cfg,
        "inputs": {"csv": str(Path(csv_path).resolve())},
        "paths": {k: str(getattr(paths, k)) for k in paths.__dataclass_fields__.keys()},
    }
    if parent_run_id:
        manifest["parent_run_id"] = parent_run_id

    # Load data
    dl = DataLoader(cfg)
    dataset = dl.load_csv(csv_path)
    pre, feature_names = dl.build_preprocessor(dataset)
    X, y_map = dl.to_xy(dataset, pre)
    joblib.dump(pre, paths.root / "preprocessor.joblib")

    manifest["dataset"] = dataset.metadata
    manifest["features"] = {"n": int(X.shape[1]), "names": feature_names}

    # Factor space
    bounds = get_factor_bounds(cfg, dataset.df)
    levels = get_categorical_levels(cfg, dataset.df)

    # Persist factor space for inverse design queries
    manifest["space"] = {
        "bounds": {k: [float(v[0]), float(v[1])] for k, v in bounds.items()},
        "categorical_levels": {k: [str(x) for x in v] for k, v in levels.items()},
    }

    designer = DOEDesigner(FactorSpace(numeric_bounds=bounds, categorical_levels=levels), random_state=cfg["training"].get("random_state", 42))

    # DOE designs
    doe_cfg = cfg.get("doe", {}) or {}
    methods = list(doe_cfg.get("methods", []))
    n_samples = int(doe_cfg.get("n_samples", 24))
    frac_res = int(doe_cfg.get("fractional_resolution", 3))

    doe_outputs: Dict[str, str] = {}
    for m in methods:
        try:
            if m == "full_factorial":
                df_d = designer.full_factorial(levels=2)
            elif m == "fractional_factorial":
                df_d = designer.fractional_factorial(resolution=frac_res)
            elif m == "plackett_burman":
                df_d = designer.plackett_burman()
            elif m == "latin_hypercube":
                df_d = designer.latin_hypercube(n_samples=n_samples)
            elif m == "central_composite":
                df_d = designer.central_composite()
            elif m == "box_behnken":
                df_d = designer.box_behnken()
            else:
                logger.warning("Unknown DOE method: %s", m)
                continue
        except Exception as e:
            logger.warning("DOE method %s failed: %s", m, e)
            continue

        out_csv = paths.doe / f"{m}.csv"
        df_d.to_csv(out_csv, index=False)
        doe_outputs[m] = str(out_csv)

    manifest["doe_designs"] = doe_outputs

    # DOE analysis on measured data
    analyzer = DOEAnalyzer(paths.reports / "doe_analysis")
    analysis_out: Dict[str, Any] = {}
    for target in dataset.target_props:
        try:
            res = analyzer.analyze(dataset.df, target=target, factors=dataset.tool_params, max_interactions=doe_cfg.get("interaction_depth", 2))
            analysis_out[target] = res
            analyzer.save_json(res, analyzer.output_dir / f"analysis_{target}.json")
            analyzer.plot_main_effects(dataset.df, target, dataset.tool_params, save_path=paths.plots / f"main_effects_{target}.png")
            analyzer.plot_pareto(res, save_path=paths.plots / f"pareto_{target}.png")
        except Exception as e:
            logger.warning("DOE analysis for %s failed: %s", target, e)

    manifest["doe_analysis"] = {t: {"r2": d.get("r2"), "r2_adj": d.get("r2_adj"), "n": d.get("n")} for t, d in analysis_out.items()}

    # DOE -> ML feature engineering specs (optional, per target)
    feature_specs: Dict[str, Any] = {}
    d2m = (cfg.get("doe_to_ml") or {})
    if bool(d2m.get("enabled", False)) and analysis_out:
        for target in dataset.target_props:
            if target not in analysis_out:
                continue
            try:
                spec = build_doe_informed_spec(cfg, feature_names, analysis_out[target])
                feature_specs[target] = spec
            except Exception as e:
                logger.warning("DOE->ML feature spec failed for %s: %s", target, e)

    manifest["doe_to_ml"] = {
        "enabled": bool(d2m.get("enabled", False)),
        "per_target": {
            t: {
                "selected_base_features": getattr(spec, "base_feature_names", []),
                "interaction_features": getattr(spec, "interaction_feature_names", []),
            }
            for t, spec in feature_specs.items()
        },
    }

    # Training
    trainer = ModelTrainer(paths.root, cfg, feature_names)
    training_results = trainer.train_all(X, y_map, feature_specs=(feature_specs or None))

    # Model plots (parity, residuals, calibration) using saved NPZ
    plotter = Plotter(paths.plots)
    for t, td in training_results.get("targets", {}).items():
        for m, rec in (td.get("models", {}) or {}).items():
            npz = rec.get("predictions", {}).get("npz")
            if not npz:
                continue
            try:
                import numpy as np

                arr = np.load(npz)
                y_true = arr["y_true"]
                y_pred = arr["y_pred"]
                y_std = arr["y_std"]
                plotter.parity(y_true, y_pred, title=f"Parity: {t} ({m})", save_name=f"parity_{t}__{m}.png")
                plotter.residuals(y_true, y_pred, title=f"Residuals: {t} ({m})", save_name=f"residuals_{t}__{m}.png")
                if np.all(np.isfinite(y_std)):
                    plotter.calibration_coverage(y_true, y_pred, y_std, title=f"Coverage: {t} ({m})", save_name=f"coverage_{t}__{m}.png")
            except Exception as e:
                logger.warning("Plotting failed for %s/%s: %s", t, m, e)

    manifest["training_results"] = str(paths.reports / "training_results.json")

    # Multi-objective Bayesian optimization (optional)
    mobo_cfg = (cfg.get("mobo") or {})
    if bool(mobo_cfg.get("enabled", False)):
        try:
            from process_optimizer.mobo.ax_ehvi import suggest_mobo

            mobo_pack = suggest_mobo(
                df=dataset.df,
                tool_params=dataset.tool_params,
                targets=dataset.target_props,
                space=manifest.get("space", {}),
                mobo_cfg=mobo_cfg,
                reports_dir=paths.reports,
                models_dir=paths.models,
                preprocessor=pre,
                training_results=training_results,
            )
            manifest["mobo"] = {
                "enabled": True,
                "suggestions_csv": str(mobo_pack.get("suggestions_csv")),
                "objectives": mobo_pack.get("objectives"),
                "constraints": mobo_pack.get("constraints"),
            }
        except Exception as e:
            logger.warning("MOBO suggestion step failed: %s", e)
            manifest["mobo"] = {"enabled": True, "error": str(e)}

    # RAG index build
    rag_cfg = cfg.get("rag", {}) or {}
    backend = rag_cfg.get("retriever", "tfidf")
    st_model = rag_cfg.get("st_model", "all-MiniLM-L6-v2")

    vs = VectorStore(paths.rag / "vector_store.joblib", backend=backend, st_model=st_model)

    docs = []
    docs.append((
        "Run summary\n" + json.dumps({"run_id": run_id, "csv": manifest["inputs"]["csv"], "targets": dataset.target_props, "tool_params": dataset.tool_params}, indent=2),
        {"doc_id": f"run_summary_{run_id}", "type": "run_summary", "run_id": run_id},
    ))

    if parent_run_id:
        docs.append((
            f"This run is an update iteration based on parent run_id={parent_run_id}. The input CSV contains the parent data plus newly appended measurements.",
            {"doc_id": f"update_note_{run_id}", "type": "update", "run_id": run_id, "parent_run_id": parent_run_id},
        ))

    # DOE designs references
    for m, path in doe_outputs.items():
        docs.append((
            f"DOE design: {m}. File: {path}. Use this file as a suggested experiment plan.",
            {"doc_id": f"doe_{m}_{run_id}", "type": "doe_plan", "method": m, "run_id": run_id, "path": path},
        ))

    # DOE analysis summary
    for t, res in analysis_out.items():
        top_effects = sorted((res.get("effects", {}) or {}).items(), key=lambda kv: kv[1].get("p", 1.0))[:5]
        txt = f"DOE analysis for target {t}: R2={res.get('r2')}, R2_adj={res.get('r2_adj')}, n={res.get('n')}. Top effects by p-value: {[(k, v.get('p')) for k, v in top_effects]}."
        docs.append((txt, {"doc_id": f"doe_analysis_{t}_{run_id}", "type": "doe_analysis", "target": t, "run_id": run_id}))

    # DOE -> ML feature engineering summary
    if manifest.get("doe_to_ml", {}).get("enabled"):
        for t, spec in feature_specs.items():
            txt = (
                f"DOE->ML feature engineering for target {t}: selected_base_features={getattr(spec, 'base_feature_names', [])}; "
                f"interaction_features={getattr(spec, 'interaction_feature_names', [])}. "
                "Models with '+doe' in their name were trained on this engineered feature space."
            )
            docs.append((txt, {"doc_id": f"doe_to_ml_{t}_{run_id}", "type": "doe_to_ml", "target": t, "run_id": run_id}))

    # Training summary
    for txt in _summarize_training_for_rag(training_results):
        docs.append((txt, {"doc_id": f"train_{hash(txt)}_{run_id}", "type": "training", "run_id": run_id}))

    # MOBO summary (if produced)
    if manifest.get("mobo", {}).get("suggestions_csv"):
        docs.append((
            f"Multi-objective Bayesian optimization (EHVI via Ax/BoTorch) generated suggested recipes: {manifest['mobo']['suggestions_csv']}.",
            {"doc_id": f"mobo_{run_id}", "type": "mobo", "run_id": run_id, "path": manifest["mobo"]["suggestions_csv"]},
        ))

    vs.add_documents(docs)

    manifest["rag_db"] = str(vs.db_path)
    rm.write_manifest(run_id, manifest)

    logger.info("Run complete. Artifacts saved under: %s", str(paths.root))
    return manifest


def run_all(config: Dict[str, Any], csv_path: str, run_name: str | None = None) -> Dict[str, Any]:
    cfg = validate_config(config)
    storage_root = Path(cfg["storage"]["root"]).resolve()
    rm = RunManager(storage_root)
    run_id, paths = rm.create_run(run_name)

    configure_logging(cfg.get("logging", {}).get("level", "INFO"), log_file=str(paths.logs / "run.log"))
    logger.info("Run id: %s", run_id)

    return _run_core(cfg=cfg, csv_path=csv_path, rm=rm, run_id=run_id, paths=paths, parent_run_id=None)


def run_update(
    *,
    parent_run_id: str,
    new_data_csv: str,
    run_name: str | None = None,
    storage_root: str | Path | None = None,
    config_override_path: str | Path | None = None,
    config_override: Optional[Dict[str, Any]] = None,
    deduplicate: bool = True,
) -> Dict[str, Any]:
    """Create a new run that retrains from scratch using parent data + new measurements.

    This is *preferred* over incremental updates (partial_fit) because it is simpler,
    avoids drift bugs, and keeps preprocessing consistent.
    """

    # Resolve storage_root.
    # Prefer explicit value; otherwise infer from the parent run path.
    rm_tmp = RunManager(Path(storage_root).resolve()) if storage_root is not None else None
    parent_root = (rm_tmp.get_run_root(parent_run_id) if rm_tmp is not None else None)
    if parent_root is None:
        # Default storage is ./storage
        rm_tmp = RunManager(Path("storage").resolve())
        parent_root = rm_tmp.get_run_root(parent_run_id)

    inferred_storage_root = parent_root.parent.parent
    rm = RunManager(inferred_storage_root)

    parent_manifest = json.loads((parent_root / "manifest.json").read_text(encoding="utf-8"))
    parent_cfg = parent_manifest.get("config") or {}

    # Load optional override file.
    file_override: Dict[str, Any] = {}
    if config_override_path is not None:
        try:
            import yaml

            file_override = yaml.safe_load(Path(config_override_path).read_text(encoding="utf-8")) or {}
        except Exception as e:
            raise ValueError(f"Failed to load config_override_path={config_override_path}: {e}")

    # Merge config override (if provided)
    cfg = _deep_merge_dict(parent_cfg, file_override)
    cfg = _deep_merge_dict(cfg, config_override or {})
    cfg = validate_config(cfg)

    run_id, paths = rm.create_run(run_name or f"update_from_{parent_run_id}")
    configure_logging(cfg.get("logging", {}).get("level", "INFO"), log_file=str(paths.logs / "run.log"))
    logger.info("Update run id: %s (parent=%s)", run_id, parent_run_id)

    # Merge CSVs and store merged copy under the new run folder for reproducibility
    parent_csv = str(parent_manifest.get("inputs", {}).get("csv"))
    if not parent_csv:
        raise ValueError("Parent manifest does not contain inputs.csv")

    data_cfg = cfg.get("data", {}) or {}
    sep = data_cfg.get("delimiter", ",")
    enc = data_cfg.get("encoding", "utf-8")

    df_parent = pd.read_csv(parent_csv, sep=sep, encoding=enc)
    df_new = pd.read_csv(new_data_csv, sep=sep, encoding=enc)
    df = pd.concat([df_parent, df_new], ignore_index=True)
    if deduplicate:
        df = df.drop_duplicates()

    merged_csv = paths.root / "merged_measurements.csv"
    df.to_csv(merged_csv, index=False)

    manifest = _run_core(cfg=cfg, csv_path=str(merged_csv), rm=rm, run_id=run_id, paths=paths, parent_run_id=parent_run_id)

    # Lightweight convergence report (observed-data deltas vs parent).
    try:
        targets = list((cfg.get("data", {}) or {}).get("target_properties", []) or [])

        def _stats(df_in: pd.DataFrame) -> Dict[str, Any]:
            out: Dict[str, Any] = {"n": int(len(df_in)), "targets": {}}
            for t in targets:
                if t not in df_in.columns:
                    continue
                s = pd.to_numeric(df_in[t], errors="coerce").dropna()
                if len(s) == 0:
                    continue
                out["targets"][t] = {
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "mean": float(s.mean()),
                    "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                }
            return out

        parent_stats = _stats(df_parent)
        updated_stats = _stats(df)

        # MOBO status is computed inside suggest_mobo (if enabled).
        parent_mobo_status = (parent_manifest.get("mobo") or {}).get("status")
        updated_mobo_status = (manifest.get("mobo") or {}).get("status")

        report: Dict[str, Any] = {
            "parent_run_id": parent_run_id,
            "updated_run_id": run_id,
            "parent": parent_stats,
            "updated": updated_stats,
            "mobo": {
                "parent": parent_mobo_status,
                "updated": updated_mobo_status,
            },
        }

        out_json = paths.reports / "convergence_update.json"
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

        # Small human-readable summary
        lines: List[str] = []
        lines.append(f"Parent run: {parent_run_id}")
        lines.append(f"Updated run: {run_id}")
        lines.append("")
        lines.append("Observed target deltas (updated - parent):")
        for t in targets:
            p = (parent_stats.get("targets") or {}).get(t)
            u = (updated_stats.get("targets") or {}).get(t)
            if not p or not u:
                continue
            lines.append(
                f"- {t}: min {u['min'] - p['min']:+.4g}, max {u['max'] - p['max']:+.4g}, mean {u['mean'] - p['mean']:+.4g}"
            )

        if updated_mobo_status is not None:
            lines.append("")
            lines.append("MOBO observed feasibility status:")
            lines.append(
                f"- feasible points: {updated_mobo_status.get('n_feasible')} / {updated_mobo_status.get('n_observations')}"
            )

        (paths.reports / "convergence_update.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to write convergence report: %s", e)

    return manifest

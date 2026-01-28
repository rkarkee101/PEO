from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from process_optimizer.config import load_config
from process_optimizer.pipeline import run_all, run_update
from process_optimizer.storage.run_manager import RunManager
from process_optimizer.api.query_handler import (
    handle_forward_predict,
    handle_query,
    handle_query_multi,
    handle_retrieve_only,
    handle_trend,
)

app = typer.Typer(add_completion=False)
console = Console()


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _maybe_number(v: str) -> Any:
    v = str(v).strip()
    if v == "":
        return v
    try:
        return float(v)
    except Exception:
        return v


def _parse_assignments(text: str) -> Dict[str, Any]:
    """Parse key=value pairs from a string.

    Supports:
      - comma/semicolon-separated: "a=1, b=2"
      - whitespace-separated: "a=1 b=2"
      - JSON object: '{"a": 1, "b": 2}'
    """
    s = (text or "").strip()
    if not s:
        return {}
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return {str(k): v for k, v in obj.items()}
        except Exception:
            pass

    # Normalize separators
    for sep in [";", "\n", "\t"]:
        s = s.replace(sep, " ")
    s = s.replace(",", " ")

    out: Dict[str, Any] = {}
    for tok in [t for t in s.split(" ") if t.strip()]:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        k = k.strip()
        if not k:
            continue
        out[str(k)] = _maybe_number(v)
    return out


def _fmt_num(x: Any, *, digits: int = 4) -> str:
    try:
        xf = float(x)
        return f"{xf:.{digits}g}"
    except Exception:
        return str(x)


def _print_suggestions(result: dict, max_rows: int = 10):
    """Print inverse-design suggestions (single or multi target)."""

    if "targets" in result:
        targets = result.get("targets", {}) or {}
        tdesc = ", ".join([f"{k}={_fmt_num(v)}" for k, v in targets.items()])
        table = Table(title=f"Suggestions for {tdesc}", box=box.SIMPLE)
        table.add_column("Rank", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("OOD", justify="right")
        table.add_column("Params")
        table.add_column("Predictions")

        for i, s in enumerate(result.get("suggestions", [])[:max_rows], start=1):
            params = s.get("params", {})
            ptxt = ", ".join([f"{k}={_fmt_num(v)}" for k, v in params.items()])
            preds = s.get("preds") or {}
            stds = s.get("stds") or {}
            pred_txt_parts = []
            for t, mu in preds.items():
                sd = stds.get(t)
                if sd is None:
                    pred_txt_parts.append(f"{t}={_fmt_num(mu)}")
                else:
                    pred_txt_parts.append(f"{t}={_fmt_num(mu)}±{_fmt_num(sd, digits=3)}")
            pred_txt = "; ".join(pred_txt_parts)
            ood = s.get("ood_distance")
            table.add_row(
                str(i),
                _fmt_num(s.get("score"), digits=3),
                ("-" if ood is None else _fmt_num(ood, digits=3)),
                ptxt,
                pred_txt,
            )
        console.print(table)
        return

    table = Table(title=f"Suggestions for {result.get('target')}={_fmt_num(result.get('target_value'))}", box=box.SIMPLE)
    table.add_column("Rank", justify="right")
    table.add_column("Model")
    table.add_column("Pred", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("OOD", justify="right")
    table.add_column("Params")

    for i, s in enumerate(result.get("suggestions", [])[:max_rows], start=1):
        params = s.get("params", {})
        ptxt = ", ".join([f"{k}={_fmt_num(v)}" for k, v in params.items()])
        table.add_row(
            str(i),
            str(s.get("model_name")),
            _fmt_num(s.get("pred")),
            ("-" if s.get("std") is None else _fmt_num(s.get("std"), digits=3)),
            _fmt_num(s.get("score"), digits=3),
            ("-" if s.get("ood_distance") is None else _fmt_num(s.get("ood_distance"), digits=3)),
            ptxt,
        )

    console.print(table)


def _print_forward_predictions(result: dict):
    preds = result.get("predictions", {}) or {}
    table = Table(title="Forward predictions", box=box.SIMPLE)
    table.add_column("Target")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Model")

    for t, rec in preds.items():
        table.add_row(
            str(t),
            _fmt_num(rec.get("mean")),
            ("-" if rec.get("std") is None else _fmt_num(rec.get("std"), digits=3)),
            str(rec.get("model")),
        )

    console.print(Panel.fit(f"OOD distance: {_fmt_num(result.get('ood_distance'), digits=3) if result.get('ood_distance') is not None else '-'}", title="Diagnostics"))
    if result.get("missing_filled"):
        console.print(
            Panel.fit(
                "Filled missing parameters with conservative defaults: " + ", ".join(map(str, result.get("missing_filled") or [])),
                title="Note",
            )
        )
    console.print(table)


def _export_suggestions_csv(path: Path, manifest: dict, result: dict) -> None:
    """Write inverse-design suggestions to a CSV template for active learning.

    The CSV includes:
      - tool parameter columns (recommended recipes)
      - blank target property columns (to be filled from experiment)
      - predicted columns (pred_* / std_*) for convenience
    """

    import numpy as np
    import pandas as pd

    cfg = manifest.get("config", {}) or {}
    tool_params = (cfg.get("data", {}) or {}).get("tool_parameters") or []
    target_props = (cfg.get("data", {}) or {}).get("target_properties") or []

    rows = []
    suggestions = result.get("suggestions", []) or []
    for rank, s in enumerate(suggestions, start=1):
        params = (s.get("params") or {}) if isinstance(s, dict) else {}
        row: Dict[str, Any] = {}

        row["suggestion_rank"] = int(rank)
        row["suggestion_score"] = s.get("score")
        row["suggestion_model"] = s.get("model_name")
        row["ood_distance"] = s.get("ood_distance")

        for p in tool_params:
            row[p] = params.get(p)

        # Placeholders for measured targets (user fills these after running experiments)
        for t in target_props:
            row.setdefault(str(t), np.nan)

        # Add predicted values
        if s.get("preds") is not None:
            for t, mu in (s.get("preds") or {}).items():
                row[f"pred_{t}"] = mu
            for t, sd in (s.get("stds") or {}).items():
                row[f"std_{t}"] = sd
        else:
            tname = result.get("target")
            if tname is not None:
                row[f"pred_{tname}"] = s.get("pred")
                row[f"std_{tname}"] = s.get("std")

        rows.append(row)

    df = pd.DataFrame(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to config.yaml"),
    data: Path = typer.Option(..., "--data", "-d", help="Path to measurement CSV"),
    name: str = typer.Option("run", "--name", "-n", help="Run name label"),
    fast: bool = typer.Option(False, "--fast", help="Fast mode (smoke tests / quick iteration)."),
):
    """Run DOE generation, DOE analysis, model training, and RAG indexing in one shot."""
    cfg = load_config(config)

    if fast:
        # Override a few knobs to keep runtime short.
        cfg = dict(cfg)
        cfg.setdefault("training", {})
        cfg["training"] = dict(cfg["training"])
        cfg["training"]["autotune"] = False
        cfg["training"].setdefault("models", ["gp", "random_forest"])
        cfg["training"]["models"] = [
            m
            for m in cfg["training"].get("models", [])
            if m in {"gp", "random_forest", "mlp", "rsm", "rsm_gp", "rsm_mlp", "doe_nn"}
        ]
        if not cfg["training"]["models"]:
            cfg["training"]["models"] = ["gp"]
        cfg["training"]["cv_folds"] = max(2, int(cfg["training"].get("cv_folds", 5)))

        cfg.setdefault("inverse_design", {})
        cfg["inverse_design"] = dict(cfg["inverse_design"])
        cfg["inverse_design"]["search_budget"] = min(int(cfg["inverse_design"].get("search_budget", 6000)), 800)
        cfg["inverse_design"]["top_k"] = min(int(cfg["inverse_design"].get("top_k", 10)), 5)

        cfg.setdefault("doe", {})
        cfg["doe"] = dict(cfg["doe"])
        cfg["doe"]["methods"] = ["latin_hypercube"]
        cfg["doe"]["n_samples"] = min(int(cfg["doe"].get("n_samples", 24)), 16)
        cfg["doe"]["interaction_depth"] = 1

    manifest = run_all(cfg, csv_path=str(data), run_name=name)
    console.print(Panel.fit(f"Run complete\nRun ID: {manifest.get('run_id')}\nArtifacts: {manifest.get('paths', {}).get('root')}", title="Done"))


@app.command()
def update(
    parent_run_id: str = typer.Option(..., "--parent-run-id", help="Parent run ID to update from"),
    new_data: Path = typer.Option(..., "--new-data", help="CSV with new measurements to append"),
    name: str = typer.Option("iter", "--name", "-n", help="New run name label"),
    config_override: Optional[Path] = typer.Option(
        None,
        "--config-override",
        help="Optional YAML to override parent run config (merged on top).",
    ),
    storage_root: Path = typer.Option(Path("storage"), "--storage-root", help="Storage root folder"),
):
    """Append new measurements to a parent run's dataset and retrain all artifacts.

    This creates a new run (keeps the parent run immutable) and retrains all models from scratch
    on the combined dataset.
    """

    manifest = run_update(
        parent_run_id=parent_run_id,
        new_data_csv=str(new_data),
        run_name=name,
        storage_root=str(storage_root),
        config_override_path=str(config_override) if config_override else None,
    )
    console.print(
        Panel.fit(
            f"Update complete\nRun ID: {manifest.get('run_id')}\nParent: {manifest.get('parent_run_id')}\nArtifacts: {manifest.get('paths', {}).get('root')}",
            title="Done",
        )
    )


@app.command()
def list_runs(
    storage_root: Path = typer.Option(Path("storage"), "--storage-root", help="Storage root folder"),
    limit: int = typer.Option(20, "--limit", help="Max runs to show"),
):
    """List available runs."""
    rm = RunManager(storage_root)
    runs = rm.list_runs()[: int(limit)]
    if not runs:
        console.print("No runs found.")
        raise typer.Exit(code=0)

    table = Table(title="Runs", box=box.SIMPLE)
    table.add_column("Run ID")
    for r in runs:
        table.add_row(r)
    console.print(table)


@app.command()
def query(
    target: str = typer.Option(..., "--target", "-t", help="Target property name"),
    value: float = typer.Option(..., "--value", "-v", help="Desired target value"),
    question: str = typer.Option("", "--question", "-q", help="Optional free text query for retrieval"),
    fixed: str = typer.Option(
        "",
        "--fixed",
        help="Optional fixed tool parameters like: 'gas=Ar, pressure_mTorr=10'.",
    ),
    export: Optional[Path] = typer.Option(
        None,
        "--export",
        help="Optional path to write a CSV template for running the suggested experiments (for active learning).",
    ),
    run_id: str = typer.Option("", "--run-id", help="Run ID. If blank, uses most recent run."),
    storage_root: Path = typer.Option(Path("storage"), "--storage-root", help="Storage root folder"),
    top_k_docs: int = typer.Option(6, "--top-k-docs", help="Docs to retrieve"),
):
    """Retrieve run knowledge and propose tool parameters for a target."""
    rm = RunManager(storage_root)
    rid = run_id or (rm.list_runs()[0] if rm.list_runs() else "")
    if not rid:
        raise typer.BadParameter("No runs found. Run 'peo run' first.")

    run_root = rm.get_run_root(rid)
    manifest_path = run_root / "manifest.json"
    if not manifest_path.exists():
        raise typer.BadParameter(f"manifest.json not found for run {rid}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    fixed_params = _parse_assignments(fixed)
    result = handle_query(
        run_root,
        manifest,
        target=target,
        value=value,
        question=question or None,
        top_k_docs=top_k_docs,
        fixed_params=fixed_params if fixed_params else None,
    )
    console.print(Panel.fit(f"Run ID: {rid}", title="Query"))
    _print_suggestions(result)

    if export is not None:
        try:
            _export_suggestions_csv(export, manifest, result)
            console.print(Panel.fit(f"Wrote: {export}", title="Export"))
        except Exception as e:
            console.print(Panel.fit(f"Failed to export CSV: {e}", title="Export"))

    # Also print retrieved snippets
    retrieved = result.get("retrieved", [])
    if retrieved:
        console.print(Panel.fit("Retrieved notes used for context:", title="RAG"))
        for d in retrieved[:top_k_docs]:
            console.print(Panel(d.get("snippet", ""), title=str(d.get("doc_id"))))


@app.command(name="query-multi")
def query_multi(
    targets: str = typer.Option(
        ...,
        "--targets",
        "-t",
        help="Targets as 'y1=..., y2=...' (comma/space separated) or a JSON dict.",
    ),
    question: str = typer.Option("", "--question", "-q", help="Optional free text query for retrieval"),
    fixed: str = typer.Option(
        "",
        "--fixed",
        help="Optional fixed tool parameters like: 'gas=Ar, pressure_mTorr=10'.",
    ),
    export: Optional[Path] = typer.Option(
        None,
        "--export",
        help="Optional path to write a CSV template for running the suggested experiments (for active learning).",
    ),
    run_id: str = typer.Option("", "--run-id", help="Run ID. If blank, uses most recent run."),
    storage_root: Path = typer.Option(Path("storage"), "--storage-root", help="Storage root folder"),
    top_k_docs: int = typer.Option(6, "--top-k-docs", help="Docs to retrieve"),
):
    """Retrieve run knowledge and propose tool parameters for multiple target values."""

    rm = RunManager(storage_root)
    rid = run_id or (rm.list_runs()[0] if rm.list_runs() else "")
    if not rid:
        raise typer.BadParameter("No runs found. Run 'peo run' first.")

    run_root = rm.get_run_root(rid)
    manifest_path = run_root / "manifest.json"
    if not manifest_path.exists():
        raise typer.BadParameter(f"manifest.json not found for run {rid}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    targets_map = _parse_assignments(targets)
    if not targets_map:
        raise typer.BadParameter("No targets parsed. Provide like: --targets 'y1=1, y2=2'")

    fixed_params = _parse_assignments(fixed)

    result = handle_query_multi(
        run_root,
        manifest,
        targets=targets_map,
        question=question or None,
        top_k_docs=top_k_docs,
        fixed_params=fixed_params if fixed_params else None,
    )

    console.print(Panel.fit(f"Run ID: {rid}", title="Query (multi)"))
    _print_suggestions(result)

    if export is not None:
        try:
            _export_suggestions_csv(export, manifest, result)
            console.print(Panel.fit(f"Wrote: {export}", title="Export"))
        except Exception as e:
            console.print(Panel.fit(f"Failed to export CSV: {e}", title="Export"))

    retrieved = result.get("retrieved", [])
    if retrieved:
        console.print(Panel.fit("Retrieved notes used for context:", title="RAG"))
        for d in retrieved[:top_k_docs]:
            console.print(Panel(d.get("snippet", ""), title=str(d.get("doc_id"))))


@app.command()
def predict(
    params: str = typer.Option(
        ...,
        "--params",
        "-p",
        help="Tool parameters as 'x1=..., x2=...' (comma/space separated) or a JSON dict.",
    ),
    targets: str = typer.Option(
        "",
        "--targets",
        help="Optional comma-separated list of target properties to predict (default: all).",
    ),
    question: str = typer.Option("", "--question", "-q", help="Optional free text query for retrieval"),
    run_id: str = typer.Option("", "--run-id", help="Run ID. If blank, uses most recent run."),
    storage_root: Path = typer.Option(Path("storage"), "--storage-root", help="Storage root folder"),
    top_k_docs: int = typer.Option(6, "--top-k-docs", help="Docs to retrieve"),
):
    """Forward prediction: tool parameters -> predicted properties."""

    rm = RunManager(storage_root)
    rid = run_id or (rm.list_runs()[0] if rm.list_runs() else "")
    if not rid:
        raise typer.BadParameter("No runs found. Run 'peo run' first.")

    run_root = rm.get_run_root(rid)
    manifest_path = run_root / "manifest.json"
    if not manifest_path.exists():
        raise typer.BadParameter(f"manifest.json not found for run {rid}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    params_map = _parse_assignments(params)
    if not params_map:
        raise typer.BadParameter("No parameters parsed. Provide like: --params 'temp_C=400, pressure_Torr=1.2'")

    tgt_list = [t.strip() for t in (targets or "").split(",") if t.strip()] or None

    result = handle_forward_predict(
        run_root,
        manifest,
        params=params_map,
        targets=tgt_list,
        question=question or None,
        top_k_docs=top_k_docs,
    )
    console.print(Panel.fit(f"Run ID: {rid}", title="Predict"))
    _print_forward_predictions(result)

    retrieved = result.get("retrieved", [])
    if retrieved:
        console.print(Panel.fit("Retrieved notes used for context:", title="RAG"))
        for d in retrieved[:top_k_docs]:
            console.print(Panel(d.get("snippet", ""), title=str(d.get("doc_id"))))


@app.command()
def trend(
    target: str = typer.Option(..., "--target", "-t", help="Target property name"),
    question: str = typer.Option("", "--question", "-q", help="Optional free text query for retrieval"),
    run_id: str = typer.Option("", "--run-id", help="Run ID. If blank, uses most recent run."),
    storage_root: Path = typer.Option(Path("storage"), "--storage-root", help="Storage root folder"),
    top_k_docs: int = typer.Option(6, "--top-k-docs", help="Docs to retrieve"),
):
    """Show a data-driven trend summary for a target property."""
    rm = RunManager(storage_root)
    rid = run_id or (rm.list_runs()[0] if rm.list_runs() else "")
    if not rid:
        raise typer.BadParameter("No runs found. Run 'peo run' first.")

    run_root = rm.get_run_root(rid)
    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    result = handle_trend(run_root, manifest, target=target, question=question or None, top_k_docs=top_k_docs)
    console.print(Panel(result.get("trend_text", ""), title=f"Trend: {target}"))
    retrieved = result.get("retrieved", [])
    if retrieved:
        console.print(Panel.fit("Retrieved notes:", title="RAG"))
        for d in retrieved[:top_k_docs]:
            console.print(Panel(d.get("snippet", ""), title=str(d.get("doc_id"))))


@app.command()
def chat(
    run_id: str = typer.Option("", "--run-id", help="Run ID. If blank, uses most recent run."),
    storage_root: Path = typer.Option(Path("storage"), "--storage-root", help="Storage root folder"),
):
    """Interactive prompt. Type 'exit' to quit.

    Examples:
      # Inverse design (single target)
      thickness_nm=350

      # Inverse design (multi-target)
      thickness_nm=350, sheet_resistance_ohm=12

      # Forward prediction
      temperature_C=200, pressure_mTorr=10, power_W=150, flow_sccm=40, gas=Ar

      # Inverse design with fixed parameters
      thickness_nm=350, gas=O2
    """
    rm = RunManager(storage_root)
    rid = run_id or (rm.list_runs()[0] if rm.list_runs() else "")
    if not rid:
        raise typer.BadParameter("No runs found. Run 'peo run' first.")

    run_root = rm.get_run_root(rid)
    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    cfg = manifest.get("config", {}) or {}
    tool_params = (cfg.get("data", {}) or {}).get("tool_parameters") or []
    target_props = (cfg.get("data", {}) or {}).get("target_properties") or []

    help_text = (
        f"Interactive chat for run {rid}\n"
        "\nInverse design:  thickness_nm=350"
        "\nMulti-target:    thickness_nm=350, sheet_resistance_ohm=12"
        "\nForward predict: temperature_C=200, pressure_mTorr=10, power_W=150, flow_sccm=40, gas=Ar"
        "\nMixed (fixed):   thickness_nm=350, gas=O2"
        "\n\nType 'trend <target>' for a data-driven trend summary. Type free text for retrieval-only."
        "\nType 'exit' to quit."
    )
    console.print(Panel.fit(help_text, title="Chat"))

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.lower() in {"exit", "quit"}:
            break
        if line.lower().strip() in {"help", "?"}:
            console.print(Panel.fit(help_text, title="Chat"))
            continue

        if "=" not in line:
            # Retrieval-only mode or trend query.
            targets = target_props

            # If user asked for a trend, try to infer target name.
            if "trend" in line.lower():
                inferred = None
                ln = _norm(line)
                for tname in targets:
                    if _norm(tname) in ln or _norm(tname).replace("ohm", "") in ln:
                        inferred = tname
                        break
                # fallback: try matching by stripping underscores
                if inferred is None and targets:
                    for tname in targets:
                        if _norm(tname.replace("_", "")) in ln:
                            inferred = tname
                            break
                if inferred is not None:
                    pack = handle_trend(run_root, manifest, target=str(inferred), question=line, top_k_docs=5)
                    console.print(Panel(pack.get("trend_text", ""), title=f"Trend: {inferred}"))
                    for d in pack.get("retrieved", [])[:5]:
                        console.print(Panel(d.get("snippet", ""), title=str(d.get("doc_id"))))
                else:
                    pack = handle_retrieve_only(run_root, manifest, question=line, top_k_docs=5)
                    console.print(Panel.fit("Retrieved notes:", title="RAG"))
                    for d in pack.get("retrieved", [])[:5]:
                        console.print(Panel(d.get("snippet", ""), title=str(d.get("doc_id"))))
                continue

            # Generic retrieval-only
            pack = handle_retrieve_only(run_root, manifest, question=line, top_k_docs=5)
            console.print(Panel.fit("Retrieved notes:", title="RAG"))
            for d in pack.get("retrieved", [])[:5]:
                console.print(Panel(d.get("snippet", ""), title=str(d.get("doc_id"))))
            continue
        assigns = _parse_assignments(line)
        if not assigns:
            console.print("Could not parse key=value assignments. Type 'help' for examples.")
            continue

        keys = set(assigns.keys())
        tool_set = set(map(str, tool_params))
        tgt_set = set(map(str, target_props))

        fixed_params = {k: v for k, v in assigns.items() if k in tool_set}
        targets_map = {k: v for k, v in assigns.items() if k in tgt_set}
        unknown = sorted([k for k in keys if k not in tool_set and k not in tgt_set])
        if unknown:
            console.print(
                Panel.fit(
                    "Unknown keys: " + ", ".join(unknown) + "\n\n"
                    "Known tool parameters: " + ", ".join(tool_params) + "\n"
                    "Known target properties: " + ", ".join(target_props),
                    title="Parse error",
                )
            )
            continue

        # Forward-only
        if fixed_params and not targets_map:
            pack = handle_forward_predict(run_root, manifest, params=fixed_params, targets=None, question=line, top_k_docs=5)
            _print_forward_predictions(pack)
            continue

        # Inverse-only or mixed (inverse with fixed parameters)
        if targets_map:
            if len(targets_map) == 1:
                (tname, tval), = list(targets_map.items())
                try:
                    tval_f = float(tval)
                except Exception:
                    console.print("Target values must be numeric.")
                    continue
                pack = handle_query(
                    run_root,
                    manifest,
                    target=str(tname),
                    value=float(tval_f),
                    question=line,
                    top_k_docs=5,
                    fixed_params=fixed_params if fixed_params else None,
                )
                _print_suggestions(pack)
            else:
                # Multi-target
                try:
                    _ = {str(k): float(v) for k, v in targets_map.items()}
                except Exception:
                    console.print("Target values must be numeric.")
                    continue
                pack = handle_query_multi(
                    run_root,
                    manifest,
                    targets=targets_map,
                    question=line,
                    top_k_docs=5,
                    fixed_params=fixed_params if fixed_params else None,
                )
                _print_suggestions(pack)
            continue

        console.print("No target properties found in the assignment. Type 'help' for examples.")


def main():
    app()


if __name__ == "__main__":
    main()

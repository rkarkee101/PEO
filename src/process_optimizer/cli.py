from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from process_optimizer.config import load_config
from process_optimizer.pipeline import run_all, run_update
from process_optimizer.storage.run_manager import RunManager
from process_optimizer.api.query_handler import handle_query, handle_retrieve_only, handle_trend

app = typer.Typer(add_completion=False)
console = Console()


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _print_suggestions(result: dict, max_rows: int = 10):
    table = Table(title=f"Suggestions for {result.get('target')}={result.get('target_value')}", box=box.SIMPLE)
    table.add_column("Rank", justify="right")
    table.add_column("Model")
    table.add_column("Pred", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Params")

    for i, s in enumerate(result.get("suggestions", [])[:max_rows], start=1):
        params = s.get("params", {})
        ptxt = ", ".join([f"{k}={v}" for k, v in params.items()])
        table.add_row(
            str(i),
            str(s.get("model_name")),
            f"{s.get('pred'):.4g}",
            (f"{s.get('std'):.3g}" if s.get("std") is not None else "-"),
            f"{s.get('score'):.3g}",
            ptxt,
        )

    console.print(table)


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
        cfg["training"]["models"] = [m for m in cfg["training"].get("models", []) if m in {"gp", "random_forest", "mlp"}]
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

    result = handle_query(run_root, manifest, target=target, value=value, question=question or None, top_k_docs=top_k_docs)
    console.print(Panel.fit(f"Run ID: {rid}", title="Query"))
    _print_suggestions(result)

    # Also print retrieved snippets
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

    Example:
      sheet_resistance=12
      thickness=350
    """
    rm = RunManager(storage_root)
    rid = run_id or (rm.list_runs()[0] if rm.list_runs() else "")
    if not rid:
        raise typer.BadParameter("No runs found. Run 'peo run' first.")

    run_root = rm.get_run_root(rid)
    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    console.print(Panel.fit(f"Interactive chat for run {rid}\nEnter like: sheet_resistance=12\nType 'exit' to quit", title="Chat"))

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.lower() in {"exit", "quit"}:
            break
        if "=" not in line:
            # Retrieval-only mode or trend query.
            cfg = manifest.get("config", {}) or {}
            targets = (cfg.get("data", {}) or {}).get("target_properties") or []

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
        t, v = line.split("=", 1)
        try:
            val = float(v.strip())
        except ValueError:
            console.print("Value must be a number")
            continue
        result = handle_query(run_root, manifest, target=t.strip(), value=val, question=line, top_k_docs=5)
        _print_suggestions(result)


def main():
    app()


if __name__ == "__main__":
    main()

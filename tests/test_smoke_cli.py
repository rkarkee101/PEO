from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _make_deposition_small(n: int = 80, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gas = rng.choice(["Ar", "O2", "N2"], size=n, p=[0.45, 0.35, 0.20])

    temperature_C = rng.uniform(110, 260, size=n)
    pressure_mTorr = rng.uniform(2, 18, size=n)
    power_W = rng.uniform(60, 220, size=n)
    flow_sccm = rng.uniform(8, 55, size=n)

    gas_thick = {"Ar": 0.0, "O2": 15.0, "N2": -8.0}
    gas_res = {"Ar": 0.0, "O2": -1.6, "N2": 2.3}

    thickness_nm = (
        0.85 * power_W
        + 0.22 * temperature_C
        - 1.8 * pressure_mTorr
        + 12.0 * np.log(flow_sccm)
        + 0.0035 * power_W * (temperature_C - 170)
        + np.vectorize(gas_thick.get)(gas)
        + rng.normal(0, 10.0, size=n)
    )
    thickness_nm = np.clip(thickness_nm, 30, None)

    sheet_resistance_ohm = (
        2100.0 / (thickness_nm + 25.0)
        + 0.075 * pressure_mTorr
        + 0.0012 * (temperature_C - 175.0) ** 2
        + np.vectorize(gas_res.get)(gas)
        + rng.normal(0, 0.65, size=n)
    )

    return pd.DataFrame(
        {
            "temperature_C": temperature_C,
            "pressure_mTorr": pressure_mTorr,
            "power_W": power_W,
            "flow_sccm": flow_sccm,
            "gas": gas,
            "thickness_nm": thickness_nm,
            "sheet_resistance_ohm": sheet_resistance_ohm,
        }
    )


def test_cli_run_and_query_smoke(tmp_path: Path) -> None:
    # Arrange
    csv_path = tmp_path / "data.csv"
    df = _make_deposition_small()
    df.to_csv(csv_path, index=False)

    storage_root = tmp_path / "storage"

    cfg = {
        "project": {"name": "smoke"},
        "storage": {"root": str(storage_root)},
        "logging": {"level": "WARNING"},
        "data": {
            "delimiter": ",",
            "encoding": "utf-8",
            "tool_parameters": ["temperature_C", "pressure_mTorr", "power_W", "flow_sccm", "gas"],
            "target_properties": ["thickness_nm", "sheet_resistance_ohm"],
            "categorical_params": {"gas": ["Ar", "O2", "N2"]},
        },
        "doe": {"methods": ["latin_hypercube"], "n_samples": 16, "interaction_depth": 1},
        "training": {"models": ["gp"], "autotune": False, "test_size": 0.25, "cv_folds": 3, "random_state": 3},
        "inverse_design": {"search_budget": 300, "top_k": 5, "lambda_uncertainty": 0.35},
        "rag": {"retriever": "tfidf", "top_k": 4},
    }

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # Act: run
    cmd = [
        sys.executable,
        "-m",
        "process_optimizer.cli",
        "run",
        "--config",
        str(cfg_path),
        "--data",
        str(csv_path),
        "--name",
        "smoke",
        "--fast",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr

    # Find run_id
    runs_dir = storage_root / "runs"
    run_ids = sorted([d.name for d in runs_dir.iterdir() if d.is_dir()], reverse=True)
    assert run_ids, "No run folders created"
    run_id = run_ids[0]

    # Assert manifest and vector store exist
    manifest_path = runs_dir / run_id / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    rag_db = Path(manifest["rag_db"])
    assert rag_db.exists()

    # Act: query
    cmd2 = [
        sys.executable,
        "-m",
        "process_optimizer.cli",
        "query",
        "--run-id",
        run_id,
        "--storage-root",
        str(storage_root),
        "--target",
        "sheet_resistance_ohm",
        "--value",
        "12",
        "--question",
        "sheet resistance 12",
    ]
    p2 = subprocess.run(cmd2, capture_output=True, text=True)
    assert p2.returncode == 0, p2.stderr
    assert "Suggestions" in p2.stdout or "Candidate tool settings" in p2.stdout

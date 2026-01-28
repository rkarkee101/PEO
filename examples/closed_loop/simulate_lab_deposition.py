"""Simulate lab measurements for the synthetic deposition example.

This is a convenience script to demo the *closed-loop* workflow without
requiring real lab experiments.

It reads a CSV containing recommended recipes (tool parameters) and writes a
CSV of new measurements by evaluating the same synthetic ground-truth function
used by `examples/generate_dummy_data.py`.

Usage:
  python examples/closed_loop/simulate_lab_deposition.py --in recommended.csv --out new_measurements.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _simulate_deposition(df: pd.DataFrame, *, seed: int, noise_thickness: float, noise_res: float) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))

    # Defensive parsing
    def fcol(name: str, default: float) -> np.ndarray:
        if name not in df.columns:
            return np.full(len(df), float(default), dtype=float)
        return pd.to_numeric(df[name], errors="coerce").astype(float).fillna(float(default)).to_numpy()

    gas = None
    if "gas" in df.columns:
        gas = df["gas"].astype(str).fillna("Ar").to_numpy(dtype=object)
    else:
        gas = np.array(["Ar"] * len(df), dtype=object)

    temperature_C = fcol("temperature_C", 180.0)
    pressure_mTorr = fcol("pressure_mTorr", 10.0)
    power_W = fcol("power_W", 150.0)
    flow_sccm = fcol("flow_sccm", 35.0)

    # Same synthetic mapping as examples/generate_dummy_data.py
    gas_thick = {"Ar": 0.0, "O2": 15.0, "N2": -8.0}
    gas_res = {"Ar": 0.0, "O2": -1.6, "N2": 2.3}

    gt_thick = np.vectorize(lambda g: float(gas_thick.get(str(g), 0.0)))(gas)
    gt_res = np.vectorize(lambda g: float(gas_res.get(str(g), 0.0)))(gas)

    thickness_nm = (
        0.85 * power_W
        + 0.22 * temperature_C
        - 1.8 * pressure_mTorr
        + 12.0 * np.log(np.maximum(flow_sccm, 1e-9))
        + 0.0035 * power_W * (temperature_C - 170.0)
        + gt_thick
        + rng.normal(0.0, float(noise_thickness), size=len(df))
    )
    thickness_nm = np.clip(thickness_nm, 30.0, None)

    sheet_resistance_ohm = (
        2100.0 / (thickness_nm + 25.0)
        + 0.075 * pressure_mTorr
        + 0.0012 * (temperature_C - 175.0) ** 2
        + gt_res
        + rng.normal(0.0, float(noise_res), size=len(df))
    )

    out = df.copy()
    out["thickness_nm"] = thickness_nm
    out["sheet_resistance_ohm"] = sheet_resistance_ohm
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True, help="Input CSV with tool parameters")
    p.add_argument("--out", dest="out_path", required=True, help="Output CSV with simulated measurements")
    p.add_argument("--seed", type=int, default=123, help="Random seed")
    p.add_argument("--noise-thickness", type=float, default=10.0, help="Thickness noise (nm)")
    p.add_argument("--noise-res", type=float, default=0.65, help="Sheet resistance noise (ohm)")
    args = p.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    df = pd.read_csv(in_path)
    df2 = _simulate_deposition(df, seed=args.seed, noise_thickness=args.noise_thickness, noise_res=args.noise_res)

    # Keep only the columns expected by the deposition example.
    keep = ["temperature_C", "pressure_mTorr", "power_W", "flow_sccm", "gas", "thickness_nm", "sheet_resistance_ohm"]
    keep = [c for c in keep if c in df2.columns]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df2[keep].to_csv(out_path, index=False)
    print(f"Wrote simulated measurements: {out_path}")


if __name__ == "__main__":
    main()

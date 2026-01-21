"""Generate synthetic process-engineering measurement CSVs.

The goal is not physical realism. The goal is a clean, repeatable dataset that:
- has a meaningful mapping from tool parameters to target properties
- contains nonlinearity and mild interactions
- includes a categorical factor

This makes it easy to test DOE, modeling, inverse design, and RAG.

Usage:
  python examples/generate_dummy_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def make_deposition(n: int = 220, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gas = rng.choice(["Ar", "O2", "N2"], size=n, p=[0.45, 0.35, 0.20])

    temperature_C = rng.uniform(110, 260, size=n)
    pressure_mTorr = rng.uniform(2, 18, size=n)
    power_W = rng.uniform(60, 220, size=n)
    flow_sccm = rng.uniform(8, 55, size=n)

    gas_thick = {"Ar": 0.0, "O2": 15.0, "N2": -8.0}
    gas_res = {"Ar": 0.0, "O2": -1.6, "N2": 2.3}

    # Thickness: mostly power + temperature, weaker pressure, mild interaction
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

    # Sheet resistance: decreases with thickness; depends on pressure and temperature
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


def make_etch(n: int = 220, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    chemistry = rng.choice(["CF4", "SF6", "Cl2"], size=n, p=[0.5, 0.3, 0.2])

    rf_power_W = rng.uniform(50, 450, size=n)
    pressure_mTorr = rng.uniform(4, 40, size=n)
    bias_V = rng.uniform(50, 300, size=n)
    flow_sccm = rng.uniform(10, 120, size=n)

    chem_rate = {"CF4": 1.0, "SF6": 1.25, "Cl2": 0.85}
    chem_sel = {"CF4": 1.0, "SF6": 0.92, "Cl2": 1.10}

    etch_rate_nm_min = (
        0.24 * rf_power_W
        + 0.12 * bias_V
        - 0.55 * pressure_mTorr
        + 3.6 * np.sqrt(flow_sccm)
        + 0.002 * rf_power_W * (bias_V - 120)
        + 18.0 * np.vectorize(chem_rate.get)(chemistry)
        + rng.normal(0, 12.0, size=n)
    )

    etch_rate_nm_min = np.clip(etch_rate_nm_min, 5, None)

    selectivity = (
        0.55
        + 0.0018 * rf_power_W
        - 0.0030 * pressure_mTorr
        + 0.0020 * (flow_sccm / 20.0)
        + 0.20 * np.vectorize(chem_sel.get)(chemistry)
        - 0.0000012 * (bias_V - 170) ** 2
        + rng.normal(0, 0.06, size=n)
    )

    return pd.DataFrame(
        {
            "rf_power_W": rf_power_W,
            "pressure_mTorr": pressure_mTorr,
            "bias_V": bias_V,
            "flow_sccm": flow_sccm,
            "chemistry": chemistry,
            "etch_rate_nm_min": etch_rate_nm_min,
            "selectivity": selectivity,
        }
    )


def make_cvd(n: int = 240, seed: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    precursor = rng.choice(["TEOS", "TDMAT"], size=n, p=[0.65, 0.35])

    temp_C = rng.uniform(250, 520, size=n)
    pressure_Torr = rng.uniform(0.4, 3.0, size=n)
    precursor_flow_sccm = rng.uniform(5, 60, size=n)
    carrier_flow_sccm = rng.uniform(50, 250, size=n)
    time_min = rng.uniform(5, 40, size=n)

    prec_factor = {"TEOS": 1.0, "TDMAT": 0.82}

    thickness_nm = (
        0.18 * temp_C
        + 12.0 * np.log(precursor_flow_sccm)
        + 0.06 * carrier_flow_sccm
        + 7.0 * time_min
        - 35.0 * pressure_Torr
        + 0.04 * time_min * np.sqrt(precursor_flow_sccm)
        + 25.0 * np.vectorize(prec_factor.get)(precursor)
        + rng.normal(0, 12.0, size=n)
    )

    thickness_nm = np.clip(thickness_nm, 20, None)

    prec_bias = np.where(precursor == "TEOS", 2.2, 1.0)
    transmittance_pct = (
        92.0
        - 0.035 * thickness_nm
        - 0.022 * (temp_C - 380)
        + 1.2 * np.log(carrier_flow_sccm)
        + prec_bias
        + rng.normal(0, 1.0, size=n)
    )

    transmittance_pct = np.clip(transmittance_pct, 10, 98)

    return pd.DataFrame(
        {
            "temp_C": temp_C,
            "pressure_Torr": pressure_Torr,
            "precursor_flow_sccm": precursor_flow_sccm,
            "carrier_flow_sccm": carrier_flow_sccm,
            "time_min": time_min,
            "precursor": precursor,
            "thickness_nm": thickness_nm,
            "transmittance_pct": transmittance_pct,
        }
    )


def main() -> None:
    root = Path(__file__).resolve().parent

    _write(make_deposition(), root / "deposition" / "sample_data.csv")
    _write(make_etch(), root / "etch" / "sample_data.csv")
    _write(make_cvd(), root / "cvd" / "sample_data.csv")

    # Small dataset for smoke tests
    _write(make_deposition(n=80, seed=123), root / "deposition" / "sample_data_small.csv")

    print("Wrote example CSVs under examples/")


if __name__ == "__main__":
    main()

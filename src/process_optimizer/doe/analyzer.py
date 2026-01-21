from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


def _safe_term(col: str) -> str:
    return f"Q('{col}')"


class DOEAnalyzer:
    """DOE analysis on measured data.

    Provides ANOVA and effect proxies and generates plots.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, df: pd.DataFrame, target: str, factors: List[str], max_interactions: int = 10) -> Dict[str, Any]:
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found")
        for f in factors:
            if f not in df.columns:
                raise ValueError(f"Factor '{f}' not found")

        data = df[factors + [target]].dropna().copy()
        data = data.rename(columns={target: "__response__"})

        rhs = " + ".join([_safe_term(c) for c in factors])
        formula = f"__response__ ~ {rhs}"
        model = ols(formula, data=data).fit()
        anova = sm.stats.anova_lm(model, typ=2)

        effects: Dict[str, Any] = {}
        for f in factors:
            if f in anova.index:
                effects[f] = {
                    "sum_sq": float(anova.loc[f, "sum_sq"]),
                    "F": float(anova.loc[f, "F"]),
                    "p": float(anova.loc[f, "PR(>F)"]),
                }

        std_effects: Dict[str, float] = {}
        for name, tval in model.tvalues.items():
            if name == "Intercept":
                continue
            std_effects[str(name)] = float(tval)

        interactions: Dict[str, Any] = {}
        pairs = [(factors[i], factors[j]) for i in range(len(factors)) for j in range(i + 1, len(factors))]
        if pairs:
            rng = np.random.default_rng(42)
            rng.shuffle(pairs)
            pairs = pairs[: min(len(pairs), int(max_interactions))]

        for a, b in pairs:
            if not np.issubdtype(data[a].dtype, np.number) or not np.issubdtype(data[b].dtype, np.number):
                continue
            tmp = data.copy()
            tmp["__int__"] = tmp[a].astype(float) * tmp[b].astype(float)
            m2 = ols(f"__response__ ~ {rhs} + __int__", data=tmp).fit()
            interactions[f"{a}:{b}"] = {
                "coef": float(m2.params.get("__int__", np.nan)),
                "p": float(m2.pvalues.get("__int__", np.nan)),
            }

        return {
            "target": target,
            "n": int(len(data)),
            "r2": float(model.rsquared),
            "r2_adj": float(model.rsquared_adj),
            "anova": self._anova_to_dict(anova),
            "effects": effects,
            "std_effects": std_effects,
            "interactions": interactions,
        }

    def plot_main_effects(self, df: pd.DataFrame, target: str, factors: List[str], save_path: str | Path) -> None:
        data = df[factors + [target]].dropna().copy()
        n = len(factors)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).reshape(-1)
        y = data[target].values

        for i, f in enumerate(factors):
            ax = axes[i]
            x = data[f]
            if np.issubdtype(x.dtype, np.number):
                ax.scatter(x, y, alpha=0.6)
                if len(x) > 2:
                    z = np.polyfit(x.astype(float), y.astype(float), 1)
                    p = np.poly1d(z)
                    xr = np.linspace(float(np.min(x)), float(np.max(x)), 100)
                    ax.plot(xr, p(xr), alpha=0.9)
            else:
                cats = x.astype(str)
                means = data.groupby(cats)[target].mean().sort_index()
                ax.plot(means.index, means.values, marker="o")
                ax.tick_params(axis="x", rotation=30)

            ax.set_title(f"Main effect: {f}")
            ax.set_xlabel(f)
            ax.set_ylabel(target)
            ax.grid(True, alpha=0.3)

        for j in range(n, len(axes)):
            axes[j].axis("off")

        fig.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def plot_pareto(self, analysis: Dict[str, Any], save_path: str | Path, top_n: int = 15) -> None:
        items = sorted(analysis.get("std_effects", {}).items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
        labels = [k for k, _ in items]
        vals = [abs(v) for _, v in items]

        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(labels))))
        ax.barh(range(len(labels))[::-1], vals[::-1])
        ax.set_yticks(range(len(labels))[::-1], labels[::-1])
        ax.set_xlabel("|t-value| (proxy for standardized effect)")
        ax.set_title(f"Pareto of effects (target: {analysis.get('target')})")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def save_json(self, analysis: Dict[str, Any], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)

    @staticmethod
    def _anova_to_dict(anova: pd.DataFrame) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for idx, row in anova.iterrows():
            d[str(idx)] = {k: (float(row[k]) if np.isfinite(row[k]) else None) for k in anova.columns}
        return d

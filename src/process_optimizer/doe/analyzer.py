from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import statsmodels.api as sm
from statsmodels.formula.api import ols


def _safe_term(col: str) -> str:
    """Quote a column name safely for statsmodels formulas."""
    return f"Q('{col}')"


def _anova_row(anova: pd.DataFrame, factor: str) -> pd.Series | None:
    """Best-effort lookup of an ANOVA row corresponding to a factor.

    We generate formulas using Q('col') to safely escape column names. Statsmodels
    will then use the quoted term name in the ANOVA table. This helper handles
    both quoted and unquoted cases for robustness/backward compatibility.
    """
    for key in (_safe_term(factor), factor, f"C({_safe_term(factor)})", f"C({factor})"):
        if key in anova.index:
            return anova.loc[key]
    return None


class DOEAnalyzer:
    """DOE analysis on measured data.

    Provides ANOVA and effect proxies and generates plots.

    Notes:
    - This is a lightweight analyzer intended for small/medium DOE datasets.
    - It supports screening of:
        * main effects (ANOVA)
        * 2-factor numeric interactions (via an added product term)
        * quadratic curvature terms for numeric factors (via an added square term)
      These are used downstream for DOE-informed feature engineering.
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        df: pd.DataFrame,
        target: str,
        factors: List[str],
        max_interactions: int = 10,
        max_quadratic: int = 10,
    ) -> Dict[str, Any]:
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found")
        for f in factors:
            if f not in df.columns:
                raise ValueError(f"Factor '{f}' not found")

        data = df[factors + [target]].dropna().copy()
        data = data.rename(columns={target: "__response__"})

        # Build RHS with explicit categorical handling.
        # Pandas StringDtype (string[python]/string[pyarrow]) can trigger patsy/statsmodels issues;
        # casting categoricals to plain str/object avoids that.
        rhs_terms: List[str] = []
        for c in factors:
            if not is_numeric_dtype(data[c]):
                data[c] = data[c].astype(str)
                rhs_terms.append(f"C({_safe_term(c)})")
            else:
                rhs_terms.append(_safe_term(c))

        rhs = " + ".join(rhs_terms)
        formula = f"__response__ ~ {rhs}"
        model = ols(formula, data=data).fit()
        anova = sm.stats.anova_lm(model, typ=2)

        # Main effects from ANOVA
        effects: Dict[str, Any] = {}
        for f in factors:
            row = _anova_row(anova, f)
            if row is None:
                continue
            effects[str(f)] = {
                "sum_sq": float(row.get("sum_sq")) if np.isfinite(row.get("sum_sq")) else None,
                "F": float(row.get("F")) if np.isfinite(row.get("F")) else None,
                "p": float(row.get("PR(>F)")) if np.isfinite(row.get("PR(>F)")) else None,
            }

        # Standardized-effect proxy: t-values of coefficients (including categorical levels)
        std_effects: Dict[str, float] = {}
        for name, tval in getattr(model, "tvalues", {}).items():
            if str(name) == "Intercept":
                continue
            if np.isfinite(tval):
                std_effects[str(name)] = float(tval)

        # Interactions: evaluate all numeric-numeric pairs, then keep the best K by p-value.
        interactions_all: List[Tuple[str, Dict[str, Any]]] = []
        pairs = [(factors[i], factors[j]) for i in range(len(factors)) for j in range(i + 1, len(factors))]
        for a, b in pairs:
            if not is_numeric_dtype(data[a]) or not is_numeric_dtype(data[b]):
                continue
            tmp = data.copy()
            tmp["__int__"] = tmp[a].astype(float) * tmp[b].astype(float)
            m2 = ols(f"__response__ ~ {rhs} + __int__", data=tmp).fit()
            meta = {
                "coef": float(m2.params.get("__int__", np.nan)),
                "p": float(m2.pvalues.get("__int__", np.nan)),
            }
            interactions_all.append((f"{a}:{b}", meta))

        interactions_all.sort(key=lambda kv: float((kv[1] or {}).get("p", 1.0)))
        interactions: Dict[str, Any] = {}
        for name, meta in interactions_all[: max(0, int(max_interactions))]:
            interactions[str(name)] = meta

        # Quadratic curvature screening (numeric factors only), keep best K by p-value.
        quad_all: List[Tuple[str, Dict[str, Any]]] = []
        for f in factors:
            if not is_numeric_dtype(data[f]):
                continue
            tmp = data.copy()
            tmp["__sq__"] = tmp[f].astype(float) ** 2
            m2 = ols(f"__response__ ~ {rhs} + __sq__", data=tmp).fit()
            meta = {
                "coef": float(m2.params.get("__sq__", np.nan)),
                "p": float(m2.pvalues.get("__sq__", np.nan)),
            }
            quad_all.append((str(f), meta))

        quad_all.sort(key=lambda kv: float((kv[1] or {}).get("p", 1.0)))
        quadratic: Dict[str, Any] = {}
        for name, meta in quad_all[: max(0, int(max_quadratic))]:
            quadratic[str(name)] = meta

        # Pure error estimate from exact replicates (same factor settings repeated).
        pure_error: Dict[str, Any] = {"available": False}
        try:
            grp = data.groupby(factors, dropna=False)["__response__"]
            counts = grp.size()
            rep_counts = counts[counts > 1]
            if len(rep_counts) > 0:
                means = grp.transform("mean")
                resid = data["__response__"].astype(float) - means.astype(float)
                # only include groups with n>1
                is_rep = data.groupby(factors, dropna=False)["__response__"].transform("size") > 1
                resid_rep = resid[is_rep.values]
                sse = float(np.sum(np.asarray(resid_rep) ** 2))
                dof = int(np.sum(rep_counts.values - 1))
                mse = (sse / dof) if dof > 0 else None
                sigma = float(np.sqrt(mse)) if mse is not None and np.isfinite(mse) else None
                pure_error = {
                    "available": True,
                    "n_replicate_groups": int(len(rep_counts)),
                    "n_replicate_points": int(np.sum(rep_counts.values)),
                    "sse": sse,
                    "dof": dof,
                    "mse": (float(mse) if mse is not None and np.isfinite(mse) else None),
                    "sigma": sigma,
                }
        except Exception:
            pure_error = {"available": False}

        return {
            "target": target,
            "n": int(len(data)),
            "r2": float(model.rsquared),
            "r2_adj": float(model.rsquared_adj),
            "anova": self._anova_to_dict(anova),
            "effects": effects,
            "std_effects": std_effects,
            "interactions": interactions,
            "quadratic": quadratic,
            "pure_error": pure_error,
        }

    def plot_main_effects(self, df: pd.DataFrame, target: str, factors: List[str], save_path: str | Path) -> None:
        data = df[factors + [target]].dropna().copy()
        n = len(factors)
        if n <= 0:
            return
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows))
        axes = np.array(axes).reshape(-1)
        y = data[target].values

        for i, f in enumerate(factors):
            ax = axes[i]
            x = data[f]
            if is_numeric_dtype(x):
                ax.scatter(x, y, alpha=0.65)
                if len(x) > 2 and np.isfinite(np.asarray(x, dtype=float)).all():
                    # Simple linear trend for visualization only
                    try:
                        z = np.polyfit(x.astype(float), y.astype(float), 1)
                        p = np.poly1d(z)
                        xr = np.linspace(float(np.min(x)), float(np.max(x)), 100)
                        ax.plot(xr, p(xr), alpha=0.9)
                    except Exception:
                        pass
            else:
                cats = x.astype(str)
                means = data.groupby(cats)[target].mean().sort_index()
                ax.plot(means.index, means.values, marker="o")

            ax.set_xlabel(str(f))
            ax.set_ylabel(str(target))
            ax.grid(True, alpha=0.25)

        # Hide unused axes
        for j in range(n, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"Main effects (target: {target})", y=1.02)
        fig.tight_layout()

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_pareto(self, analysis: Dict[str, Any], save_path: str | Path, top_n: int = 15) -> None:
        items = sorted(analysis.get("std_effects", {}).items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
        if not items:
            return
        labels = [k for k, _ in items]
        vals = [abs(v) for _, v in items]

        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(labels))))
        ax.barh(range(len(labels))[::-1], vals[::-1])
        ax.set_yticks(range(len(labels))[::-1], labels[::-1])
        ax.set_xlabel("|t-value| (proxy for standardized effect)")
        ax.set_title(f"Pareto of effects (target: {analysis.get('target')})")
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
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

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import qmc

logger = logging.getLogger(__name__)


@dataclass
class FactorSpace:
    numeric_bounds: Dict[str, Tuple[float, float]]
    categorical_levels: Dict[str, List[str]]


class DOEDesigner:
    """DOE planning (suggested experiments).

    DOE designs produced here do not include measured responses.
    """

    def __init__(self, factor_space: FactorSpace, random_state: int = 42):
        self.fs = factor_space
        self.rng = np.random.default_rng(random_state)

    @property
    def numeric_factors(self) -> List[str]:
        return list(self.fs.numeric_bounds.keys())

    @property
    def categorical_factors(self) -> List[str]:
        return list(self.fs.categorical_levels.keys())

    def full_factorial(self, levels: int = 2, max_runs: int = 4096) -> pd.DataFrame:
        num = self.numeric_factors
        cat = self.categorical_factors

        coded = [-1, 1] if levels == 2 else np.linspace(-1, 1, levels).tolist()

        num_grid = list(itertools.product(coded, repeat=len(num))) if num else [()]
        cat_grid = list(itertools.product(*[self.fs.categorical_levels[c] for c in cat])) if cat else [()]

        runs = len(num_grid) * len(cat_grid)
        if runs > max_runs:
            logger.warning("Full factorial would create %d runs; capping to %d by random sampling.", runs, max_runs)

        rows = []
        for npt in num_grid:
            for cpt in cat_grid:
                row = {}
                for i, f in enumerate(num):
                    lo, hi = self.fs.numeric_bounds[f]
                    x = float(npt[i])
                    row[f] = lo + (x + 1.0) * 0.5 * (hi - lo)
                for j, c in enumerate(cat):
                    row[c] = cpt[j]
                rows.append(row)

        if len(rows) > max_runs:
            idx = self.rng.choice(len(rows), size=max_runs, replace=False)
            rows = [rows[i] for i in idx]

        return pd.DataFrame(rows)

    def latin_hypercube(self, n_samples: int) -> pd.DataFrame:
        num = self.numeric_factors
        cat = self.categorical_factors

        if num:
            sampler = qmc.LatinHypercube(d=len(num), seed=int(self.rng.integers(0, 2**31 - 1)))
            u = sampler.random(n=int(n_samples))
            lo = [self.fs.numeric_bounds[f][0] for f in num]
            hi = [self.fs.numeric_bounds[f][1] for f in num]
            x = qmc.scale(u, lo, hi)
            df = pd.DataFrame(x, columns=num)
        else:
            df = pd.DataFrame(index=range(int(n_samples)))

        for c in cat:
            levels = self.fs.categorical_levels[c]
            df[c] = self.rng.choice(levels, size=len(df), replace=True)

        return df

    def plackett_burman(self, n_runs: Optional[int] = None) -> pd.DataFrame:
        """Plackett-Burman screening design for numeric factors."""
        num = self.numeric_factors
        if not num:
            raise ValueError("Plackett-Burman requires at least one numeric factor")

        k = len(num)
        if n_runs is None:
            n_runs = int(np.ceil((k + 1) / 4.0) * 4)

        mat = self._pb_matrix(int(n_runs))
        if mat.shape[1] < k:
            raise ValueError(f"PB matrix ({n_runs} runs) has only {mat.shape[1]} columns but need {k}")

        coded = mat[:, :k]
        df = pd.DataFrame(coded, columns=num)

        for f in num:
            lo, hi = self.fs.numeric_bounds[f]
            df[f] = df[f].apply(lambda v: lo if float(v) < 0 else hi)

        for c in self.categorical_factors:
            levels = self.fs.categorical_levels[c]
            df[c] = self.rng.choice(levels, size=len(df), replace=True)

        return df

    def fractional_factorial(self, resolution: int = 3) -> pd.DataFrame:
        """Lightweight fractional factorial for quick experimentation."""
        num = self.numeric_factors
        if not num:
            raise ValueError("Fractional factorial requires numeric factors")

        k = len(num)
        base_k = max(2, k - max(0, int(resolution) - 2))
        base = np.array(list(itertools.product([-1, 1], repeat=base_k)), dtype=int)
        design = base

        while design.shape[1] < k:
            i = design.shape[1]
            a = (i - 1) % base_k
            b = i % base_k
            new_col = design[:, a] * design[:, b]
            design = np.column_stack([design, new_col])

        df = pd.DataFrame(design[:, :k], columns=num)
        for f in num:
            lo, hi = self.fs.numeric_bounds[f]
            df[f] = df[f].apply(lambda v: lo if float(v) < 0 else hi)

        for c in self.categorical_factors:
            levels = self.fs.categorical_levels[c]
            df[c] = self.rng.choice(levels, size=len(df), replace=True)

        return df

    def box_behnken(self, center_points: int = 3) -> pd.DataFrame:
        num = self.numeric_factors
        if len(num) < 3:
            raise ValueError("Box-Behnken requires at least 3 numeric factors")

        rows = []
        for i in range(len(num)):
            for j in range(i + 1, len(num)):
                for a, b in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    pt = {f: 0 for f in num}
                    pt[num[i]] = a
                    pt[num[j]] = b
                    rows.append(pt)

        for _ in range(int(center_points)):
            rows.append({f: 0 for f in num})

        df = pd.DataFrame(rows)
        df = self._scale_coded(df, alpha=1.0)

        for c in self.categorical_factors:
            levels = self.fs.categorical_levels[c]
            df[c] = self.rng.choice(levels, size=len(df), replace=True)

        return df

    def central_composite(self, center_points: int = 4, alpha: Optional[float] = None) -> pd.DataFrame:
        num = self.numeric_factors
        if len(num) < 2:
            raise ValueError("Central composite requires at least 2 numeric factors")

        k = len(num)
        if alpha is None:
            alpha = float(np.sqrt(k))

        factorial = np.array(list(itertools.product([-1, 1], repeat=k)), dtype=float)

        axial = []
        for i in range(k):
            v_pos = np.zeros(k)
            v_neg = np.zeros(k)
            v_pos[i] = float(alpha)
            v_neg[i] = -float(alpha)
            axial.append(v_pos)
            axial.append(v_neg)
        axial = np.array(axial, dtype=float)

        center = np.zeros((int(center_points), k), dtype=float)

        coded = np.vstack([factorial, axial, center])
        df = pd.DataFrame(coded, columns=num)
        df = self._scale_coded(df, alpha=float(alpha))

        for c in self.categorical_factors:
            levels = self.fs.categorical_levels[c]
            df[c] = self.rng.choice(levels, size=len(df), replace=True)

        return df

    def _scale_coded(self, coded_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
        df = coded_df.copy()
        for f in self.numeric_factors:
            lo, hi = self.fs.numeric_bounds[f]
            x = df[f].astype(float)
            df[f] = lo + (x + alpha) * (hi - lo) / (2.0 * alpha)
        return df

    def _pb_matrix(self, n_runs: int) -> np.ndarray:
        """Return a plus/minus 1 PB matrix with n_runs rows and n_runs-1 columns."""
        standard: Dict[int, List[int]] = {
            12: [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1],
            16: [1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, -1],
            20: [1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1],
            24: [1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1],
        }

        if n_runs in standard:
            base = np.array(standard[n_runs], dtype=int)
        else:
            base = self.rng.choice([-1, 1], size=n_runs - 1)

        mat = np.zeros((n_runs, n_runs - 1), dtype=int)
        for i in range(n_runs):
            mat[i] = np.roll(base, i)
        return mat

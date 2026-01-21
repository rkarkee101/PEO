from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, plots_dir: str | Path):
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def parity(self, y_true, y_pred, title: str, save_name: str) -> str:
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true, y_pred, alpha=0.7)
        mn = float(min(y_true.min(), y_pred.min()))
        mx = float(max(y_true.max(), y_pred.max()))
        ax.plot([mn, mx], [mn, mx])
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = self.plots_dir / save_name
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def residuals(self, y_true, y_pred, title: str, save_name: str) -> str:
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        res = y_true - y_pred

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(y_pred, res, alpha=0.7)
        ax.axhline(0)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = self.plots_dir / save_name
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def calibration_coverage(self, y_true, y_pred, y_std, title: str, save_name: str) -> str:
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        y_std = np.asarray(y_std, dtype=float).ravel()

        multipliers = np.linspace(0.5, 3.0, 11)
        cover = []
        for m in multipliers:
            lo = y_pred - m * y_std
            hi = y_pred + m * y_std
            cover.append(np.mean((y_true >= lo) & (y_true <= hi)))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(multipliers, cover, marker="o")
        ax.set_xlabel("Multiplier m in y_pred +- m*y_std")
        ax.set_ylabel("Empirical coverage")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = self.plots_dir / save_name
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return str(path)

from __future__ import annotations

from pathlib import Path
from math import erf, sqrt

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, plots_dir: str | Path):
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Plot defaults
        self.dpi = 300

    def parity(self, y_true, y_pred, title: str, save_name: str) -> str:
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()

        # Basic metrics
        try:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            r2 = float(r2_score(y_true, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae = float(mean_absolute_error(y_true, y_pred))
        except Exception:
            r2, rmse, mae = None, None, None

        fig, ax = plt.subplots(figsize=(6.2, 6.2))
        ax.scatter(y_true, y_pred, alpha=0.65)
        mn = float(np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)]))
        mx = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)]))
        pad = 0.02 * (mx - mn) if np.isfinite(mx - mn) and (mx - mn) > 0 else 1.0
        mn, mx = mn - pad, mx + pad
        ax.plot([mn, mx], [mn, mx], linestyle="--")
        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

        # Metrics box
        if r2 is not None:
            txt = f"R²={r2:.3f}\nRMSE={rmse:.3g}\nMAE={mae:.3g}"
            ax.text(
                0.05,
                0.95,
                txt,
                transform=ax.transAxes,
                va="top",
                ha="left",
                bbox=dict(boxstyle="round", alpha=0.15),
            )
        fig.tight_layout()

        path = self.plots_dir / save_name
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def residuals(self, y_true, y_pred, title: str, save_name: str) -> str:
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        res = y_true - y_pred

        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        ax.scatter(y_pred, res, alpha=0.65)
        ax.axhline(0, linestyle="--")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        path = self.plots_dir / save_name
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def residual_histogram(self, y_true, y_pred, title: str, save_name: str) -> str:
        """Histogram of residuals (y_true - y_pred)."""

        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        res = y_true - y_pred

        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        ax.hist(res, bins=30)
        ax.axvline(0, linestyle="--")
        ax.set_xlabel("Residual")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        path = self.plots_dir / save_name
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
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

        # Expected coverage for a well-calibrated Gaussian uncertainty.
        expected = [float(erf(float(m) / sqrt(2.0))) for m in multipliers]

        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        ax.plot(multipliers, cover, marker="o")
        ax.plot(multipliers, expected, linestyle="--")
        ax.set_xlabel("Multiplier m in y_pred +- m*y_std")
        ax.set_ylabel("Empirical coverage")
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        path = self.plots_dir / save_name
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def uncertainty_vs_error(self, y_true, y_pred, y_std, title: str, save_name: str) -> str:
        """Scatter of predicted std vs absolute error (diagnostic for uncertainty usefulness)."""

        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
        y_std = np.asarray(y_std, dtype=float).ravel()

        err = np.abs(y_true - y_pred)
        fig, ax = plt.subplots(figsize=(6.2, 4.4))
        ax.scatter(y_std, err, alpha=0.65)
        ax.set_xlabel("Predicted std")
        ax.set_ylabel("Absolute error")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        path = self.plots_dir / save_name
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return str(path)

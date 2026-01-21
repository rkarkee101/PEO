from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from process_optimizer.feature_engineering import FeatureSpec
from process_optimizer.models.base import ProbabilisticRegressor
from process_optimizer.models.feature_transformed import FeatureTransformedRegressor
from process_optimizer.models.gp import GaussianProcessModel
from process_optimizer.models.random_forest import RandomForestModel
from process_optimizer.models.mlp_ensemble import MLPEnsembleModel

logger = logging.getLogger(__name__)


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


@dataclass
class ModelRecord:
    name: str
    fit_info: Dict[str, Any]
    metrics: Dict[str, Any]
    predictions: Dict[str, Any]


class ModelTrainer:
    """Train multiple models for one or more targets.

    - Optional hyperparameter tuning (Optuna if installed)
    - Overfit guard using train/test R2 gap
    - Artifact persistence (joblib models, json metrics)
    """

    def __init__(self, output_dir: str | Path, cfg: Dict[str, Any], feature_names: List[str]):
        self.output_dir = Path(output_dir)
        self.cfg = cfg
        self.feature_names = feature_names

        self.models_dir = self.output_dir / "models"
        self.reports_dir = self.output_dir / "reports"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def train_all(
        self,
        X: np.ndarray,
        y_map: Dict[str, np.ndarray],
        feature_specs: Optional[Dict[str, FeatureSpec]] = None,
    ) -> Dict[str, Any]:
        tr_cfg = self.cfg["training"]
        test_size = float(tr_cfg.get("test_size", 0.2))
        random_state = int(tr_cfg.get("random_state", 42))

        out: Dict[str, Any] = {
            "targets": {},
            "training": {
                "test_size": test_size,
                "random_state": random_state,
                "models_requested": list(tr_cfg.get("models", [])),
                "autotune": bool(tr_cfg.get("autotune", True)),
            },
        }

        feature_specs = feature_specs or {}

        for target, y in y_map.items():
            logger.info("Training models for target: %s", target)
            out["targets"][target] = self._train_one_target(X, y, target, feature_spec=feature_specs.get(target))

        # save a compact report for quick loading
        report_path = self.reports_dir / "training_results.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        return out

    def _train_one_target(self, X: np.ndarray, y: np.ndarray, target_name: str, feature_spec: Optional[FeatureSpec] = None) -> Dict[str, Any]:
        tr_cfg = self.cfg["training"]
        test_size = float(tr_cfg.get("test_size", 0.2))
        random_state = int(tr_cfg.get("random_state", 42))
        model_names: List[str] = list(tr_cfg.get("models", []))
        autotune = bool(tr_cfg.get("autotune", True))
        cv_folds = int(tr_cfg.get("cv_folds", 5))
        overfit_guard = tr_cfg.get("overfit_guard", {}) or {}
        max_gap = float(overfit_guard.get("max_train_test_r2_gap", 0.15))

        Xtr, Xte, ytr, yte = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

        target_out: Dict[str, Any] = {
            "n": int(len(y)),
            "models": {},
        }

        # Train baseline models (all features)
        for mname in model_names:
            try:
                wrapper, fit_info = self._fit_model(Xtr, ytr, mname, autotune=autotune, cv_folds=cv_folds)
            except ImportError as e:
                logger.warning("Skipping model %s (missing dependency): %s", mname, e)
                continue
            except Exception as e:
                logger.exception("Model %s failed: %s", mname, e)
                continue

            # evaluate
            yhat_tr, ystd_tr = wrapper.predict(Xtr, return_std=True)
            yhat_te, ystd_te = wrapper.predict(Xte, return_std=True)

            metrics_tr = self._metrics(ytr, yhat_tr)
            metrics_te = self._metrics(yte, yhat_te)
            gap = abs(metrics_tr["r2"] - metrics_te["r2"])
            overfit_flag = bool(gap > max_gap)

            cov = self._coverage(yte, yhat_te, ystd_te)

            rec = {
                "fit_info": fit_info,
                "metrics": {
                    "train": metrics_tr,
                    "test": metrics_te,
                    "train_test_r2_gap": gap,
                    "overfit_flag": overfit_flag,
                    "uncertainty_coverage": cov,
                },
                "predictions": {
                    "n_test": int(len(yte)),
                    "y_true": yte[:100].tolist(),
                    "y_pred": yhat_te[:100].tolist(),
                    "y_std": (ystd_te[:100].tolist() if ystd_te is not None else None),
                    "note": "Only the first 100 test points are stored in JSON. Full arrays are saved as NPZ.",
                },
            }

            # persist full test arrays
            npz_path = self.reports_dir / f"pred_{target_name}__{mname}.npz"
            np.savez_compressed(npz_path, y_true=yte, y_pred=yhat_te, y_std=(ystd_te if ystd_te is not None else np.nan))
            rec["predictions"]["npz"] = str(npz_path)

            # save model
            model_file = self.models_dir / f"{target_name}__{mname}.joblib"
            joblib.dump(
                {
                    "wrapper": wrapper,
                    "target": target_name,
                    "model_name": mname,
                    "feature_names": list(self.feature_names),
                    "fit_info": fit_info,
                },
                model_file,
            )

            target_out["models"][mname] = rec

        # Train DOE-informed variants (feature selection + interactions)
        if feature_spec is not None and (len(feature_spec.base_feature_names) < len(self.feature_names) or feature_spec.interaction_pairs):
            for mname in model_names:
                doe_name = f"{mname}+doe"
                try:
                    # Tune + fit on the engineered feature space (not the baseline space).
                    Xt = feature_spec.transform(Xtr)
                    base_wrapper, base_fit = self._fit_model(Xt, ytr, mname, autotune=autotune, cv_folds=cv_folds)
                    wrapper = FeatureTransformedRegressor(base_wrapper, feature_spec, name_suffix="+doe")
                    fit_info = {**(base_fit or {}), "doe_informed": True, "feature_spec": feature_spec.to_dict(), "engineered_feature_names": feature_spec.feature_names}
                except ImportError as e:
                    logger.warning("Skipping DOE variant %s (missing dependency): %s", doe_name, e)
                    continue
                except Exception as e:
                    logger.exception("DOE variant %s failed: %s", doe_name, e)
                    continue

                yhat_tr, ystd_tr = wrapper.predict(Xtr, return_std=True)
                yhat_te, ystd_te = wrapper.predict(Xte, return_std=True)

                metrics_tr = self._metrics(ytr, yhat_tr)
                metrics_te = self._metrics(yte, yhat_te)
                gap = abs(metrics_tr["r2"] - metrics_te["r2"])
                overfit_flag = bool(gap > max_gap)
                cov = self._coverage(yte, yhat_te, ystd_te)

                rec = {
                    "fit_info": fit_info,
                    "metrics": {
                        "train": metrics_tr,
                        "test": metrics_te,
                        "train_test_r2_gap": gap,
                        "overfit_flag": overfit_flag,
                        "uncertainty_coverage": cov,
                    },
                    "predictions": {
                        "n_test": int(len(yte)),
                        "y_true": yte[:100].tolist(),
                        "y_pred": yhat_te[:100].tolist(),
                        "y_std": (ystd_te[:100].tolist() if ystd_te is not None else None),
                        "note": "Only the first 100 test points are stored in JSON. Full arrays are saved as NPZ.",
                    },
                    "feature_engineering": {
                        "doe_informed": True,
                        "base_feature_count": int(len(self.feature_names)),
                        "engineered_feature_count": int(len(feature_spec.feature_names)),
                        "selected_base_features": list(feature_spec.base_feature_names),
                        "interaction_features": list(feature_spec.interaction_feature_names),
                    },
                }

                npz_path = self.reports_dir / f"pred_{target_name}__{doe_name}.npz"
                np.savez_compressed(npz_path, y_true=yte, y_pred=yhat_te, y_std=(ystd_te if ystd_te is not None else np.nan))
                rec["predictions"]["npz"] = str(npz_path)

                model_file = self.models_dir / f"{target_name}__{doe_name}.joblib"
                joblib.dump(
                    {
                        "wrapper": wrapper,
                        "target": target_name,
                        "model_name": doe_name,
                        "feature_names": list(self.feature_names),
                        "engineered_feature_names": list(feature_spec.feature_names),
                        "fit_info": fit_info,
                        "feature_spec": feature_spec.to_dict(),
                    },
                    model_file,
                )

                target_out["models"][doe_name] = rec

        # simple ranking
        ranking = []
        for mname, rec in target_out["models"].items():
            rmse_te = float(rec["metrics"]["test"]["rmse"])
            of = 1.0 if rec["metrics"].get("overfit_flag", False) else 0.0
            ranking.append((rmse_te + 0.25 * of, mname))
        ranking.sort(key=lambda t: t[0])
        target_out["model_rank"] = [m for _, m in ranking]

        return target_out

    def _metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            "rmse": _rmse(y_true, y_pred),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    def _coverage(self, y_true: np.ndarray, y_pred: np.ndarray, y_std: Optional[np.ndarray]) -> Dict[str, Any]:
        if y_std is None:
            return {"available": False}
        y_std = np.asarray(y_std, dtype=float).ravel()
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()

        out: Dict[str, Any] = {"available": True}
        for m in [0.5, 1.0, 2.0, 3.0]:
            lo = y_pred - m * y_std
            hi = y_pred + m * y_std
            out[f"coverage_{m}sigma"] = float(np.mean((y_true >= lo) & (y_true <= hi)))
        return out

    def _fit_model(self, X: np.ndarray, y: np.ndarray, model_name: str, autotune: bool, cv_folds: int):
        wrapper = self._build_model(model_name)

        if not autotune:
            fit_res = wrapper.fit(X, y)
            return wrapper, fit_res.info

        # Try Optuna tuning (optional dependency)
        try:
            import optuna
        except Exception:
            fit_res = wrapper.fit(X, y)
            return wrapper, {**fit_res.info, "tuned": False, "reason": "optuna not installed"}

        max_trials = int(self.cfg["training"].get("max_tuning_trials", 40))
        timeout_s = float(self.cfg["training"].get("tuning_timeout_s", 0))
        timeout = None if timeout_s <= 0 else timeout_s

        def objective(trial: optuna.Trial) -> float:
            candidate = self._build_model(model_name, trial=trial)
            kf = KFold(n_splits=max(2, int(cv_folds)), shuffle=True, random_state=42)
            rmses: List[float] = []
            for tr_idx, va_idx in kf.split(X):
                candidate.fit(X[tr_idx], y[tr_idx])
                pred, _ = candidate.predict(X[va_idx], return_std=False)
                rmses.append(_rmse(y[va_idx], pred))
            return float(np.mean(rmses))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=max_trials, timeout=timeout)

        best = study.best_params
        best_model = self._build_model(model_name, fixed_params=best)
        fit_res = best_model.fit(X, y)

        info = {**fit_res.info, "tuned": True, "best_params": best, "best_value": float(study.best_value)}
        return best_model, info

    def _build_model(self, model_name: str, trial=None, fixed_params: Optional[Dict[str, Any]] = None) -> ProbabilisticRegressor:
        model_name = str(model_name).lower()
        # Optuna returns raw trial parameter names (e.g., h1/h2). We normalize
        # those into constructor-compatible params below.
        fixed_params = dict(fixed_params or {})

        if model_name in {"gp", "gaussian_process"}:
            params: Dict[str, Any] = {
                "nu": 2.5,
                "length_scale": 1.0,
                "noise_level": 1e-4,
            }
            if trial is not None:
                params["nu"] = trial.suggest_categorical("nu", [1.5, 2.5])
                params["noise_level"] = trial.suggest_float("noise_level", 1e-8, 1e-2, log=True)
                params["length_scale"] = trial.suggest_float("length_scale", 0.1, 10.0, log=True)
            params.update(fixed_params)
            return GaussianProcessModel(**params)

        if model_name in {"random_forest", "rf"}:
            params = {
                "n_estimators": 400,
                "max_depth": None,
                "min_samples_leaf": 1,
            }
            if trial is not None:
                params["n_estimators"] = trial.suggest_int("n_estimators", 200, 900, step=100)
                params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 6)
                params["max_depth"] = trial.suggest_categorical("max_depth", [None, 6, 10, 16, 24])
            params.update(fixed_params)
            return RandomForestModel(**params)

        if model_name in {"xgboost", "xgb"}:
            from process_optimizer.models.xgboost_ensemble import XGBoostEnsembleModel

            params = {
                "n_models": 20,
                "n_estimators": 700,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 1.0,
            }
            if trial is not None:
                params["n_models"] = trial.suggest_int("n_models", 10, 35, step=5)
                params["n_estimators"] = trial.suggest_int("n_estimators", 300, 1200, step=100)
                params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
                params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
                params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
                params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)
                params["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True)
            params.update(fixed_params)
            return XGBoostEnsembleModel(**params)

        if model_name in {"mlp"}:
            params = {
                "n_models": 15,
                "hidden_layer_sizes": (128, 64),
                "alpha": 1e-4,
                "learning_rate_init": 1e-3,
                "max_iter": 1500,
            }
            if trial is not None:
                params["n_models"] = trial.suggest_int("n_models", 8, 25, step=1)
                params["alpha"] = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
                params["learning_rate_init"] = trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True)
                h1 = trial.suggest_int("h1", 32, 256, step=32)
                h2 = trial.suggest_int("h2", 16, 160, step=16)
                params["hidden_layer_sizes"] = (h1, h2)

            # If fixed_params contains h1/h2 (from Optuna best_params), convert to
            # hidden_layer_sizes and drop h1/h2 so we don't pass unknown kwargs.
            if "h1" in fixed_params or "h2" in fixed_params:
                h1 = int(fixed_params.pop("h1", params["hidden_layer_sizes"][0]))
                h2 = int(fixed_params.pop("h2", params["hidden_layer_sizes"][1]))
                fixed_params["hidden_layer_sizes"] = (h1, h2)

            # Normalize list -> tuple if user provides a list
            if isinstance(fixed_params.get("hidden_layer_sizes"), list):
                fixed_params["hidden_layer_sizes"] = tuple(int(x) for x in fixed_params["hidden_layer_sizes"])
            params.update(fixed_params)
            return MLPEnsembleModel(**params)

        raise ValueError(f"Unknown model name: {model_name}")

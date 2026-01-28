from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

from process_optimizer.feature_engineering import FeatureSpec
from process_optimizer.models.base import ProbabilisticRegressor
from process_optimizer.models.calibrated import UncertaintyCalibratedRegressor
from process_optimizer.models.feature_transformed import FeatureTransformedRegressor
from process_optimizer.models.gp import GaussianProcessModel
from process_optimizer.models.mlp_ensemble import MLPEnsembleModel
from process_optimizer.models.random_forest import RandomForestModel
from process_optimizer.models.response_surface import ResponseSurfaceModel
from process_optimizer.models.rsm_gp_residual import RSMGPResidualModel
from process_optimizer.models.rsm_mlp_residual import RSMMLPResidualModel

logger = logging.getLogger(__name__)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


@dataclass
class _CVPack:
    rmse_mean: float
    rmse_std: float
    residuals: np.ndarray
    pred_std: Optional[np.ndarray]


class ModelTrainer:
    """Train probabilistic regression surrogates for process engineering.

    Improvements vs the initial implementation:
    - CV-first model selection for DOE-sized datasets
    - Optional Optuna hyperparameter tuning on the training split only
    - Automatic overfit guard (train/test R2 gap) with selection penalties
    - Uncertainty calibration via CV residuals (std scaling)
    - DOE-informed feature engineering variants (+doe) using FeatureSpec
    """

    def __init__(self, output_dir: str | Path, cfg: Dict[str, Any], feature_names: List[str]):
        self.output_dir = Path(output_dir)
        self.cfg = cfg
        self.feature_names = list(feature_names)

        self.models_dir = self.output_dir / "models"
        self.reports_dir = self.output_dir / "reports"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def train_all(
        self,
        X: np.ndarray,
        y_map: Dict[str, np.ndarray],
        feature_specs: Optional[Dict[str, FeatureSpec]] = None,
        *,
        train_idx: Optional[Sequence[int]] = None,
        test_idx: Optional[Sequence[int]] = None,
        pure_error: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        tr_cfg = self.cfg.get("training", {}) or {}

        test_size = _safe_float(tr_cfg.get("test_size", 0.2), 0.2)
        random_state = _safe_int(tr_cfg.get("random_state", 42), 42)

        out: Dict[str, Any] = {
            "targets": {},
            "training": {
                "test_size": test_size,
                "random_state": random_state,
                "models_requested": list(tr_cfg.get("models", [])),
                "autotune": bool(tr_cfg.get("autotune", True)),
                "cv_folds": _safe_int(tr_cfg.get("cv_folds", 5), 5),
                "model_selection": dict(tr_cfg.get("model_selection", {}) or {}),
                "uncertainty_calibration": dict(tr_cfg.get("uncertainty_calibration", {}) or {}),
            },
        }

        feature_specs = feature_specs or {}
        pure_error = pure_error or {}

        X = np.asarray(X, dtype=float)

        for target, y in y_map.items():
            y = np.asarray(y, dtype=float).ravel()
            out["targets"][str(target)] = self._train_one_target(
                X,
                y,
                str(target),
                feature_spec=feature_specs.get(str(target)),
                train_idx=train_idx,
                test_idx=test_idx,
                pure_error_info=pure_error.get(str(target)),
            )

        report_path = self.reports_dir / "training_results.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        return out

    def _train_one_target(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_name: str,
        *,
        feature_spec: Optional[FeatureSpec],
        train_idx: Optional[Sequence[int]],
        test_idx: Optional[Sequence[int]],
        pure_error_info: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        tr_cfg = self.cfg.get("training", {}) or {}

        model_names: List[str] = list(tr_cfg.get("models", []))
        autotune = bool(tr_cfg.get("autotune", True))

        cv_folds_req = _safe_int(tr_cfg.get("cv_folds", 5), 5)
        overfit_guard = tr_cfg.get("overfit_guard", {}) or {}
        max_gap = _safe_float(overfit_guard.get("max_train_test_r2_gap", 0.15), 0.15)

        if train_idx is None or test_idx is None:
            test_size = _safe_float(tr_cfg.get("test_size", 0.2), 0.2)
            random_state = _safe_int(tr_cfg.get("random_state", 42), 42)
            Xtr, Xte, ytr, yte = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
            )
        else:
            tr_idx = np.asarray(list(train_idx), dtype=int)
            te_idx = np.asarray(list(test_idx), dtype=int)
            Xtr, Xte = X[tr_idx], X[te_idx]
            ytr, yte = y[tr_idx], y[te_idx]

        # Adjust folds for tiny datasets.
        cv_folds = int(max(2, min(cv_folds_req, len(ytr)))) if len(ytr) >= 2 else 2

        # Measurement-noise floor (if replicate-based pure error is available)
        noise_floor = None
        try:
            if pure_error_info and bool(pure_error_info.get("available")):
                noise_floor = pure_error_info.get("sigma")
                if noise_floor is not None and np.isfinite(noise_floor):
                    noise_floor = float(noise_floor)
                else:
                    noise_floor = None
        except Exception:
            noise_floor = None

        target_out: Dict[str, Any] = {
            "n": int(len(y)),
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
            "pure_error": dict(pure_error_info or {}),
            "models": {},
        }

        def _train_variant(
            base_model_name: str,
            *,
            variant_key: str,
            spec: Optional[FeatureSpec],
        ) -> None:
            try:
                wrapper, fit_info = self._fit_model(
                    Xtr,
                    ytr,
                    base_model_name,
                    autotune=autotune,
                    cv_folds=cv_folds,
                    feature_spec=spec,
                    noise_floor=noise_floor,
                )
            except ImportError as e:
                logger.warning("Skipping model %s (missing dependency): %s", variant_key, e)
                return
            except Exception as e:
                logger.exception("Model %s failed: %s", variant_key, e)
                return

            # Evaluate
            yhat_tr, ystd_tr = wrapper.predict(Xtr, return_std=True)
            yhat_te, ystd_te = wrapper.predict(Xte, return_std=True)

            metrics_tr = self._metrics(ytr, yhat_tr)
            metrics_te = self._metrics(yte, yhat_te)
            r2_gap = float(metrics_tr["r2"] - metrics_te["r2"])
            overfit_flag = bool(r2_gap > max_gap)

            cov = self._coverage(yte, yhat_te, ystd_te)

            # Persist full test arrays
            npz_path = self.reports_dir / f"pred_{target_name}__{variant_key}.npz"
            np.savez_compressed(
                npz_path,
                y_true=yte,
                y_pred=yhat_te,
                y_std=(ystd_te if ystd_te is not None else np.nan),
            )

            rec = {
                "fit_info": fit_info,
                "metrics": {
                    "train": metrics_tr,
                    "test": metrics_te,
                    "train_test_r2_gap": r2_gap,
                    "overfit_flag": overfit_flag,
                    "cv": {
                        "rmse_mean": fit_info.get("cv_rmse_mean"),
                        "rmse_std": fit_info.get("cv_rmse_std"),
                    },
                    "uncertainty_coverage": cov,
                },
                "predictions": {
                    "n_test": int(len(yte)),
                    "y_true": yte[:100].tolist(),
                    "y_pred": yhat_te[:100].tolist(),
                    "y_std": (ystd_te[:100].tolist() if ystd_te is not None else None),
                    "npz": str(npz_path),
                    "note": "Only the first 100 test points are stored in JSON. Full arrays are saved as NPZ.",
                },
            }

            if spec is not None:
                rec["feature_engineering"] = {
                    "doe_informed": True,
                    "base_feature_count": int(len(self.feature_names)),
                    "engineered_feature_count": int(len(spec.feature_names)),
                    "selected_base_features": list(spec.base_feature_names),
                    "interaction_features": list(spec.interaction_feature_names),
                    "power_features": list(spec.power_feature_names),
                }

            # Save model
            model_file = self.models_dir / f"{target_name}__{variant_key}.joblib"
            joblib.dump(
                {
                    "wrapper": wrapper,
                    "target": target_name,
                    "model_name": variant_key,
                    "feature_names": list(self.feature_names),
                    "engineered_feature_names": (list(spec.feature_names) if spec is not None else None),
                    "feature_spec": (spec.to_dict() if spec is not None else None),
                    "fit_info": fit_info,
                },
                model_file,
            )

            target_out["models"][variant_key] = rec

        # Baseline models
        for mname in model_names:
            _train_variant(mname, variant_key=str(mname), spec=None)

        # DOE-informed variants
        if feature_spec is not None:
            # Train DOE-informed variants for every requested base model.
            for mname in model_names:
                _train_variant(mname, variant_key=f"{mname}+doe", spec=feature_spec)

            # If user didn't explicitly request an RSM model, still consider it when DOE-informed spec exists.
            if "rsm" not in {str(m).lower() for m in model_names}:
                _train_variant("rsm", variant_key="rsm+doe", spec=feature_spec)

            if "rsm_gp" not in {str(m).lower() for m in model_names} and "rsm_gp_residual" not in {str(m).lower() for m in model_names}:
                _train_variant("rsm_gp", variant_key="rsm_gp+doe", spec=feature_spec)

        # Ranking
        sel_cfg = tr_cfg.get("model_selection", {}) or {}
        metric = str(sel_cfg.get("metric", "cv_rmse")).lower()
        overfit_penalty = _safe_float(sel_cfg.get("overfit_penalty", 0.25), 0.25)

        # Automatic overfit protection policy:
        # - Penalize overfit models in the score (always)
        # - Optionally prefer a non-overfit model if it's close in score
        prefer_non_overfit = bool(sel_cfg.get("prefer_non_overfit", True))
        enforce_non_overfit = bool(sel_cfg.get("enforce_non_overfit", False))
        tol = _safe_float(sel_cfg.get("non_overfit_score_tolerance", 0.05), 0.05)  # relative tolerance

        scored: List[Tuple[float, str, bool]] = []
        for mkey, rec in (target_out.get("models", {}) or {}).items():
            base_score = None
            try:
                if metric == "cv_rmse":
                    base_score = float((rec.get("metrics", {}).get("cv", {}) or {}).get("rmse_mean"))
                if base_score is None or not np.isfinite(base_score):
                    base_score = float(rec.get("metrics", {}).get("test", {}).get("rmse"))
            except Exception:
                base_score = float(rec.get("metrics", {}).get("test", {}).get("rmse", np.inf))

            of = 1.0 if bool(rec.get("metrics", {}).get("overfit_flag")) else 0.0
            scored.append((float(base_score) + overfit_penalty * of, str(mkey), bool(of > 0)))

        scored.sort(key=lambda t: t[0])
        target_out["model_rank"] = [m for _, m, _ in scored]

        best_model = target_out["model_rank"][0] if target_out["model_rank"] else None
        best_score = scored[0][0] if scored else None
        best_overfit = scored[0][2] if scored else False

        chosen_policy = "lowest_score"
        if best_model is not None and best_score is not None and best_overfit and (prefer_non_overfit or enforce_non_overfit):
            # Find the best non-overfit model.
            non_overfit = [(s, m) for (s, m, oflag) in scored if not oflag]
            if non_overfit:
                s2, m2 = non_overfit[0]
                # If enforce_non_overfit -> always switch.
                # Else switch only if score degradation is acceptable.
                if enforce_non_overfit:
                    best_model = m2
                    chosen_policy = "enforce_non_overfit"
                else:
                    denom = float(best_score) if float(best_score) > 0 else 1.0
                    rel = float(s2 - best_score) / denom
                    if rel <= float(tol):
                        best_model = m2
                        chosen_policy = "prefer_non_overfit_within_tolerance"

        target_out["best_model"] = best_model
        target_out["selection"] = {
            "metric": metric,
            "overfit_penalty": overfit_penalty,
            "prefer_non_overfit": prefer_non_overfit,
            "enforce_non_overfit": enforce_non_overfit,
            "non_overfit_score_tolerance": tol,
            "chosen_policy": chosen_policy,
        }

        return target_out

    def _metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred = np.asarray(y_pred, dtype=float).ravel()
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

        ok = np.isfinite(y_std) & (y_std > 0) & np.isfinite(y_true) & np.isfinite(y_pred)
        if ok.sum() < 3:
            return {"available": False}

        y_std = y_std[ok]
        y_true = y_true[ok]
        y_pred = y_pred[ok]

        out: Dict[str, Any] = {"available": True, "n": int(len(y_std))}
        for m in [0.5, 1.0, 2.0, 3.0]:
            lo = y_pred - m * y_std
            hi = y_pred + m * y_std
            out[f"coverage_{m}sigma"] = float(np.mean((y_true >= lo) & (y_true <= hi)))
        return out

    def _cv_eval(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        *,
        cv_folds: int,
        fixed_params: Optional[Dict[str, Any]],
        feature_spec: Optional[FeatureSpec],
    ) -> _CVPack:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        n_splits = int(max(2, min(int(cv_folds), len(y))))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        rmses: List[float] = []
        resid_all: List[np.ndarray] = []
        std_all: List[np.ndarray] = []
        have_std = True

        for tr_idx, va_idx in kf.split(X):
            base = self._build_model(model_name, fixed_params=fixed_params)
            wrapper: ProbabilisticRegressor
            if feature_spec is not None:
                wrapper = FeatureTransformedRegressor(base, feature_spec, name_suffix="+doe")
            else:
                wrapper = base

            wrapper.fit(X[tr_idx], y[tr_idx])
            pred, std = wrapper.predict(X[va_idx], return_std=True)
            pred = np.asarray(pred).ravel()
            rmses.append(_rmse(y[va_idx], pred))
            resid_all.append(np.asarray(y[va_idx]).ravel() - pred)
            if std is None:
                have_std = False
            else:
                std_all.append(np.asarray(std).ravel())

        resid = np.concatenate(resid_all, axis=0) if resid_all else np.asarray([], dtype=float)
        pred_std = None
        if have_std and std_all:
            pred_std = np.concatenate(std_all, axis=0)

        return _CVPack(
            rmse_mean=float(np.mean(rmses)) if rmses else float("nan"),
            rmse_std=float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
            residuals=resid,
            pred_std=pred_std,
        )

    def _compute_uncertainty_calibration(
        self,
        residuals: np.ndarray,
        pred_std: np.ndarray,
        *,
        target_coverage: float,
        min_scale: float,
        max_scale: float,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute multiplicative scale factor for predictive std.

        Method: quantile matching of |residual| / std against a Normal reference.
        - For target_coverage ~0.6827, the reference quantile is 1.0.
        """
        from scipy.stats import norm

        residuals = np.asarray(residuals, dtype=float).ravel()
        pred_std = np.asarray(pred_std, dtype=float).ravel()

        ok = np.isfinite(residuals) & np.isfinite(pred_std) & (pred_std > 0)
        residuals = residuals[ok]
        pred_std = pred_std[ok]

        info: Dict[str, Any] = {
            "method": "quantile",
            "n": int(len(residuals)),
            "target_coverage": float(target_coverage),
            "scale": 1.0,
        }

        if len(residuals) < 12:
            info["available"] = False
            info["reason"] = "too_few_points"
            return 1.0, info

        c = float(_clamp(float(target_coverage), 0.10, 0.99))
        ratios = np.abs(residuals) / np.maximum(pred_std, 1e-12)
        q_emp = float(np.quantile(ratios, c))
        q_ref = float(norm.ppf((1.0 + c) / 2.0))
        if not np.isfinite(q_ref) or q_ref <= 0:
            q_ref = 1.0

        scale = float(q_emp / q_ref) if np.isfinite(q_emp) else 1.0
        scale = float(_clamp(scale, float(min_scale), float(max_scale)))

        # Some diagnostics
        z = ratios / max(scale, 1e-12)
        info.update(
            {
                "available": True,
                "q_emp": q_emp,
                "q_ref": q_ref,
                "median_abs_z": float(np.median(z)),
                "scale": scale,
            }
        )
        return scale, info

    def _fit_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        *,
        autotune: bool,
        cv_folds: int,
        feature_spec: Optional[FeatureSpec],
        noise_floor: Optional[float],
    ) -> Tuple[ProbabilisticRegressor, Dict[str, Any]]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        tr_cfg = self.cfg.get("training", {}) or {}
        cal_cfg = tr_cfg.get("uncertainty_calibration", {}) or {}

        # ---------------------------
        # Hyperparameter tuning (optional)
        # ---------------------------
        best_params: Optional[Dict[str, Any]] = None
        tune_meta: Dict[str, Any] = {"tuned": False, "method": None}

        if autotune:
            # Tuning policy:
            #  - If optuna is available, use it (best quality).
            #  - Otherwise, fall back to an internal random-search tuner.
            tune_cfg = (tr_cfg.get("tuning") or {})
            method = str(tune_cfg.get("method", "auto")).lower().strip()
            max_trials = _safe_int(tr_cfg.get("max_tuning_trials", 40), 40)
            timeout_s = _safe_float(tr_cfg.get("tuning_timeout_s", 0), 0)
            timeout = None if timeout_s <= 0 else float(timeout_s)

            def _cv_objective_for_trial(trial_obj: Any) -> float:
                kf = KFold(n_splits=int(max(2, min(int(cv_folds), len(y)))), shuffle=True, random_state=42)
                rmses: List[float] = []
                for tr_idx, va_idx in kf.split(X):
                    base = self._build_model(model_name, trial=trial_obj)
                    wrapper: ProbabilisticRegressor
                    if feature_spec is not None:
                        wrapper = FeatureTransformedRegressor(base, feature_spec, name_suffix="+doe")
                    else:
                        wrapper = base
                    wrapper.fit(X[tr_idx], y[tr_idx])
                    pred, _ = wrapper.predict(X[va_idx], return_std=False)
                    rmses.append(_rmse(y[va_idx], pred))
                return float(np.mean(rmses))

            # -----------------
            # Optuna tuner
            # -----------------
            if method in {"auto", "optuna"}:
                try:
                    import optuna  # type: ignore

                    def objective(trial: "optuna.Trial") -> float:
                        return _cv_objective_for_trial(trial)

                    study = optuna.create_study(direction="minimize")
                    study.optimize(objective, n_trials=max_trials, timeout=timeout)

                    best_params = dict(study.best_params)
                    tune_meta = {
                        "tuned": True,
                        "method": "optuna",
                        "best_params": best_params,
                        "best_value": float(study.best_value),
                        "n_trials": int(len(study.trials)),
                    }
                except Exception as e:
                    if method == "optuna":
                        tune_meta = {"tuned": False, "method": "optuna", "reason": f"optuna_failed: {e}"}
                    else:
                        # fall through to random search
                        tune_meta = {"tuned": False, "method": "optuna", "reason": f"optuna_unavailable_or_failed: {e}"}

            # -----------------
            # Internal random-search tuner
            # -----------------
            if not bool(tune_meta.get("tuned")) and method in {"auto", "random"}:
                import time

                class _RandomTrial:
                    """Optuna-like sampling interface used by _build_model()."""

                    def __init__(self, rng: np.random.Generator):
                        self.rng = rng
                        self.params: Dict[str, Any] = {}

                    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
                        low = float(low)
                        high = float(high)
                        if log:
                            v = float(np.exp(self.rng.uniform(np.log(low), np.log(high))))
                        else:
                            v = float(self.rng.uniform(low, high))
                        self.params[name] = v
                        return v

                    def suggest_int(self, name: str, low: int, high: int, step: int = 1) -> int:
                        low_i = int(low)
                        high_i = int(high)
                        step_i = int(max(1, step))
                        grid = np.arange(low_i, high_i + 1, step_i, dtype=int)
                        v = int(self.rng.choice(grid))
                        self.params[name] = v
                        return v

                    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any:
                        v = self.rng.choice(list(choices))
                        # numpy can return np.scalar types; normalize
                        if isinstance(v, np.generic):
                            v = v.item()
                        self.params[name] = v
                        return v

                rng = np.random.default_rng(_safe_int(tr_cfg.get("random_state", 42), 42) + 1337)
                start = time.time()
                best_val = float("inf")
                best_p: Optional[Dict[str, Any]] = None
                n_done = 0
                for _ in range(int(max_trials)):
                    if timeout is not None and (time.time() - start) > float(timeout):
                        break
                    rt = _RandomTrial(rng)
                    try:
                        val = _cv_objective_for_trial(rt)
                    except Exception:
                        continue
                    n_done += 1
                    if np.isfinite(val) and float(val) < float(best_val):
                        best_val = float(val)
                        best_p = dict(rt.params)

                if best_p is not None:
                    best_params = best_p
                    tune_meta = {
                        "tuned": True,
                        "method": "random",
                        "best_params": best_params,
                        "best_value": float(best_val),
                        "n_trials": int(n_done),
                    }
                else:
                    tune_meta = {
                        "tuned": False,
                        "method": "random",
                        "reason": "random_search_failed",
                        "n_trials": int(n_done),
                    }

        # ---------------------------
        # Fit final model on full training set
        # ---------------------------
        base = self._build_model(model_name, fixed_params=best_params)
        wrapper: ProbabilisticRegressor
        if feature_spec is not None:
            wrapper = FeatureTransformedRegressor(base, feature_spec, name_suffix="+doe")
        else:
            wrapper = base

        fit_res = wrapper.fit(X, y)
        fit_info = dict(fit_res.info or {})

        # ---------------------------
        # CV evaluation (for selection and calibration)
        # ---------------------------
        cv_pack = self._cv_eval(
            X,
            y,
            model_name,
            cv_folds=cv_folds,
            fixed_params=best_params,
            feature_spec=feature_spec,
        )

        fit_info.update(
            {
                **tune_meta,
                "cv_rmse_mean": float(cv_pack.rmse_mean) if np.isfinite(cv_pack.rmse_mean) else None,
                "cv_rmse_std": float(cv_pack.rmse_std) if np.isfinite(cv_pack.rmse_std) else None,
            }
        )

        # ---------------------------
        # Uncertainty calibration
        # ---------------------------
        calibrated_wrapper: ProbabilisticRegressor = wrapper
        if bool(cal_cfg.get("enabled", True)) and cv_pack.pred_std is not None:
            target_cov = _safe_float(cal_cfg.get("target_coverage", 0.6827), 0.6827)
            min_scale = _safe_float(cal_cfg.get("min_scale", 0.25), 0.25)
            max_scale = _safe_float(cal_cfg.get("max_scale", 25.0), 25.0)
            scale, cal_info = self._compute_uncertainty_calibration(
                cv_pack.residuals,
                cv_pack.pred_std,
                target_coverage=target_cov,
                min_scale=min_scale,
                max_scale=max_scale,
            )

            # Measurement-noise floor: std shouldn't fall below aleatoric noise.
            min_std_cfg = _safe_float(cal_cfg.get("min_std", 1e-9), 1e-9)
            min_std = float(min_std_cfg)
            if noise_floor is not None and np.isfinite(noise_floor) and noise_floor > 0:
                min_std = max(min_std, float(noise_floor))
                cal_info["noise_floor_sigma"] = float(noise_floor)

            calibrated_wrapper = UncertaintyCalibratedRegressor(wrapper, scale=scale, min_std=min_std)
            fit_info["uncertainty_calibration"] = cal_info
        else:
            fit_info["uncertainty_calibration"] = {
                "available": False,
                "enabled": bool(cal_cfg.get("enabled", True)),
                "reason": "no_predicted_std" if cv_pack.pred_std is None else "disabled",
            }

        return calibrated_wrapper, fit_info

    def _build_model(
        self,
        model_name: str,
        *,
        trial: Any = None,
        fixed_params: Optional[Dict[str, Any]] = None,
    ) -> ProbabilisticRegressor:
        model_name = str(model_name).lower().strip()
        fixed_params = dict(fixed_params or {})
        tr_cfg = (self.cfg.get("training", {}) or {})
        random_state = _safe_int(tr_cfg.get("random_state", 42), 42)

        # GP options (used by gp and rsm_gp models)
        gp_cfg = (tr_cfg.get("gp") or {})
        gp_tune = (gp_cfg.get("tuning") or {})

        def _safe_2tuple(v: Any, default: Tuple[float, float]) -> Tuple[float, float]:
            try:
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    return (float(v[0]), float(v[1]))
            except Exception:
                pass
            return (float(default[0]), float(default[1]))

        # If we already do outer-loop tuning (autotune=True), disable scikit-learn's inner
        # kernel optimizer by default to avoid nested optimization + ConvergenceWarnings.
        autotune = bool(tr_cfg.get("autotune", True))
        optimize_kernel = bool(gp_cfg.get("optimize_kernel", (not autotune)))
        gp_alpha = _safe_float(gp_cfg.get("alpha", 1e-8), 1e-8)
        gp_length_scale_bounds = _safe_2tuple(gp_cfg.get("length_scale_bounds"), (1e-5, 1e5))
        gp_noise_level_bounds = _safe_2tuple(gp_cfg.get("noise_level_bounds"), (1e-8, 1e-1))
        gp_n_restarts = _safe_int(gp_cfg.get("n_restarts_optimizer", 5), 5)

        # Optuna search ranges (outer-loop)
        gp_noise_min = _safe_float(gp_tune.get("noise_level_min", 1e-8), 1e-8)
        gp_noise_max = _safe_float(gp_tune.get("noise_level_max", 1e-2), 1e-2)
        gp_ls_min = _safe_float(gp_tune.get("length_scale_min", 0.1), 0.1)
        gp_ls_max = _safe_float(gp_tune.get("length_scale_max", 10.0), 10.0)


        if model_name in {"gp", "gaussian_process"}:
            params: Dict[str, Any] = {
                "nu": 2.5,
                "length_scale": 1.0,
                "noise_level": 1e-4,
                "length_scale_bounds": gp_length_scale_bounds,
                "noise_level_bounds": gp_noise_level_bounds,
                "alpha": gp_alpha,
                "optimize_kernel": optimize_kernel,
                "n_restarts_optimizer": gp_n_restarts,
                "random_state": random_state,
            }
            if trial is not None:
                params["nu"] = trial.suggest_categorical("nu", [1.5, 2.5])
                noise_lo, noise_hi = sorted([gp_noise_min, gp_noise_max])
                ls_lo, ls_hi = sorted([gp_ls_min, gp_ls_max])
                params["noise_level"] = trial.suggest_float("noise_level", noise_lo, noise_hi, log=True)
                params["length_scale"] = trial.suggest_float("length_scale", ls_lo, ls_hi, log=True)
            params.update(fixed_params)
            return GaussianProcessModel(**params)

        if model_name in {"random_forest", "rf"}:
            params = {
                "n_estimators": 400,
                "max_depth": None,
                "min_samples_leaf": 1,
                "random_state": random_state,
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
                "random_state": random_state,
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
                "random_state": random_state,
            }
            if trial is not None:
                params["n_models"] = trial.suggest_int("n_models", 8, 25, step=1)
                params["alpha"] = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)
                params["learning_rate_init"] = trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True)
                h1 = trial.suggest_int("h1", 32, 256, step=32)
                h2 = trial.suggest_int("h2", 16, 160, step=16)
                params["hidden_layer_sizes"] = (h1, h2)

            # Normalize Optuna best_params format
            if "h1" in fixed_params or "h2" in fixed_params:
                h1 = int(fixed_params.pop("h1", params["hidden_layer_sizes"][0]))
                h2 = int(fixed_params.pop("h2", params["hidden_layer_sizes"][1]))
                fixed_params["hidden_layer_sizes"] = (h1, h2)

            if isinstance(fixed_params.get("hidden_layer_sizes"), list):
                fixed_params["hidden_layer_sizes"] = tuple(int(x) for x in fixed_params["hidden_layer_sizes"])

            params.update(fixed_params)
            return MLPEnsembleModel(**params)

        if model_name in {"rsm", "response_surface"}:
            params = {
                "alpha_1": 1e-6,
                "alpha_2": 1e-6,
                "lambda_1": 1e-6,
                "lambda_2": 1e-6,
                "fit_intercept": True,
                "random_state": random_state,
            }
            if trial is not None:
                params["alpha_1"] = trial.suggest_float("alpha_1", 1e-9, 1e-3, log=True)
                params["alpha_2"] = trial.suggest_float("alpha_2", 1e-9, 1e-3, log=True)
                params["lambda_1"] = trial.suggest_float("lambda_1", 1e-9, 1e-3, log=True)
                params["lambda_2"] = trial.suggest_float("lambda_2", 1e-9, 1e-3, log=True)
            params.update(fixed_params)
            return ResponseSurfaceModel(**params)

        if model_name in {"rsm_gp", "rsm_gp_residual", "rsm+gp"}:
            params = {
                "nu": 2.5,
                "length_scale": 1.0,
                "noise_level": 1e-4,
                "length_scale_bounds": gp_length_scale_bounds,
                "noise_level_bounds": gp_noise_level_bounds,
                "alpha": gp_alpha,
                "optimize_kernel": optimize_kernel,
                "n_restarts_optimizer": gp_n_restarts,
                "alpha_1": 1e-6,
                "alpha_2": 1e-6,
                "lambda_1": 1e-6,
                "lambda_2": 1e-6,
                "random_state": random_state,
            }
            if trial is not None:
                noise_lo, noise_hi = sorted([gp_noise_min, gp_noise_max])
                ls_lo, ls_hi = sorted([gp_ls_min, gp_ls_max])
                params["noise_level"] = trial.suggest_float("noise_level", noise_lo, noise_hi, log=True)
                params["length_scale"] = trial.suggest_float("length_scale", ls_lo, ls_hi, log=True)
                params["nu"] = trial.suggest_categorical("nu", [1.5, 2.5])
                # Trend regularization
                params["lambda_1"] = trial.suggest_float("lambda_1", 1e-9, 1e-3, log=True)
                params["lambda_2"] = trial.suggest_float("lambda_2", 1e-9, 1e-3, log=True)
            params.update(fixed_params)
            return RSMGPResidualModel(**params)

        if model_name in {"rsm_mlp", "rsm_mlp_residual", "rsm+mlp", "doe_nn", "doe_mlp"}:
            # Semi-parametric: Bayesian RSM trend + MLP ensemble residual.
            # NOTE: if DOE-informed features are enabled, this model is typically
            # wrapped via FeatureTransformedRegressor to receive screened interactions
            # and quadratic terms.
            params = {
                # Trend (BayesianRidge)
                "alpha_1": 1e-6,
                "alpha_2": 1e-6,
                "lambda_1": 1e-6,
                "lambda_2": 1e-6,
                "fit_intercept": True,
                # Residual NN ensemble
                "n_models": 12,
                "hidden_layer_sizes": (128, 64),
                "nn_alpha": 1e-4,
                "learning_rate_init": 1e-3,
                "max_iter": 1500,
                "random_state": random_state,
            }
            if trial is not None:
                params["n_models"] = trial.suggest_int("n_models", 6, 22, step=1)
                params["nn_alpha"] = trial.suggest_float("nn_alpha", 1e-6, 1e-2, log=True)
                params["learning_rate_init"] = trial.suggest_float("learning_rate_init", 1e-4, 5e-3, log=True)
                h1 = trial.suggest_int("h1", 32, 256, step=32)
                h2 = trial.suggest_int("h2", 16, 160, step=16)
                params["hidden_layer_sizes"] = (h1, h2)
                # Trend regularization
                params["lambda_1"] = trial.suggest_float("lambda_1", 1e-9, 1e-3, log=True)
                params["lambda_2"] = trial.suggest_float("lambda_2", 1e-9, 1e-3, log=True)

            # Normalize Optuna best_params format
            if "h1" in fixed_params or "h2" in fixed_params:
                h1 = int(fixed_params.pop("h1", params["hidden_layer_sizes"][0]))
                h2 = int(fixed_params.pop("h2", params["hidden_layer_sizes"][1]))
                fixed_params["hidden_layer_sizes"] = (h1, h2)
            if isinstance(fixed_params.get("hidden_layer_sizes"), list):
                fixed_params["hidden_layer_sizes"] = tuple(int(x) for x in fixed_params["hidden_layer_sizes"])

            params.update(fixed_params)
            return RSMMLPResidualModel(**params)

        raise ValueError(f"Unknown model name: {model_name}")

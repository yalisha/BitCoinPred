#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json
import numpy as np

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover
    XGBRegressor = None
    _XGB_IMPORT_ERROR = exc
else:
    _XGB_IMPORT_ERROR = None

from .. import config
from ..utils import create_next_run_dir
from ..metrics import ensure_finite, compute_diagnostics, targets_to_price
from ..plots import plot_price_series, plot_step_series, plot_bar
from ..common import prepare_datasets, flatten_sequence


@dataclass
class XGBoostParams:
    max_depth: int = 6
    learning_rate: float = 0.03
    n_estimators: int = 1500
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    min_child_weight: float = 1.0
    gamma: float = 0.0
    early_stopping_rounds: int = 100
    seed: int = 20240501


def _check_xgboost() -> None:
    if XGBRegressor is None:
        raise RuntimeError(
            "XGBoost is not installed. Please install xgboost to run this experiment."  # pragma: no cover
        ) from _XGB_IMPORT_ERROR


def run_experiment(overrides: Optional[Dict[str, Any]] = None,
                   global_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    _check_xgboost()

    overrides = dict(overrides or {})
    base_params = XGBoostParams()
    param_updates = {k: v for k, v in overrides.items() if hasattr(base_params, k)}
    params = XGBoostParams(**{**base_params.__dict__, **param_updates})

    prep = prepare_datasets(global_settings, overrides)
    train_seq = prep["train"]
    val_seq = prep["val"]
    test_seq = prep["test"]
    target_type = prep["target_type"]
    settings = prep["settings"]

    quantiles = overrides.get("QUANTILES", config.QUANTILES)

    X_tr, y_tr = flatten_sequence(train_seq)
    X_va, y_va = flatten_sequence(val_seq)
    X_te, y_te = flatten_sequence(test_seq)

    X_tr = X_tr.astype(np.float32)
    X_va = X_va.astype(np.float32)
    X_te = X_te.astype(np.float32)

    ensure_finite("train_features", X_tr)

    horizon = y_tr.shape[1]
    q_val = np.zeros((y_va.shape[0], horizon, len(quantiles)), dtype=float)
    q_test = np.zeros((y_te.shape[0], horizon, len(quantiles)), dtype=float)

    feature_importances: list[np.ndarray] = []
    model_details: Dict[str, Dict[str, Any]] = {}

    for h in range(horizon):
        y_tr_h = y_tr[:, h]
        y_va_h = y_va[:, h]

        model = XGBRegressor(
            max_depth=params.max_depth,
            learning_rate=params.learning_rate,
            n_estimators=params.n_estimators,
            subsample=params.subsample,
            colsample_bytree=params.colsample_bytree,
            reg_lambda=params.reg_lambda,
            reg_alpha=params.reg_alpha,
            min_child_weight=params.min_child_weight,
            gamma=params.gamma,
            random_state=params.seed + h,
            objective="reg:squarederror",
            verbosity=0,
        )

        model.fit(
            X_tr,
            y_tr_h,
            eval_set=[(X_va, y_va_h)],
            eval_metric="rmse",
            early_stopping_rounds=params.early_stopping_rounds if params.early_stopping_rounds > 0 else None,
            verbose=False,
        )

        best_iter = getattr(model, "best_iteration", None)
        if best_iter is None:
            best_iter = model.n_estimators

        val_pred = model.predict(X_va, iteration_range=(0, best_iter))
        test_pred = model.predict(X_te, iteration_range=(0, best_iter))

        for qi in range(len(quantiles)):
            q_val[:, h, qi] = val_pred
            q_test[:, h, qi] = test_pred

        feature_importances.append(model.feature_importances_)
        model_details[f"h{h+1}"] = {
            "best_iteration": int(best_iter),
            "train_samples": int(y_tr.shape[0]),
        }

    ensure_finite("val_predictions", q_val)
    ensure_finite("test_predictions", q_test)

    val_diag = compute_diagnostics(val_seq.y, q_val, val_seq.c0, target_type, quantiles, dates=val_seq.dates_fut)
    test_diag = compute_diagnostics(test_seq.y, q_test, test_seq.c0, target_type, quantiles, dates=test_seq.dates_fut)

    metrics_flat: Dict[str, Any] = {}
    for split, diag in ("val", val_diag), ("test", test_diag):
        for key, value in diag["metrics"].items():
            metrics_flat[f"{split}_{key}"] = float(value)
    if val_diag.get("segments"):
        metrics_flat["segments"] = {"val": val_diag["segments"], "test": test_diag.get("segments", {})}

    mean_importance = np.mean(np.stack(feature_importances, axis=0), axis=0).tolist()
    metrics_flat.update({
        "model": "xgboost",
        "hyperparams": params.__dict__,
        "feature_importance_gain": mean_importance,
    })

    out_root = Path(settings["out_dir"]) / overrides.get("output_subdir", "xgboost")
    run_dir = create_next_run_dir(out_root)
    (run_dir / "pred").mkdir(parents=True, exist_ok=True)
    (run_dir / "figs").mkdir(parents=True, exist_ok=True)

    np.save(run_dir / "pred/val_quantiles.npy", q_val)
    np.save(run_dir / "pred/test_quantiles.npy", q_test)

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_flat, f, ensure_ascii=False, indent=2)

    with (run_dir / "model_details.json").open("w", encoding="utf-8") as f:
        json.dump(model_details, f, ensure_ascii=False, indent=2)

    plot_price_series(val_seq.dates_fut,
                      targets_to_price(val_seq.c0, val_seq.y, target_type)[:, 0],
                      targets_to_price(val_seq.c0, q_val[:, :, 0], target_type)[:, 0],
                      "VAL Price h=1", run_dir / "figs/val_price_h1.png")
    plot_price_series(test_seq.dates_fut,
                      targets_to_price(test_seq.c0, test_seq.y, target_type)[:, 0],
                      targets_to_price(test_seq.c0, q_test[:, :, 0], target_type)[:, 0],
                      "TEST Price h=1", run_dir / "figs/test_price_h1.png")

    h_last = test_seq.y.shape[1] - 1
    plot_step_series(val_seq.dates_fut,
                     targets_to_price(val_seq.c0, val_seq.y, target_type)[:, h_last],
                     targets_to_price(val_seq.c0, q_val[:, :, 0], target_type)[:, h_last],
                     h_last + 1, "VAL Price", run_dir / "figs/val_price_hLast.png")
    plot_step_series(test_seq.dates_fut,
                     targets_to_price(test_seq.c0, test_seq.y, target_type)[:, h_last],
                     targets_to_price(test_seq.c0, q_test[:, :, 0], target_type)[:, h_last],
                     h_last + 1, "TEST Price", run_dir / "figs/test_price_hLast.png")

    plot_bar(np.array(val_diag["coverage_steps"]), "VAL Coverage", run_dir / "figs/val_coverage.png", ylabel="coverage")
    plot_bar(np.array(test_diag["coverage_steps"]), "TEST Coverage", run_dir / "figs/test_coverage.png", ylabel="coverage")

    print(f"[XGBoost] Finished. Results saved to {run_dir}")

    return {
        "run_dir": run_dir,
        "metrics": metrics_flat,
        "metrics_path": run_dir / "metrics.json",
        "model_details_path": run_dir / "model_details.json",
        "applied_overrides": overrides,
        "settings": settings,
    }


if __name__ == "__main__":
    run_experiment()


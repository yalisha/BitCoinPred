#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd
from device import get_torch_device, device_as_str

from . import config
from .utils import create_next_run_dir
from .data import load_raw, build_feature_frame, split_by_date, make_sequences, SeqData
from .model_tft import TFTCfg, TFTMultiHQuantile, train_one, predict
from .plots import plot_price_series, plot_step_series, plot_quantile_band, plot_step_metric, plot_bar


def _ensure_finite(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        bad = np.argwhere(~np.isfinite(arr))
        loc = bad[0].tolist() if bad.size else []
        raise ValueError(f"Non-finite values detected in {name} at index {loc}")


def to_price_from_target(c0: np.ndarray, y: np.ndarray, target_type: str) -> np.ndarray:
    t = target_type.lower()
    if t == 'price':
        return y
    if t == 'logprice':
        return np.exp(y)
    if t == 'rel_logprice':
        return c0 * np.exp(y)
    if t == 'logret':
        return c0 * np.exp(y)
    if t == 'pctret':
        return c0 * (1.0 + y)
    raise ValueError('unknown target_type')


def _fit_scalers(Xo: np.ndarray, Xk: np.ndarray, Xs: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    stats: Dict[str, np.ndarray] = {}
    stats["mu_o"] = Xo.mean(axis=(0, 1), keepdims=True)
    stats["sd_o"] = Xo.std(axis=(0, 1), keepdims=True)
    stats["sd_o"][stats["sd_o"] < 1e-8] = 1.0

    stats["mu_k"] = Xk.mean(axis=(0, 1), keepdims=True)
    stats["sd_k"] = Xk.std(axis=(0, 1), keepdims=True)
    stats["sd_k"][stats["sd_k"] < 1e-8] = 1.0

    stats["mu_s"] = Xs.mean(axis=0, keepdims=True)
    stats["sd_s"] = Xs.std(axis=0, keepdims=True)
    stats["sd_s"][stats["sd_s"] < 1e-8] = 1.0

    stats["y_mu"] = y.mean(axis=0, keepdims=True)
    stats["y_sd"] = y.std(axis=0, keepdims=True)
    stats["y_sd"][stats["y_sd"] < 1e-8] = 1.0
    return stats


def _normalize_dataset(stats: Dict[str, np.ndarray], *,
                       Xo: Optional[np.ndarray] = None,
                       Xk: Optional[np.ndarray] = None,
                       Xs: Optional[np.ndarray] = None,
                       y: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    def _norm(arr: Optional[np.ndarray], mu: np.ndarray, sd: np.ndarray) -> Optional[np.ndarray]:
        if arr is None:
            return None
        return (np.asarray(arr, dtype=float) - mu) / sd

    return (
        _norm(Xo, stats["mu_o"], stats["sd_o"]),
        _norm(Xk, stats["mu_k"], stats["sd_k"]),
        _norm(Xs, stats["mu_s"], stats["sd_s"]),
        _norm(y, stats["y_mu"], stats["y_sd"]),
    )


def _denormalize_targets(q_pred_z: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    y_mu = stats["y_mu"].reshape(1, stats["y_mu"].shape[1], 1)
    y_sd = stats["y_sd"].reshape(1, stats["y_sd"].shape[1], 1)
    return q_pred_z * y_sd + y_mu


def _resolve_nhead(d_model: int, requested: int) -> int:
    requested = max(1, int(requested))
    if d_model % requested == 0:
        return requested
    for n in range(requested, 0, -1):
        if d_model % n == 0:
            return n
    for n in range(requested + 1, d_model + 1):
        if d_model % n == 0:
            return n
    return 1


def _format_ts(value) -> str:
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def _build_time_series_folds(n_samples: int, val_len: int, n_folds: int, min_train: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_folds):
        val_end = n_samples - fold * val_len
        val_start = val_end - val_len
        if val_start <= 0:
            break
        if val_start < min_train:
            break
        train_idx = np.arange(0, val_start)
        val_idx = np.arange(val_start, val_end)
        if len(val_idx) == 0:
            continue
        folds.append((train_idx, val_idx))
    folds.reverse()
    return folds


def _compute_fold_metrics(y_true: np.ndarray, q_pred: np.ndarray, c0: np.ndarray,
                          target_type: str, quantiles: List[float]) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    q_index = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    yhat = q_pred[..., q_index]
    price_true_h1 = to_price_from_target(c0, y_true[:, 0], target_type)
    price_pred_h1 = to_price_from_target(c0, yhat[:, 0], target_type)

    def rmse(a, b):
        a = np.asarray(a); b = np.asarray(b)
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def mae(a, b):
        a = np.asarray(a); b = np.asarray(b)
        return float(np.mean(np.abs(a - b)))

    rmse_steps: List[float] = []
    mae_steps: List[float] = []
    cov_steps: List[float] = []
    for h in range(y_true.shape[1]):
        price_true = to_price_from_target(c0, y_true[:, h], target_type)
        price_pred = to_price_from_target(c0, yhat[:, h], target_type)
        rmse_steps.append(rmse(price_true, price_pred))
        mae_steps.append(mae(price_true, price_pred))
        lo = q_pred[:, h, 0]
        hi = q_pred[:, h, -1]
        cov_steps.append(float(np.mean((y_true[:, h] >= lo) & (y_true[:, h] <= hi))))

    metrics = {
        "val_rmse_h1_price": rmse(price_true_h1, price_pred_h1),
        "val_mae_h1_price": mae(price_true_h1, price_pred_h1),
        "val_rmse_avg_steps": float(np.mean(rmse_steps)),
        "val_mae_avg_steps": float(np.mean(mae_steps)),
        "val_cov_avg_steps": float(np.mean(cov_steps)),
    }
    detail = {
        "rmse_steps": [float(v) for v in rmse_steps],
        "mae_steps": [float(v) for v in mae_steps],
        "coverage_steps": [float(v) for v in cov_steps],
    }
    return metrics, detail


def _compute_diagnostics(y_true: np.ndarray, q_pred: np.ndarray, c0: np.ndarray,
                         target_type: str, quantiles: List[float]) -> Dict[str, object]:
    q_index = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    yhat = q_pred[..., q_index]
    price_true_h1 = to_price_from_target(c0, y_true[:, 0], target_type)
    price_pred_h1 = to_price_from_target(c0, yhat[:, 0], target_type)

    def rmse(a, b):
        a = np.asarray(a); b = np.asarray(b)
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def mae(a, b):
        a = np.asarray(a); b = np.asarray(b)
        return float(np.mean(np.abs(a - b)))

    rmse_steps: List[float] = []
    mae_steps: List[float] = []
    cov_steps: List[float] = []
    for h in range(y_true.shape[1]):
        price_true = to_price_from_target(c0, y_true[:, h], target_type)
        price_pred = to_price_from_target(c0, yhat[:, h], target_type)
        rmse_steps.append(rmse(price_true, price_pred))
        mae_steps.append(mae(price_true, price_pred))
        lo = q_pred[:, h, 0]
        hi = q_pred[:, h, -1]
        cov_steps.append(float(np.mean((y_true[:, h] >= lo) & (y_true[:, h] <= hi))))

    metrics = {
        "rmse_h1_price": rmse(price_true_h1, price_pred_h1),
        "mae_h1_price": mae(price_true_h1, price_pred_h1),
        "rmse_avg_steps": float(np.mean(rmse_steps)),
        "mae_avg_steps": float(np.mean(mae_steps)),
        "coverage_avg_steps": float(np.mean(cov_steps)),
    }
    return {
        "yhat": yhat,
        "price_true_h1": price_true_h1,
        "price_pred_h1": price_pred_h1,
        "rmse_steps": rmse_steps,
        "mae_steps": mae_steps,
        "coverage_steps": cov_steps,
        "metrics": metrics,
    }


def run_time_series_cv(base_cfg: TFTCfg,
                       seq_cv: SeqData,
                       step_weights_np: Optional[np.ndarray],
                       device,
                       train_params: Dict[str, float],
                       target_type: str,
                       quantiles: List[float],
                       val_window_len: int,
                       verbose: bool = True) -> Optional[Dict[str, object]]:
    n_folds = int(train_params.get("cv_folds", 0))
    if n_folds < 2:
        return None
    total = len(seq_cv.y)
    if total < n_folds + 1:
        if verbose:
            print("[CV] Skipped (insufficient samples).")
        return None
    if val_window_len <= 0 or val_window_len * n_folds >= total:
        val_window_len = max(1, total // (n_folds + 1))
    folds = _build_time_series_folds(total, val_window_len, n_folds, int(train_params.get("min_train", 1)))
    if not folds:
        if verbose:
            print("[CV] Skipped (insufficient history for requested folds).")
        return None

    results: List[Dict[str, object]] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(folds, start=1):
        stats = _fit_scalers(seq_cv.X_obs[tr_idx], seq_cv.X_known[tr_idx], seq_cv.X_static[tr_idx], seq_cv.y[tr_idx])
        Xo_tr, Xk_tr, Xs_tr, y_tr = _normalize_dataset(stats, Xo=seq_cv.X_obs[tr_idx], Xk=seq_cv.X_known[tr_idx], Xs=seq_cv.X_static[tr_idx], y=seq_cv.y[tr_idx])
        Xo_va, Xk_va, Xs_va, y_va = _normalize_dataset(stats, Xo=seq_cv.X_obs[va_idx], Xk=seq_cv.X_known[va_idx], Xs=seq_cv.X_static[va_idx], y=seq_cv.y[va_idx])

        fold_cfg = replace(base_cfg)
        model = TFTMultiHQuantile(num_obs=seq_cv.X_obs.shape[-1], num_kn=seq_cv.X_known.shape[-1], num_static=seq_cv.X_static.shape[-1], cfg=fold_cfg)
        model = train_one(
            fold_cfg,
            model,
            (Xo_tr, Xk_tr, Xs_tr, y_tr),
            (Xo_va, Xk_va, Xs_va, y_va),
            epochs=int(train_params.get("epochs", config.CV_EPOCHS)),
            lr=float(train_params.get("lr", config.LR)),
            batch_size=int(train_params.get("batch_size", config.BATCH_SIZE)),
            device=device,
            step_weights_np=step_weights_np,
            mse_aux_weight=float(train_params.get("mse_aux_weight", config.MSE_AUX_WEIGHT)),
            grad_clip_norm=float(train_params.get("grad_clip", config.GRAD_CLIP)),
            weight_decay=float(train_params.get("weight_decay", config.WEIGHT_DECAY)),
        )

        q_va_z, _ = predict(model, Xo_va, Xk_va, Xs_va, device=device)
        q_va = _denormalize_targets(q_va_z, stats)
        metrics, detail = _compute_fold_metrics(seq_cv.y[va_idx], q_va, seq_cv.c0[va_idx], target_type, quantiles)

        fold_info = {
            "fold": fold_idx,
            "train_samples": int(len(tr_idx)),
            "val_samples": int(len(va_idx)),
            "train_range": {
                "start": _format_ts(seq_cv.dates_t.iloc[tr_idx[0]]) if len(tr_idx) else "",
                "end": _format_ts(seq_cv.dates_t.iloc[tr_idx[-1]]) if len(tr_idx) else "",
            },
            "val_range": {
                "start": _format_ts(seq_cv.dates_t.iloc[va_idx[0]]),
                "end": _format_ts(seq_cv.dates_t.iloc[va_idx[-1]]),
            },
            "metrics": {k: float(v) for k, v in metrics.items()},
            "details": detail,
        }
        results.append(fold_info)
        if verbose:
            print(f"[CV] Fold {fold_idx}/{len(folds)} | train={len(tr_idx)} | val={len(va_idx)} | rmse_h1={metrics['val_rmse_h1_price']:.4f}")

    if not results:
        return None

    metric_keys = results[0]["metrics"].keys()
    mean_metrics = {k: float(np.mean([f["metrics"][k] for f in results])) for k in metric_keys}
    std_metrics = {k: float(np.std([f["metrics"][k] for f in results], ddof=0)) for k in metric_keys}
    summary = {
        "fold_count": len(results),
        "mean": mean_metrics,
        "std": std_metrics,
        "folds": results,
        "val_window": int(val_window_len),
    }
    if verbose:
        print(f"[CV] Mean val_rmse_h1_price={mean_metrics['val_rmse_h1_price']:.4f} (±{std_metrics['val_rmse_h1_price']:.4f})")
    return summary


def run_optuna_study(base_cfg: TFTCfg,
                     seq_cv: SeqData,
                     step_weights_np: Optional[np.ndarray],
                     device,
                     train_params: Dict[str, float],
                     target_type: str,
                     quantiles: List[float],
                     val_window_len: int) -> Optional[Dict[str, object]]:
    try:
        import optuna
    except ImportError:
        print("Optuna 未安装，跳过超参搜索。")
        return None

    study_kwargs = {"direction": "minimize", "study_name": config.OPTUNA_STUDY}
    if config.OPTUNA_STORAGE:
        study_kwargs["storage"] = config.OPTUNA_STORAGE
        study_kwargs["load_if_exists"] = True
    study = optuna.create_study(**study_kwargs)

    max_trials = int(config.OPTUNA_TRIALS)
    timeout = int(config.OPTUNA_TIMEOUT) if config.OPTUNA_TIMEOUT else None

    def objective(trial: "optuna.trial.Trial") -> float:
        hidden_size = trial.suggest_int("hidden_size", 64, 256, step=32)
        nhead_candidates = [n for n in [1, 2, 4, 8, 16] if n <= hidden_size and hidden_size % n == 0]
        if not nhead_candidates:
            nhead_candidates = [1]
        nhead = trial.suggest_categorical("nhead", nhead_candidates)
        dropout = trial.suggest_float("dropout", 0.1, 0.35, step=0.05)
        attn_dropout = trial.suggest_float("attn_dropout", 0.1, 0.3, step=0.05)
        ff_dropout = trial.suggest_float("ff_dropout", 0.05, 0.3, step=0.05)
        lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        grad_clip = trial.suggest_float("grad_clip", 1.0, 5.0)
        mse_aux = trial.suggest_float("mse_aux_weight", 0.05, 0.3)
        batch_size = trial.suggest_categorical("batch_size", [16, 24, 32, 48])

        cfg_trial = replace(
            base_cfg,
            d_model=hidden_size,
            d_hidden=max(2 * hidden_size, 64),
            dropout=dropout,
            nhead=nhead,
            lstm_layers=lstm_layers,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )

        trial_params = train_params.copy()
        trial_params.update({
            "lr": lr,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "mse_aux_weight": mse_aux,
            "batch_size": batch_size,
            "cv_folds": max(2, min(train_params.get("cv_folds", config.CV_FOLDS), 3)),
        })

        cv_result = run_time_series_cv(
            cfg_trial,
            seq_cv,
            step_weights_np,
            device,
            trial_params,
            target_type,
            quantiles,
            val_window_len,
            verbose=False,
        )
        if not cv_result:
            raise optuna.TrialPruned("insufficient data for CV")
        return cv_result["mean"]["val_rmse_h1_price"]

    try:
        study.optimize(objective, n_trials=max_trials, timeout=timeout, show_progress_bar=False)
    except Exception as exc:  # pragma: no cover - optuna runtime guard
        print(f"Optuna 搜索失败: {exc}")
        return {"error": str(exc)}

    if not study.trials:
        return None

    best_trial = study.best_trial if study.best_trial else None
    summary = {
        "n_trials": len(study.trials),
        "best_value": float(best_trial.value) if best_trial and best_trial.value is not None else None,
        "best_params": best_trial.params if best_trial else None,
        "direction": study.direction.name,
    }
    history: List[Dict[str, object]] = []
    for t in study.trials:
        if t.value is None:
            continue
        history.append({
            "number": t.number,
            "value": float(t.value),
            "params": t.params,
            "state": t.state.name,
        })
    summary["history"] = history
    return summary


def main():
    raw = load_raw(config.DATA_CSV)
    feat = build_feature_frame(raw, target_type=config.PRED_TARGET)
    parts = split_by_date(feat, config.TRAIN_END, config.VAL_END)

    tr_seq = make_sequences(parts['train'], config.SEQ_LEN, config.HORIZON, config.PRED_TARGET)
    va_seq = make_sequences(parts['val'],   config.SEQ_LEN, config.HORIZON, config.PRED_TARGET)
    te_seq = make_sequences(parts['test'],  config.SEQ_LEN, config.HORIZON, config.PRED_TARGET)

    need_cv_seq = ((config.CV_ENABLED and config.CV_FOLDS >= 2) or config.ENABLE_OPTUNA)
    cv_seq: Optional[SeqData] = None
    if need_cv_seq:
        cv_df = pd.concat([parts['train'], parts['val']], ignore_index=True)
        cv_df = cv_df.sort_values('date').reset_index(drop=True)
        cv_seq = make_sequences(cv_df, config.SEQ_LEN, config.HORIZON, config.PRED_TARGET)

    # 步长加权
    if config.STEP_WEIGHT_MODE == 'linear':
        step_w = np.arange(1, config.HORIZON + 1, dtype=float)
        step_w = 1.0 + config.STEP_WEIGHT_ALPHA * step_w
    elif config.STEP_WEIGHT_MODE == 'square':
        step_w = (np.arange(1, config.HORIZON + 1, dtype=float) / config.HORIZON) ** 2
        step_w = 1.0 + step_w
    else:
        step_w = None

    torch_device = get_torch_device()
    device_name = device_as_str(torch_device)
    print(f"Using torch device: {device_name}")

    requested_heads = min(config.NHEAD, max(1, config.HIDDEN_SIZE // 8))
    nhead = _resolve_nhead(config.HIDDEN_SIZE, requested_heads)

    cfg = TFTCfg(
        d_model=config.HIDDEN_SIZE,
        d_hidden=max(2 * config.HIDDEN_SIZE, 64),
        nhead=nhead,
        dropout=config.DROPOUT,
        lstm_layers=config.LSTM_LAYERS,
        attn_dropout=config.ATTN_DROPOUT,
        ff_dropout=config.FF_DROPOUT,
        horizon=config.HORIZON,
        quantiles=config.QUANTILES,
        device=device_name,
    )

    cv_summary = None
    val_window_len = config.CV_VAL_SAMPLES if config.CV_VAL_SAMPLES > 0 else len(va_seq.y)
    if config.CV_ENABLED and cv_seq is not None:
        cv_params = {
            "epochs": min(config.EPOCHS, config.CV_EPOCHS),
            "lr": config.LR,
            "batch_size": config.BATCH_SIZE,
            "mse_aux_weight": config.MSE_AUX_WEIGHT,
            "grad_clip": config.GRAD_CLIP,
            "weight_decay": config.WEIGHT_DECAY,
            "cv_folds": config.CV_FOLDS,
            "min_train": config.CV_MIN_TRAIN_SAMPLES,
        }
        cv_summary = run_time_series_cv(cfg, cv_seq, step_w, torch_device, cv_params, config.PRED_TARGET, config.QUANTILES, val_window_len, verbose=True)

    optuna_summary = None
    if config.ENABLE_OPTUNA and cv_seq is not None:
        optuna_params = {
            "epochs": min(config.EPOCHS, config.CV_EPOCHS),
            "lr": config.LR,
            "batch_size": config.BATCH_SIZE,
            "mse_aux_weight": config.MSE_AUX_WEIGHT,
            "grad_clip": config.GRAD_CLIP,
            "weight_decay": config.WEIGHT_DECAY,
            "cv_folds": config.CV_FOLDS,
            "min_train": config.CV_MIN_TRAIN_SAMPLES,
        }
        optuna_summary = run_optuna_study(cfg, cv_seq, step_w, torch_device, optuna_params, config.PRED_TARGET, config.QUANTILES, val_window_len)
        if optuna_summary and optuna_summary.get("best_params"):
            print("Optuna 最佳参数:", optuna_summary["best_params"])

    # 标准化并训练最终模型
    stats = _fit_scalers(tr_seq.X_obs, tr_seq.X_known, tr_seq.X_static, tr_seq.y)
    Xo_tr, Xk_tr, Xs_tr, y_tr = _normalize_dataset(stats, Xo=tr_seq.X_obs, Xk=tr_seq.X_known, Xs=tr_seq.X_static, y=tr_seq.y)
    Xo_va, Xk_va, Xs_va, y_va = _normalize_dataset(stats, Xo=va_seq.X_obs, Xk=va_seq.X_known, Xs=va_seq.X_static, y=va_seq.y)
    Xo_te, Xk_te, Xs_te, _ = _normalize_dataset(stats, Xo=te_seq.X_obs, Xk=te_seq.X_known, Xs=te_seq.X_static, y=None)

    model = TFTMultiHQuantile(num_obs=Xo_tr.shape[-1], num_kn=Xk_tr.shape[-1], num_static=Xs_tr.shape[-1], cfg=cfg)
    model = train_one(
        cfg,
        model,
        (Xo_tr, Xk_tr, Xs_tr, y_tr),
        (Xo_va, Xk_va, Xs_va, y_va),
        epochs=config.EPOCHS,
        lr=config.LR,
        batch_size=config.BATCH_SIZE,
        device=torch_device,
        step_weights_np=step_w,
        mse_aux_weight=config.MSE_AUX_WEIGHT,
        grad_clip_norm=config.GRAD_CLIP,
        weight_decay=config.WEIGHT_DECAY,
    )

    q_va_z, det_va = predict(model, Xo_va, Xk_va, Xs_va, device=torch_device)
    q_te_z, det_te = predict(model, Xo_te, Xk_te, Xs_te, device=torch_device)
    q_va = _denormalize_targets(q_va_z, stats)
    q_te = _denormalize_targets(q_te_z, stats)
    _ensure_finite("val_quantiles", q_va)
    _ensure_finite("test_quantiles", q_te)

    val_diag = _compute_diagnostics(va_seq.y, q_va, va_seq.c0, config.PRED_TARGET, config.QUANTILES)
    test_diag = _compute_diagnostics(te_seq.y, q_te, te_seq.c0, config.PRED_TARGET, config.QUANTILES)

    metrics = {
        "val_rmse_h1_price": val_diag["metrics"]["rmse_h1_price"],
        "val_mae_h1_price": val_diag["metrics"]["mae_h1_price"],
        "val_rmse_avg_steps": val_diag["metrics"]["rmse_avg_steps"],
        "val_mae_avg_steps": val_diag["metrics"]["mae_avg_steps"],
        "val_cov_avg_steps": val_diag["metrics"]["coverage_avg_steps"],
        "test_rmse_h1_price": test_diag["metrics"]["rmse_h1_price"],
        "test_mae_h1_price": test_diag["metrics"]["mae_h1_price"],
        "test_rmse_avg_steps": test_diag["metrics"]["rmse_avg_steps"],
        "test_mae_avg_steps": test_diag["metrics"]["mae_avg_steps"],
        "test_cov_avg_steps": test_diag["metrics"]["coverage_avg_steps"],
    }
    metrics = {k: float(v) for k, v in metrics.items()}

    run_dir = create_next_run_dir(config.OUT_DIR)
    (run_dir / "pred").mkdir(parents=True, exist_ok=True)
    (run_dir / "figs").mkdir(parents=True, exist_ok=True)

    np.save(run_dir / "pred/val_quantiles.npy", q_va)
    np.save(run_dir / "pred/test_quantiles.npy", q_te)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if cv_summary:
        with (run_dir / "cv_summary.json").open("w", encoding="utf-8") as f:
            json.dump(cv_summary, f, ensure_ascii=False, indent=2)
    if optuna_summary:
        with (run_dir / "optuna_summary.json").open("w", encoding="utf-8") as f:
            json.dump(optuna_summary, f, ensure_ascii=False, indent=2)

    # 绘图：step=1 的价格时序对比 + 分位带（验证/测试）
    plot_price_series(va_seq.dates_fut, val_diag["price_true_h1"], val_diag["price_pred_h1"], "VAL Price h=1 (q50)", run_dir / "figs/val_price_h1.png")
    plot_price_series(te_seq.dates_fut, test_diag["price_true_h1"], test_diag["price_pred_h1"], "TEST Price h=1 (q50)", run_dir / "figs/test_price_h1.png")

    q_index = config.QUANTILES.index(0.5) if 0.5 in config.QUANTILES else len(config.QUANTILES) // 2
    ql_va, qm_va, qh_va = q_va[:, 0, 0], q_va[:, 0, q_index], q_va[:, 0, -1]
    ql_te, qm_te, qh_te = q_te[:, 0, 0], q_te[:, 0, q_index], q_te[:, 0, -1]
    ql_p = to_price_from_target(va_seq.c0, ql_va, config.PRED_TARGET)
    qm_p = to_price_from_target(va_seq.c0, qm_va, config.PRED_TARGET)
    qh_p = to_price_from_target(va_seq.c0, qh_va, config.PRED_TARGET)
    plot_quantile_band(va_seq.dates_fut, ql_p, qm_p, qh_p, "VAL Quantile band h=1", run_dir / "figs/val_band_h1.png")

    ql_t = to_price_from_target(te_seq.c0, ql_te, config.PRED_TARGET)
    qm_t = to_price_from_target(te_seq.c0, qm_te, config.PRED_TARGET)
    qh_t = to_price_from_target(te_seq.c0, qh_te, config.PRED_TARGET)
    plot_quantile_band(te_seq.dates_fut, ql_t, qm_t, qh_t, "TEST Quantile band h=1", run_dir / "figs/test_band_h1.png")

    h_last = config.HORIZON - 1
    price_true_va_last = to_price_from_target(va_seq.c0, va_seq.y[:, h_last], config.PRED_TARGET)
    price_pred_va_last = to_price_from_target(va_seq.c0, val_diag["yhat"][:, h_last], config.PRED_TARGET)
    plot_step_series(va_seq.dates_fut, price_true_va_last, price_pred_va_last, h_last + 1, "VAL Price", run_dir / "figs/val_price_hLast.png")

    price_true_te_last = to_price_from_target(te_seq.c0, te_seq.y[:, h_last], config.PRED_TARGET)
    price_pred_te_last = to_price_from_target(te_seq.c0, test_diag["yhat"][:, h_last], config.PRED_TARGET)
    plot_step_series(te_seq.dates_fut, price_true_te_last, price_pred_te_last, h_last + 1, "TEST Price", run_dir / "figs/test_price_hLast.png")

    np.save(run_dir / "pred/val_rmse_steps.npy", np.array(val_diag["rmse_steps"]))
    np.save(run_dir / "pred/test_rmse_steps.npy", np.array(test_diag["rmse_steps"]))
    np.save(run_dir / "pred/val_cov_steps.npy", np.array(val_diag["coverage_steps"]))
    np.save(run_dir / "pred/test_cov_steps.npy", np.array(test_diag["coverage_steps"]))
    plot_step_metric(np.array(val_diag["rmse_steps"]), "VAL RMSE per step (price)", run_dir / "figs/val_rmse_steps.png")
    plot_step_metric(np.array(test_diag["rmse_steps"]), "TEST RMSE per step (price)", run_dir / "figs/test_rmse_steps.png")
    plot_bar(np.array(val_diag["coverage_steps"]), "VAL Coverage q10–q90 (target)", run_dir / "figs/val_coverage.png", ylabel='coverage')
    plot_bar(np.array(test_diag["coverage_steps"]), "TEST Coverage q10–q90 (target)", run_dir / "figs/test_coverage.png", ylabel='coverage')

    if config.CALIBRATE_Q50:
        a_coef: List[float] = []
        b_coef: List[float] = []
        yhat_va_med = val_diag["yhat"].copy()
        yhat_te_med = test_diag["yhat"].copy()
        for h in range(config.HORIZON):
            Y = va_seq.y[:, h]
            X = yhat_va_med[:, h]
            A = np.vstack([X, np.ones_like(X)]).T
            sol, *_ = np.linalg.lstsq(A, Y, rcond=None)
            a_coef.append(float(sol[0]))
            b_coef.append(float(sol[1]))
            yhat_te_med[:, h] = a_coef[-1] * yhat_te_med[:, h] + b_coef[-1]
        _ensure_finite("calibrated_q50_val", yhat_va_med)
        _ensure_finite("calibrated_q50_test", yhat_te_med)
        test_rmse_steps_cal = []
        for h in range(config.HORIZON):
            test_rmse_steps_cal.append(float(np.sqrt(np.mean((to_price_from_target(te_seq.c0, te_seq.y[:, h], config.PRED_TARGET) -
                                                              to_price_from_target(te_seq.c0, yhat_te_med[:, h], config.PRED_TARGET)) ** 2))))
        np.save(run_dir / "pred/test_rmse_steps_calibrated.npy", np.array(test_rmse_steps_cal))
        plot_step_metric(np.array(test_rmse_steps_cal), "TEST RMSE per step (calibrated)", run_dir / "figs/test_rmse_steps_calibrated.png")
        with (run_dir / "calibration_q50.json").open("w", encoding="utf-8") as f:
            json.dump({"a": a_coef, "b": b_coef}, f, ensure_ascii=False, indent=2)

    print("ver2 运行完成。结果目录:", str(run_dir))


if __name__ == "__main__":
    main()

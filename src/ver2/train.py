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
from .insights import (
    save_variable_weights,
    save_attn_weights,
    plot_variable_bar,
    plot_time_heatmap,
    plot_attention_heatmap,
)


def _ensure_finite(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        bad = np.argwhere(~np.isfinite(arr))
        loc = bad[0].tolist() if bad.size else []
        raise ValueError(f"Non-finite values detected in {name} at index {loc}")


def targets_to_price(c0: np.ndarray, y: np.ndarray, target_type: str) -> np.ndarray:
    """Convert target space values to price trajectories for each horizon."""
    target_type = target_type.lower()
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    c0_arr = np.asarray(c0, dtype=float).reshape(-1, 1)

    if target_type == 'price':
        return y_arr
    if target_type == 'logprice':
        return np.exp(y_arr)
    if target_type == 'rel_logprice':
        return c0_arr * np.exp(y_arr)
    if target_type == 'logret':
        cum = np.cumsum(y_arr, axis=1)
        return c0_arr * np.exp(cum)
    if target_type == 'pctret':
        cum = np.cumprod(1.0 + y_arr, axis=1)
        return c0_arr * cum
    raise ValueError('unknown target_type')


def to_price_from_target(c0: np.ndarray, y: np.ndarray, target_type: str) -> np.ndarray:
    prices = targets_to_price(c0, y, target_type)
    if np.asarray(y).ndim == 1:
        return prices[:, 0]
    return prices


def _summarize_weights(w: Optional[np.ndarray], feature_names: List[str]) -> Optional[Dict[str, object]]:
    if w is None:
        return None
    w = np.asarray(w, dtype=float)
    if w.ndim < 2:
        w = w.reshape(1, -1)
    mean = w.mean(axis=tuple(range(w.ndim - 1)))
    time_mean = w.mean(axis=0)
    feature_names = list(feature_names)
    if len(feature_names) != mean.shape[-1]:
        feature_names = [f"f{i}" for i in range(mean.shape[-1])]
    order = np.argsort(-mean)
    top_features = [{"name": feature_names[i], "weight": float(mean[i])} for i in order[:10]]
    return {
        "feature_names": feature_names,
        "mean": mean.tolist(),
        "time_mean": np.squeeze(time_mean).tolist() if time_mean.ndim > 1 else time_mean.tolist(),
        "top_features": top_features,
    }


def _summarize_tft_details(details: Dict[str, object], feat_names: Dict[str, List[str]]) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    w_obs = details.get("w_obs")
    w_kn = details.get("w_kn")
    w_st = details.get("w_st")
    attn = details.get("attn")

    if w_obs is not None:
        obs_summary = _summarize_weights(w_obs, feat_names.get("obs", []))
        if obs_summary:
            summary["observed"] = obs_summary
    if w_kn is not None:
        kn_summary = _summarize_weights(w_kn, feat_names.get("known", []))
        if kn_summary:
            summary["known"] = kn_summary
    if w_st is not None:
        st_summary = _summarize_weights(w_st, feat_names.get("static", []))
        if st_summary:
            summary["static"] = st_summary
    if attn is not None:
        attn = np.asarray(attn, dtype=float)
        attn_mean = attn.mean(axis=0)
        summary["attention"] = {
            "mean": attn_mean.tolist(),
            "shape": list(attn_mean.shape),
        }
    return summary


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


def _compute_metric_summary(y_true: np.ndarray,
                            q_pred: np.ndarray,
                            c0: np.ndarray,
                            target_type: str,
                            quantiles: List[float]) -> Dict[str, object]:
    q_index = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
    yhat = q_pred[..., q_index]

    price_true_steps = targets_to_price(c0, y_true, target_type)
    price_pred_steps = targets_to_price(c0, yhat, target_type)

    diff_price = price_pred_steps - price_true_steps
    rmse_steps = np.sqrt(np.mean(diff_price ** 2, axis=0))
    mae_steps = np.mean(np.abs(diff_price), axis=0)

    denom = np.clip(np.abs(price_true_steps), 1e-8, None)
    rel_rmse_steps = np.sqrt(np.mean((diff_price / denom) ** 2, axis=0))
    rel_mae_steps = np.mean(np.abs(diff_price) / denom, axis=0)

    diff_log = y_true - yhat
    log_rmse_steps = np.sqrt(np.mean(diff_log ** 2, axis=0))
    log_mae_steps = np.mean(np.abs(diff_log), axis=0)

    coverage_steps: List[float] = []
    for h in range(y_true.shape[1]):
        lo = q_pred[:, h, 0]
        hi = q_pred[:, h, -1]
        coverage_steps.append(float(np.mean((y_true[:, h] >= lo) & (y_true[:, h] <= hi))))

    metrics = {
        "rmse_h1_price": float(rmse_steps[0]),
        "mae_h1_price": float(mae_steps[0]),
        "rel_rmse_h1_price": float(rel_rmse_steps[0]),
        "rel_mae_h1_price": float(rel_mae_steps[0]),
        "rmse_avg_steps": float(np.mean(rmse_steps)),
        "mae_avg_steps": float(np.mean(mae_steps)),
        "rel_rmse_avg_steps": float(np.mean(rel_rmse_steps)),
        "rel_mae_avg_steps": float(np.mean(rel_mae_steps)),
        "log_rmse_h1": float(log_rmse_steps[0]),
        "log_mae_h1": float(log_mae_steps[0]),
        "log_rmse_avg_steps": float(np.mean(log_rmse_steps)),
        "log_mae_avg_steps": float(np.mean(log_mae_steps)),
        "coverage_h1": float(coverage_steps[0]),
        "coverage_avg_steps": float(np.mean(coverage_steps)),
    }

    return {
        "yhat": yhat,
        "price_true_steps": price_true_steps,
        "price_pred_steps": price_pred_steps,
        "rmse_steps_price": [float(x) for x in rmse_steps],
        "mae_steps_price": [float(x) for x in mae_steps],
        "rel_rmse_steps": [float(x) for x in rel_rmse_steps],
        "rel_mae_steps": [float(x) for x in rel_mae_steps],
        "log_rmse_steps": [float(x) for x in log_rmse_steps],
        "log_mae_steps": [float(x) for x in log_mae_steps],
        "coverage_steps": coverage_steps,
        "metrics": metrics,
    }


def _segment_indices(dates: Optional[pd.Series], seg_cfg: Dict[str, object]) -> np.ndarray:
    if dates is None or len(dates) == 0:
        return np.array([], dtype=int)
    d = pd.to_datetime(dates).reset_index(drop=True)
    n = len(d)
    if "fraction" in seg_cfg and isinstance(seg_cfg["fraction"], (list, tuple)) and len(seg_cfg["fraction"]) == 2:
        start_frac, end_frac = seg_cfg["fraction"]
        start = int(np.floor(max(0.0, min(1.0, start_frac)) * n))
        end = int(np.ceil(max(0.0, min(1.0, end_frac)) * n))
        end = max(end, start + 1)
        end = min(end, n)
        return np.arange(start, end, dtype=int)
    if "days" in seg_cfg:
        days = int(seg_cfg["days"])
        cutoff = d.iloc[-1] - pd.Timedelta(days=days)
        idx = np.where(d >= cutoff)[0]
        return idx.astype(int)
    start = seg_cfg.get("start") or seg_cfg.get("start_date")
    end = seg_cfg.get("end") or seg_cfg.get("end_date")
    if start is None and end is None:
        return np.arange(0, n, dtype=int)
    start_ts = pd.Timestamp(start) if start is not None else d.iloc[0]
    end_ts = pd.Timestamp(end) if end is not None else d.iloc[-1]
    mask = (d >= start_ts) & (d <= end_ts)
    return np.where(mask)[0].astype(int)


def _normalize_anchor_lag(value) -> Optional[int]:
    if value is None:
        return None
    try:
        lag = int(value)
    except (TypeError, ValueError):
        return None
    return lag if lag >= 0 else None


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
    summary = _compute_metric_summary(y_true, q_pred, c0, target_type, quantiles)
    metrics = {f"val_{k}": float(v) for k, v in summary["metrics"].items()}
    detail = {
        "rmse_steps_price": summary["rmse_steps_price"],
        "mae_steps_price": summary["mae_steps_price"],
        "rel_rmse_steps": summary["rel_rmse_steps"],
        "rel_mae_steps": summary["rel_mae_steps"],
        "log_rmse_steps": summary["log_rmse_steps"],
        "log_mae_steps": summary["log_mae_steps"],
        "coverage_steps": summary["coverage_steps"],
    }
    return metrics, detail


def _compute_diagnostics(y_true: np.ndarray, q_pred: np.ndarray, c0: np.ndarray,
                         target_type: str, quantiles: List[float],
                         dates: Optional[pd.Series] = None) -> Dict[str, object]:
    summary = _compute_metric_summary(y_true, q_pred, c0, target_type, quantiles)

    segments: Dict[str, Dict[str, object]] = {}
    seg_cfgs = getattr(config, "EVAL_SEGMENTS", [])
    for seg in seg_cfgs:
        name = seg.get("name") or "segment"
        idx = _segment_indices(dates, seg)
        if idx.size == 0:
            continue
        seg_summary = _compute_metric_summary(y_true[idx], q_pred[idx], c0[idx], target_type, quantiles)
        segments[name] = {
            "count": int(idx.size),
            "metrics": {k: float(v) for k, v in seg_summary["metrics"].items()},
        }

    summary["price_true_h1"] = summary["price_true_steps"][:, 0]
    summary["price_pred_h1"] = summary["price_pred_steps"][:, 0]
    summary["segments"] = segments
    return summary


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
            temporal_smooth_weight=float(train_params.get("temporal_smooth_weight", config.TEMPORAL_SMOOTHING_WEIGHT)),
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
        anchor_choice = trial.suggest_categorical("anchor_lag", [-1, 0, 1, 3, 7, 14])
        anchor_lag = None if int(anchor_choice) < 0 else int(anchor_choice)

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
        temporal_smooth = trial.suggest_float("temporal_smooth_weight", 0.0, 0.15)

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

        trial_params = dict(train_params)
        trial_params.update({
            "lr": lr,
            "weight_decay": weight_decay,
            "grad_clip": grad_clip,
            "mse_aux_weight": mse_aux,
            "batch_size": batch_size,
            "cv_folds": max(2, min(train_params.get("cv_folds", config.CV_FOLDS), 3)),
            "temporal_smooth_weight": temporal_smooth,
        })

        try:
            seq_cv = make_sequences(cv_df, seq_len, horizon, target_type, anchor_lag=anchor_lag)
        except ValueError:
            raise optuna.TrialPruned("sequence construction failed")

        if len(seq_cv.y) < max(3, trial_params.get("cv_folds", 2) + 1):
            raise optuna.TrialPruned("insufficient samples for CV")

        val_window = val_window_len if (val_window_len and val_window_len > 0) else None
        if val_window is None:
            folds = max(2, trial_params.get("cv_folds", 2))
            val_window = max(1, len(seq_cv.y) // (folds + 1))

        cv_result = run_time_series_cv(
            cfg_trial,
            seq_cv,
            step_weights_np,
            device,
            trial_params,
            target_type,
            quantiles,
            val_window,
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
    target_type = config.RETURN_TARGET if config.PREDICT_RETURNS else config.PRED_TARGET
    feat = build_feature_frame(raw, target_type=target_type)
    parts = split_by_date(feat, config.TRAIN_END, config.VAL_END)
    cv_df = None
    if (config.CV_ENABLED and config.CV_FOLDS >= 2) or config.ENABLE_OPTUNA:
        cv_df = pd.concat([parts['train'], parts['val']], ignore_index=True)
        cv_df = cv_df.sort_values('date').reset_index(drop=True)

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

    base_hidden_size = config.HIDDEN_SIZE
    requested_heads = min(config.NHEAD, max(1, base_hidden_size // 8))
    base_nhead = _resolve_nhead(base_hidden_size, requested_heads)

    base_cfg = TFTCfg(
        d_model=base_hidden_size,
        d_hidden=max(2 * base_hidden_size, 64),
        nhead=base_nhead,
        dropout=config.DROPOUT,
        lstm_layers=config.LSTM_LAYERS,
        attn_dropout=config.ATTN_DROPOUT,
        ff_dropout=config.FF_DROPOUT,
        horizon=config.HORIZON,
        quantiles=config.QUANTILES,
        device=device_name,
    )

    base_train_params = {
        "epochs": min(config.EPOCHS, config.CV_EPOCHS),
        "lr": config.LR,
        "batch_size": config.BATCH_SIZE,
        "mse_aux_weight": config.MSE_AUX_WEIGHT,
        "grad_clip": config.GRAD_CLIP,
        "weight_decay": config.WEIGHT_DECAY,
        "cv_folds": config.CV_FOLDS,
        "min_train": config.CV_MIN_TRAIN_SAMPLES,
        "temporal_smooth_weight": config.TEMPORAL_SMOOTHING_WEIGHT,
    }

    optuna_summary = None
    best_params = None
    if config.ENABLE_OPTUNA:
        optuna_summary = run_optuna_study(
            base_cfg,
            cv_df,
            target_type,
            config.SEQ_LEN,
            config.HORIZON,
            step_w,
            torch_device,
            base_train_params,
            config.QUANTILES,
            config.CV_VAL_SAMPLES if config.CV_VAL_SAMPLES > 0 else None,
        )
        if optuna_summary and optuna_summary.get("best_params"):
            best_params = optuna_summary["best_params"]
            print("Optuna 最佳参数:", best_params)

    anchor_lag_final = _normalize_anchor_lag(config.ANCHOR_LAG)
    if best_params and "anchor_lag" in best_params:
        anchor_candidate = int(best_params["anchor_lag"])
        anchor_lag_final = None if anchor_candidate < 0 else anchor_candidate

    hidden_size_final = int(best_params["hidden_size"]) if best_params and "hidden_size" in best_params else config.HIDDEN_SIZE
    dropout_final = float(best_params["dropout"]) if best_params and "dropout" in best_params else config.DROPOUT
    attn_dropout_final = float(best_params["attn_dropout"]) if best_params and "attn_dropout" in best_params else config.ATTN_DROPOUT
    ff_dropout_final = float(best_params["ff_dropout"]) if best_params and "ff_dropout" in best_params else config.FF_DROPOUT
    lstm_layers_final = int(best_params["lstm_layers"]) if best_params and "lstm_layers" in best_params else config.LSTM_LAYERS
    nhead_final = int(best_params["nhead"]) if best_params and "nhead" in best_params else _resolve_nhead(hidden_size_final, min(config.NHEAD, max(1, hidden_size_final // 8)))

    batch_size_final = int(best_params["batch_size"]) if best_params and "batch_size" in best_params else config.BATCH_SIZE
    lr_final = float(best_params["lr"]) if best_params and "lr" in best_params else config.LR
    weight_decay_final = float(best_params["weight_decay"]) if best_params and "weight_decay" in best_params else config.WEIGHT_DECAY
    grad_clip_final = float(best_params["grad_clip"]) if best_params and "grad_clip" in best_params else config.GRAD_CLIP
    mse_aux_final = float(best_params["mse_aux_weight"]) if best_params and "mse_aux_weight" in best_params else config.MSE_AUX_WEIGHT
    temporal_smooth_final = float(best_params["temporal_smooth_weight"]) if best_params and "temporal_smooth_weight" in best_params else config.TEMPORAL_SMOOTHING_WEIGHT
    epochs_final = config.EPOCHS

    cfg = TFTCfg(
        d_model=hidden_size_final,
        d_hidden=max(2 * hidden_size_final, 64),
        nhead=nhead_final,
        dropout=dropout_final,
        lstm_layers=lstm_layers_final,
        attn_dropout=attn_dropout_final,
        ff_dropout=ff_dropout_final,
        horizon=config.HORIZON,
        quantiles=config.QUANTILES,
        device=device_name,
    )

    tr_seq = make_sequences(parts['train'], config.SEQ_LEN, config.HORIZON, target_type, anchor_lag=anchor_lag_final)
    va_seq = make_sequences(parts['val'],   config.SEQ_LEN, config.HORIZON, target_type, anchor_lag=anchor_lag_final)
    te_seq = make_sequences(parts['test'],  config.SEQ_LEN, config.HORIZON, target_type, anchor_lag=anchor_lag_final)

    val_window_len = config.CV_VAL_SAMPLES if config.CV_VAL_SAMPLES > 0 else len(va_seq.y)

    cv_summary = None
    if config.CV_ENABLED and cv_df is not None and config.CV_FOLDS >= 2:
        cv_seq = make_sequences(cv_df, config.SEQ_LEN, config.HORIZON, target_type, anchor_lag=anchor_lag_final)
        cv_params = {
            "epochs": min(epochs_final, config.CV_EPOCHS),
            "lr": lr_final,
            "batch_size": batch_size_final,
            "mse_aux_weight": mse_aux_final,
            "grad_clip": grad_clip_final,
            "weight_decay": weight_decay_final,
            "cv_folds": config.CV_FOLDS,
            "min_train": config.CV_MIN_TRAIN_SAMPLES,
            "temporal_smooth_weight": temporal_smooth_final,
        }
        cv_summary = run_time_series_cv(cfg, cv_seq, step_w, torch_device, cv_params, target_type, config.QUANTILES, val_window_len, verbose=True)

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
        epochs=epochs_final,
        lr=lr_final,
        batch_size=batch_size_final,
        device=torch_device,
        step_weights_np=step_w,
        mse_aux_weight=mse_aux_final,
        grad_clip_norm=grad_clip_final,
        weight_decay=weight_decay_final,
        temporal_smooth_weight=temporal_smooth_final,
    )

    q_va_z, det_va = predict(model, Xo_va, Xk_va, Xs_va, device=torch_device)
    q_te_z, det_te = predict(model, Xo_te, Xk_te, Xs_te, device=torch_device)
    q_va = _denormalize_targets(q_va_z, stats)
    q_te = _denormalize_targets(q_te_z, stats)
    _ensure_finite("val_quantiles", q_va)
    _ensure_finite("test_quantiles", q_te)

    val_diag = _compute_diagnostics(va_seq.y, q_va, va_seq.c0, target_type, config.QUANTILES, dates=va_seq.dates_fut)
    test_diag = _compute_diagnostics(te_seq.y, q_te, te_seq.c0, target_type, config.QUANTILES, dates=te_seq.dates_fut)

    val_tft_summary = _summarize_tft_details(det_va, va_seq.feat_names)
    test_tft_summary = _summarize_tft_details(det_te, te_seq.feat_names)

    metrics_flat: Dict[str, object] = {}
    for split_name, diag in ("val", val_diag), ("test", test_diag):
        for key, value in diag["metrics"].items():
            metrics_flat[f"{split_name}_{key}"] = float(value)

    segments_payload: Dict[str, Dict[str, object]] = {}
    if val_diag.get("segments"):
        segments_payload["val"] = val_diag["segments"]
    if test_diag.get("segments"):
        segments_payload["test"] = test_diag["segments"]
    if segments_payload:
        metrics_flat["segments"] = segments_payload

    run_dir = create_next_run_dir(config.OUT_DIR)
    (run_dir / "pred").mkdir(parents=True, exist_ok=True)
    (run_dir / "figs").mkdir(parents=True, exist_ok=True)

    np.save(run_dir / "pred/val_quantiles.npy", q_va)
    np.save(run_dir / "pred/test_quantiles.npy", q_te)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_flat, f, ensure_ascii=False, indent=2)

    insights_base = run_dir / "pred"
    if val_tft_summary:
        with (insights_base / "val_tft_summary.json").open("w", encoding="utf-8") as f:
            json.dump(val_tft_summary, f, ensure_ascii=False, indent=2)
        obs = val_tft_summary.get("observed")
        if obs:
            save_variable_weights(insights_base, "val_observed", obs)
            plot_variable_bar(obs["feature_names"], np.asarray(obs["mean"], dtype=float),
                              "Observed Variable Importance (VAL)",
                              run_dir / "figs/val_var_obs_top20.png")
            plot_time_heatmap(np.asarray(obs["time_mean"], dtype=float),
                              "Observed Variable Weights Over Time (VAL)",
                              run_dir / "figs/val_var_obs_time.png",
                              ylabel="Feature")
        kn = val_tft_summary.get("known")
        if kn:
            save_variable_weights(insights_base, "val_known", kn)
            plot_variable_bar(kn["feature_names"], np.asarray(kn["mean"], dtype=float),
                              "Known Variable Importance (VAL)",
                              run_dir / "figs/val_var_known_top20.png")
            plot_time_heatmap(np.asarray(kn["time_mean"], dtype=float),
                              "Known Variable Weights Over Time (VAL)",
                              run_dir / "figs/val_var_known_time.png",
                              ylabel="Feature")
        st = val_tft_summary.get("static")
        if st:
            save_variable_weights(insights_base, "val_static", st)
            plot_variable_bar(st["feature_names"], np.asarray(st["mean"], dtype=float),
                              "Static Variable Importance (VAL)",
                              run_dir / "figs/val_var_static.png", top_k=len(st["feature_names"]))
        attn = val_tft_summary.get("attention")
        if attn:
            save_attn_weights(insights_base, "val", attn)
            plot_attention_heatmap(np.asarray(attn["mean"], dtype=float),
                                   "Temporal Attention (VAL)",
                                   run_dir / "figs/val_attention.png")

    if test_tft_summary:
        with (insights_base / "test_tft_summary.json").open("w", encoding="utf-8") as f:
            json.dump(test_tft_summary, f, ensure_ascii=False, indent=2)
        obs = test_tft_summary.get("observed")
        if obs:
            save_variable_weights(insights_base, "test_observed", obs)
            plot_variable_bar(obs["feature_names"], np.asarray(obs["mean"], dtype=float),
                              "Observed Variable Importance (TEST)",
                              run_dir / "figs/test_var_obs_top20.png")
            plot_time_heatmap(np.asarray(obs["time_mean"], dtype=float),
                              "Observed Variable Weights Over Time (TEST)",
                              run_dir / "figs/test_var_obs_time.png",
                              ylabel="Feature")
        kn = test_tft_summary.get("known")
        if kn:
            save_variable_weights(insights_base, "test_known", kn)
            plot_variable_bar(kn["feature_names"], np.asarray(kn["mean"], dtype=float),
                              "Known Variable Importance (TEST)",
                              run_dir / "figs/test_var_known_top20.png")
            plot_time_heatmap(np.asarray(kn["time_mean"], dtype=float),
                              "Known Variable Weights Over Time (TEST)",
                              run_dir / "figs/test_var_known_time.png",
                              ylabel="Feature")
        st = test_tft_summary.get("static")
        if st:
            save_variable_weights(insights_base, "test_static", st)
            plot_variable_bar(st["feature_names"], np.asarray(st["mean"], dtype=float),
                              "Static Variable Importance (TEST)",
                              run_dir / "figs/test_var_static.png", top_k=len(st["feature_names"]))
        attn = test_tft_summary.get("attention")
        if attn:
            save_attn_weights(insights_base, "test", attn)
            plot_attention_heatmap(np.asarray(attn["mean"], dtype=float),
                                   "Temporal Attention (TEST)",
                                   run_dir / "figs/test_attention.png")

    if cv_summary:
        with (run_dir / "cv_summary.json").open("w", encoding="utf-8") as f:
            json.dump(cv_summary, f, ensure_ascii=False, indent=2)
    if optuna_summary:
        with (run_dir / "optuna_summary.json").open("w", encoding="utf-8") as f:
            json.dump(optuna_summary, f, ensure_ascii=False, indent=2)

    # 绘图：step=1 的价格时序对比 + 分位带（验证/测试）
    price_true_va_steps = val_diag["price_true_steps"]
    price_pred_va_steps = val_diag["price_pred_steps"]
    price_true_te_steps = test_diag["price_true_steps"]
    price_pred_te_steps = test_diag["price_pred_steps"]

    plot_price_series(va_seq.dates_fut, price_true_va_steps[:, 0], price_pred_va_steps[:, 0], "VAL Price h=1 (q50)", run_dir / "figs/val_price_h1.png")
    plot_price_series(te_seq.dates_fut, price_true_te_steps[:, 0], price_pred_te_steps[:, 0], "TEST Price h=1 (q50)", run_dir / "figs/test_price_h1.png")

    q_index = config.QUANTILES.index(0.5) if 0.5 in config.QUANTILES else len(config.QUANTILES) // 2
    ql_va, qm_va, qh_va = q_va[:, 0, 0], q_va[:, 0, q_index], q_va[:, 0, -1]
    ql_te, qm_te, qh_te = q_te[:, 0, 0], q_te[:, 0, q_index], q_te[:, 0, -1]
    ql_p = targets_to_price(va_seq.c0, ql_va, target_type)
    qm_p = targets_to_price(va_seq.c0, qm_va, target_type)
    qh_p = targets_to_price(va_seq.c0, qh_va, target_type)
    plot_quantile_band(va_seq.dates_fut, ql_p[:, 0], qm_p[:, 0], qh_p[:, 0], "VAL Quantile band h=1", run_dir / "figs/val_band_h1.png")

    ql_t = targets_to_price(te_seq.c0, ql_te, target_type)
    qm_t = targets_to_price(te_seq.c0, qm_te, target_type)
    qh_t = targets_to_price(te_seq.c0, qh_te, target_type)
    plot_quantile_band(te_seq.dates_fut, ql_t[:, 0], qm_t[:, 0], qh_t[:, 0], "TEST Quantile band h=1", run_dir / "figs/test_band_h1.png")

    h_last = config.HORIZON - 1
    plot_step_series(va_seq.dates_fut, price_true_va_steps[:, h_last], price_pred_va_steps[:, h_last], h_last + 1, "VAL Price", run_dir / "figs/val_price_hLast.png")

    plot_step_series(te_seq.dates_fut, price_true_te_steps[:, h_last], price_pred_te_steps[:, h_last], h_last + 1, "TEST Price", run_dir / "figs/test_price_hLast.png")

    np.save(run_dir / "pred/val_rmse_steps.npy", np.array(val_diag["rmse_steps_price"]))
    np.save(run_dir / "pred/test_rmse_steps.npy", np.array(test_diag["rmse_steps_price"]))
    np.save(run_dir / "pred/val_cov_steps.npy", np.array(val_diag["coverage_steps"]))
    np.save(run_dir / "pred/test_cov_steps.npy", np.array(test_diag["coverage_steps"]))
    plot_step_metric(np.array(val_diag["rmse_steps_price"]), "VAL RMSE per step (price)", run_dir / "figs/val_rmse_steps.png")
    plot_step_metric(np.array(test_diag["rmse_steps_price"]), "TEST RMSE per step (price)", run_dir / "figs/test_rmse_steps.png")
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
        true_price_cal = targets_to_price(te_seq.c0, te_seq.y, target_type)
        pred_price_cal = targets_to_price(te_seq.c0, yhat_te_med, target_type)
        test_rmse_steps_cal = np.sqrt(np.mean((true_price_cal - pred_price_cal) ** 2, axis=0))
        np.save(run_dir / "pred/test_rmse_steps_calibrated.npy", test_rmse_steps_cal)
        plot_step_metric(test_rmse_steps_cal, "TEST RMSE per step (calibrated)", run_dir / "figs/test_rmse_steps_calibrated.png")
        with (run_dir / "calibration_q50.json").open("w", encoding="utf-8") as f:
            json.dump({"a": a_coef, "b": b_coef}, f, ensure_ascii=False, indent=2)

    print("ver2 运行完成。结果目录:", str(run_dir))


if __name__ == "__main__":
    main()

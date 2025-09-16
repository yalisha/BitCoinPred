#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import json
import numpy as np
import pandas as pd

from . import config
from .utils import create_next_run_dir
from .data import load_raw, build_feature_frame, split_by_date, make_sequences, SeqData
from .model_tft import TFTCfg, TFTMultiHQuantile, train_one, predict
from .plots import plot_price_series, plot_step_series, plot_quantile_band, plot_step_metric, plot_bar


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


def main():
    raw = load_raw(config.DATA_CSV)
    feat = build_feature_frame(raw, target_type=config.PRED_TARGET)
    parts = split_by_date(feat, config.TRAIN_END, config.VAL_END)

    tr_seq = make_sequences(parts['train'], config.SEQ_LEN, config.HORIZON, config.PRED_TARGET)
    va_seq = make_sequences(parts['val'],   config.SEQ_LEN, config.HORIZON, config.PRED_TARGET)
    te_seq = make_sequences(parts['test'],  config.SEQ_LEN, config.HORIZON, config.PRED_TARGET)

    # 归一化（仅基于训练集）
    def fit_std(X):
        mu = X.mean(axis=(0,1), keepdims=True)
        sd = X.std(axis=(0,1), keepdims=True); sd[sd<1e-8] = 1.0
        return mu, sd
    def apply_std(X, mu, sd):
        return (X - mu)/sd

    mu_o, sd_o = fit_std(tr_seq.X_obs)
    mu_k, sd_k = fit_std(tr_seq.X_known)
    mu_s = tr_seq.X_static.mean(axis=0, keepdims=True); sd_s = tr_seq.X_static.std(axis=0, keepdims=True); sd_s[sd_s<1e-8]=1.0

    Xo_tr = apply_std(tr_seq.X_obs, mu_o, sd_o)
    Xk_tr = apply_std(tr_seq.X_known, mu_k, sd_k)
    Xs_tr = (tr_seq.X_static - mu_s)/sd_s

    Xo_va = apply_std(va_seq.X_obs, mu_o, sd_o)
    Xk_va = apply_std(va_seq.X_known, mu_k, sd_k)
    Xs_va = (va_seq.X_static - mu_s)/sd_s

    Xo_te = apply_std(te_seq.X_obs, mu_o, sd_o)
    Xk_te = apply_std(te_seq.X_known, mu_k, sd_k)
    Xs_te = (te_seq.X_static - mu_s)/sd_s

    # 目标标准化（按步长逐列标准化，提升数值稳定性）
    y_mu = tr_seq.y.mean(axis=0, keepdims=True)
    y_sd = tr_seq.y.std(axis=0, keepdims=True)
    y_sd[y_sd < 1e-8] = 1.0
    y_tr = (tr_seq.y - y_mu) / y_sd
    y_va = (va_seq.y - y_mu) / y_sd
    y_te = (te_seq.y - y_mu) / y_sd

    # 模型
    cfg = TFTCfg(d_model=config.HIDDEN_SIZE, d_hidden=max(2*config.HIDDEN_SIZE,64), nhead=min(config.NHEAD, max(1, config.HIDDEN_SIZE//8)),
                 dropout=config.DROPOUT, horizon=config.HORIZON, quantiles=config.QUANTILES, device='cpu')
    model = TFTMultiHQuantile(num_obs=Xo_tr.shape[-1], num_kn=Xk_tr.shape[-1], num_static=Xs_tr.shape[-1], cfg=cfg)

    # 步长加权
    if config.STEP_WEIGHT_MODE == 'linear':
        step_w = np.arange(1, config.HORIZON+1, dtype=float)
        step_w = 1.0 + config.STEP_WEIGHT_ALPHA * step_w
    elif config.STEP_WEIGHT_MODE == 'square':
        step_w = (np.arange(1, config.HORIZON+1, dtype=float) / config.HORIZON) ** 2
        step_w = 1.0 + step_w
    else:
        step_w = None

    model = train_one(cfg, model,
                      (Xo_tr, Xk_tr, Xs_tr, y_tr),
                      (Xo_va, Xk_va, Xs_va, y_va),
                      epochs=config.EPOCHS, lr=config.LR, batch_size=config.BATCH_SIZE, device='cpu',
                      step_weights_np=step_w, mse_aux_weight=config.MSE_AUX_WEIGHT)

    # 预测
    q_va_z, det_va = predict(model, Xo_va, Xk_va, Xs_va)
    q_te_z, det_te = predict(model, Xo_te, Xk_te, Xs_te)
    # 反标准化回原始目标空间（按步长逐列，匹配 [N,H,Q]）
    y_mu3 = y_mu.reshape(1, y_mu.shape[1], 1)
    y_sd3 = y_sd.reshape(1, y_sd.shape[1], 1)
    q_va = q_va_z * y_sd3 + y_mu3
    q_te = q_te_z * y_sd3 + y_mu3
    q_va = np.nan_to_num(q_va, nan=0.0, posinf=0.0, neginf=0.0)
    q_te = np.nan_to_num(q_te, nan=0.0, posinf=0.0, neginf=0.0)
    # 取中位数
    q_index = config.QUANTILES.index(0.5) if 0.5 in config.QUANTILES else len(config.QUANTILES)//2
    yhat_va = q_va[..., q_index]
    yhat_te = q_te[..., q_index]

    # 从目标还原到价格（逐步：对每个样本的每个步长用 c0 映射）
    # 这里为了与一对一日期对齐，先用 step=1 的价格时序对比；同时保存全部步长的预测矩阵
    price_true_va_1 = to_price_from_target(va_seq.c0, va_seq.y[:,0], config.PRED_TARGET)
    price_pred_va_1 = to_price_from_target(va_seq.c0, yhat_va[:,0], config.PRED_TARGET)
    price_true_te_1 = to_price_from_target(te_seq.c0, te_seq.y[:,0], config.PRED_TARGET)
    price_pred_te_1 = to_price_from_target(te_seq.c0, yhat_te[:,0], config.PRED_TARGET)

    # 评估：以 step=1 的中位数预测为主，也计算各步 RMSE/MAE 与覆盖率
    def rmse(a,b):
        a=np.asarray(a); b=np.asarray(b); return float(np.sqrt(np.mean((a-b)**2)))
    def mae(a,b):
        a=np.asarray(a); b=np.asarray(b); return float(np.mean(np.abs(a-b)))
    def coverage(y_true, ql, qh):
        yt = np.asarray(y_true)
        return float(np.mean((yt >= ql) & (yt <= qh)))

    # 每步 RMSE/MAE（价格空间）
    val_rmse_steps = []
    val_mae_steps = []
    test_rmse_steps = []
    test_mae_steps = []
    # 覆盖率（目标空间）
    # 获取对应分位索引
    ql_idx = 0
    qh_idx = -1
    val_cov_steps = []
    test_cov_steps = []
    for h in range(config.HORIZON):
        val_rmse_steps.append(rmse(to_price_from_target(va_seq.c0, va_seq.y[:,h], config.PRED_TARGET),
                                   to_price_from_target(va_seq.c0, yhat_va[:,h], config.PRED_TARGET)))
        val_mae_steps.append(mae(to_price_from_target(va_seq.c0, va_seq.y[:,h], config.PRED_TARGET),
                                 to_price_from_target(va_seq.c0, yhat_va[:,h], config.PRED_TARGET)))
        test_rmse_steps.append(rmse(to_price_from_target(te_seq.c0, te_seq.y[:,h], config.PRED_TARGET),
                                    to_price_from_target(te_seq.c0, yhat_te[:,h], config.PRED_TARGET)))
        test_mae_steps.append(mae(to_price_from_target(te_seq.c0, te_seq.y[:,h], config.PRED_TARGET),
                                  to_price_from_target(te_seq.c0, yhat_te[:,h], config.PRED_TARGET)))
        # 覆盖率（目标空间）
        val_cov_steps.append(coverage(va_seq.y[:,h], q_va[:,h,ql_idx], q_va[:,h,qh_idx]))
        test_cov_steps.append(coverage(te_seq.y[:,h], q_te[:,h,ql_idx], q_te[:,h,qh_idx]))

    metrics = {
        "val_rmse_h1_price": rmse(price_true_va_1, price_pred_va_1),
        "val_mae_h1_price": mae(price_true_va_1, price_pred_va_1),
        "test_rmse_h1_price": rmse(price_true_te_1, price_pred_te_1),
        "test_mae_h1_price": mae(price_true_te_1, price_pred_te_1),
        "val_rmse_avg_steps": float(np.mean(val_rmse_steps)),
        "test_rmse_avg_steps": float(np.mean(test_rmse_steps)),
    }

    # 输出
    run_dir = create_next_run_dir(config.OUT_DIR)
    (run_dir / "pred").mkdir(parents=True, exist_ok=True)
    (run_dir / "figs").mkdir(parents=True, exist_ok=True)

    # 保存分位预测矩阵（验证/测试） [N, H, Q]
    np.save(run_dir / "pred/val_quantiles.npy", q_va)
    np.save(run_dir / "pred/test_quantiles.npy", q_te)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 绘图：step=1 的价格时序对比 + 分位带（验证/测试）
    plot_price_series(va_seq.dates_fut, price_true_va_1, price_pred_va_1, f"VAL Price h=1 (q50)", run_dir / "figs/val_price_h1.png")
    plot_price_series(te_seq.dates_fut, price_true_te_1, price_pred_te_1, f"TEST Price h=1 (q50)", run_dir / "figs/test_price_h1.png")

    # 分位带（以 step=1为例）
    ql, qm, qh = q_va[:,0,0], q_va[:,0,q_index], q_va[:,0,-1]
    qt_l, qt_m, qt_h = q_te[:,0,0], q_te[:,0,q_index], q_te[:,0,-1]
    # 转为价格空间
    ql_p = to_price_from_target(va_seq.c0, ql, config.PRED_TARGET)
    qm_p = to_price_from_target(va_seq.c0, qm, config.PRED_TARGET)
    qh_p = to_price_from_target(va_seq.c0, qh, config.PRED_TARGET)
    plot_quantile_band(va_seq.dates_fut, ql_p, qm_p, qh_p, "VAL Quantile band h=1", run_dir / "figs/val_band_h1.png")

    ql_t = to_price_from_target(te_seq.c0, qt_l, config.PRED_TARGET)
    qm_t = to_price_from_target(te_seq.c0, qt_m, config.PRED_TARGET)
    qh_t = to_price_from_target(te_seq.c0, qt_h, config.PRED_TARGET)
    plot_quantile_band(te_seq.dates_fut, ql_t, qm_t, qh_t, "TEST Quantile band h=1", run_dir / "figs/test_band_h1.png")

    # 各步步长的中位数预测时序图（可选：只画h=1与h=H）
    h_last = config.HORIZON-1
    price_true_va_last = to_price_from_target(va_seq.c0, va_seq.y[:,h_last], config.PRED_TARGET)
    price_pred_va_last = to_price_from_target(va_seq.c0, yhat_va[:,h_last], config.PRED_TARGET)
    plot_step_series(va_seq.dates_fut, price_true_va_last, price_pred_va_last, h_last+1, "VAL Price", run_dir / "figs/val_price_hLast.png")

    price_true_te_last = to_price_from_target(te_seq.c0, te_seq.y[:,h_last], config.PRED_TARGET)
    price_pred_te_last = to_price_from_target(te_seq.c0, yhat_te[:,h_last], config.PRED_TARGET)
    plot_step_series(te_seq.dates_fut, price_true_te_last, price_pred_te_last, h_last+1, "TEST Price", run_dir / "figs/test_price_hLast.png")

    # 每步误差曲线与覆盖率柱状图
    np.save(run_dir / "pred/val_rmse_steps.npy", np.array(val_rmse_steps))
    np.save(run_dir / "pred/test_rmse_steps.npy", np.array(test_rmse_steps))
    np.save(run_dir / "pred/val_cov_steps.npy", np.array(val_cov_steps))
    np.save(run_dir / "pred/test_cov_steps.npy", np.array(test_cov_steps))
    plot_step_metric(np.array(val_rmse_steps), "VAL RMSE per step (price)", run_dir / "figs/val_rmse_steps.png")
    plot_step_metric(np.array(test_rmse_steps), "TEST RMSE per step (price)", run_dir / "figs/test_rmse_steps.png")
    plot_bar(np.array(val_cov_steps), "VAL Coverage q10–q90 (target)", run_dir / "figs/val_coverage.png", ylabel='coverage')
    plot_bar(np.array(test_cov_steps), "TEST Coverage q10–q90 (target)", run_dir / "figs/test_coverage.png", ylabel='coverage')

    # q50 校准（基于验证集），并应用于测试集
    if config.CALIBRATE_Q50:
        a = []
        b = []
        yhat_va_med = yhat_va.copy()
        yhat_te_med = yhat_te.copy()
        for h in range(config.HORIZON):
            Y = va_seq.y[:,h]
            X = yhat_va_med[:,h]
            # 最小二乘拟合 Y ≈ a*X + b
            A = np.vstack([X, np.ones_like(X)]).T
            sol, *_ = np.linalg.lstsq(A, Y, rcond=None)
            a.append(float(sol[0])); b.append(float(sol[1]))
            # 应用到测试
            yhat_te_med[:,h] = a[-1]*yhat_te_med[:,h] + b[-1]
        # 校准后的价格指标
        test_rmse_steps_cal = []
        for h in range(config.HORIZON):
            test_rmse_steps_cal.append(rmse(to_price_from_target(te_seq.c0, te_seq.y[:,h], config.PRED_TARGET),
                                            to_price_from_target(te_seq.c0, yhat_te_med[:,h], config.PRED_TARGET)))
        np.save(run_dir / "pred/test_rmse_steps_calibrated.npy", np.array(test_rmse_steps_cal))
        plot_step_metric(np.array(test_rmse_steps_cal), "TEST RMSE per step (calibrated)", run_dir / "figs/test_rmse_steps_calibrated.png")
        with (run_dir / "calibration_q50.json").open("w", encoding="utf-8") as f:
            json.dump({"a": a, "b": b}, f, ensure_ascii=False, indent=2)

    print("ver2 运行完成。结果目录:", str(run_dir))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

from . import config
from .data.loaders import load_macro_btc, split_by_date
from .features.engineering import make_features
from .models.linear import OLSRegressor, RidgeRegressor
from .models.svm import SVRRegressor
from .models.xgb import XGBModel
from .models.lgbm import LGBMModel
from .models.arima import fit_predict_arima
from .models.varx import fit_predict_varx
from .models.garch import fit_predict_garch
from .utils.scaler import Standardizer
from .eval.metrics import rmse, mae, mape, r2, directional_accuracy
from .features.sequence import build_sequences
from .models.nn import NNConfig, fit_predict_seq
from .models.tft import TFTConfig, TFTModel
from .utils.run_dir import create_next_run_dir
from .report import plots as rpt
import json


@dataclass
class TrainConfig:
    data_path: Path = config.DEFAULT_DATA
    horizon: int = config.HORIZON
    train_end: str = config.TRAIN_END
    val_end: str = config.VAL_END
    model: str = "ridge"  # ridge | ols | svr | xgb | lgbm | arima | varx | garch | lstm | gru | cnn | tft
    ridge_alpha: float = 1.0
    out_dir: Path = config.OUT_DIR
    # 序列模型参数
    seq_len: int = 30
    epochs: int = 20
    batch_size: int = 64
    hidden_size: int = 64
    lr: float = 1e-3
    target_type: str = "logret"  # logret | pctret | price


def _pick_model(name: str, ridge_alpha: float):
    name = name.lower()
    if name == "ols":
        return OLSRegressor()
    elif name in ("ridge", "auto"):
        return RidgeRegressor(alpha=ridge_alpha)
    elif name == "svr":
        return SVRRegressor(C=2.0, epsilon=0.01, kernel="rbf")
    elif name == "xgb":
        return XGBModel()
    elif name == "lgbm":
        return LGBMModel()
    else:
        raise ValueError(f"未知模型: {name}")


def fit_and_eval(cfg: TrainConfig, run_dir: Optional[Path] = None) -> Dict[str, float]:
    # 加载与造特征
    raw = load_macro_btc(cfg.data_path)
    X_all, y_all, dates_t, dates_fut, c0_all = make_features(raw, h=cfg.horizon, target_type=cfg.target_type)

    # 合并回含日期
    feat = X_all.copy()
    feat["target"] = y_all
    feat["date"] = dates_t
    feat["date_future"] = dates_fut
    feat["c0"] = c0_all

    # 按日期切分
    parts = split_by_date(feat, cfg.train_end, cfg.val_end)
    feat_cols = parts["train"].drop(columns=["target", "date"]).columns.tolist()
    def XY(df: pd.DataFrame):
        X = df.drop(columns=["target", "date", "date_future", "c0"]).to_numpy(dtype=float)
        y = df["target"].to_numpy(dtype=float)
        d = df["date"].copy()
        dfut = df["date_future"].copy()
        c0 = df["c0"].to_numpy(dtype=float)
        return X, y, d, dfut, c0

    Xtr, ytr, dtr, dtr_fut, c0_tr = XY(parts["train"])  # noqa: F841
    Xva, yva, dva, dva_fut, c0_va = XY(parts["val"])    # noqa: F841
    Xte, yte, dte, dte_fut, c0_te = XY(parts["test"])   # noqa: F841

    model_name = cfg.model.lower()
    if model_name in {"ridge", "ols", "svr", "xgb", "lgbm"}:
        # 标准化（仅基于训练集）
        scaler = Standardizer()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)
        Xte_s = scaler.transform(Xte)

        if model_name in {"xgb", "lgbm"}:
            model = _pick_model(cfg.model, cfg.ridge_alpha)
            model.fit(Xtr_s, ytr, Xva_s, yva)
        else:
            model = _pick_model(cfg.model, cfg.ridge_alpha)
            model.fit(Xtr_s, ytr)

        pred_va = model.predict(Xva_s)
        pred_te = model.predict(Xte_s)

    elif model_name == "arima":
        # 使用 ARIMAX：exog = 标准化后的 X
        scaler = Standardizer()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)
        Xte_s = scaler.transform(Xte)
        pred_va, pred_te = fit_predict_arima(ytr, Xtr_s, yva, Xva_s, yte, Xte_s, order=(1,0,1))

    elif model_name == "varx":
        scaler = Standardizer()
        Xtr_s = scaler.fit_transform(Xtr)
        Xva_s = scaler.transform(Xva)
        Xte_s = scaler.transform(Xte)
        pred_va, pred_te = fit_predict_varx(ytr.reshape(-1,1), Xtr_s, yva.reshape(-1,1), Xva_s, yte.reshape(-1,1), Xte_s, order=(1,0))

    elif model_name == "garch":
        try:
            scaler = Standardizer()
            Xtr_s = scaler.fit_transform(Xtr)
            Xva_s = scaler.transform(Xva)
            Xte_s = scaler.transform(Xte)
            pred_va, pred_te = fit_predict_garch(ytr, Xtr_s, yva, Xva_s, yte, Xte_s, mean_lags=1)
        except Exception:
            # 回退到 ARIMA(1,0,1)
            scaler = Standardizer()
            Xtr_s = scaler.fit_transform(Xtr)
            Xva_s = scaler.transform(Xva)
            Xte_s = scaler.transform(Xte)
            pred_va, pred_te = fit_predict_arima(ytr, Xtr_s, yva, Xva_s, yte, Xte_s, order=(1,0,1))

    elif model_name in {"lstm", "gru", "cnn", "tft"}:
        # 针对每个分片单独构造序列，避免跨越泄漏
        tr_seq_X, tr_seq_y, _ , _ , _ = build_sequences(parts["train"], cfg.seq_len)
        va_seq_X, va_seq_y, dva, dva_fut_seq, c0_va_seq = build_sequences(parts["val"], cfg.seq_len)
        te_seq_X, te_seq_y, dte, dte_fut_seq, c0_te_seq = build_sequences(parts["test"], cfg.seq_len)

        # 替换 dates 与 y 为序列对齐版本
        yva, yte = va_seq_y, te_seq_y

        # 标准化（基于训练序列的所有时间步）
        scaler = Standardizer()
        F = tr_seq_X.shape[2]
        def fit_transform_seq(X: np.ndarray) -> np.ndarray:
            n, t, f = X.shape
            Z = X.reshape(n*t, f)
            if not hasattr(scaler, 'mean_') or scaler.mean_ is None:
                Zs = scaler.fit_transform(Z)
            else:
                Zs = scaler.transform(Z)
            return Zs.reshape(n, t, f)

        tr_seq_Xs = fit_transform_seq(tr_seq_X)
        va_seq_Xs = fit_transform_seq(va_seq_X)
        te_seq_Xs = fit_transform_seq(te_seq_X)

        if model_name == "tft":
            tft_cfg = TFTConfig(num_vars=F, d_model=cfg.hidden_size, d_hidden=max(2*cfg.hidden_size, 64),
                                nhead=min(4, cfg.hidden_size//8 if cfg.hidden_size>=8 else 1),
                                dropout=0.1, epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr, device="cpu")
            tft = TFTModel(tft_cfg)
            out = tft.fit_predict(tr_seq_Xs, tr_seq_y, va_seq_Xs, va_seq_y, te_seq_Xs, return_details=True)
            pred_va, pred_te, tft_details = out
        else:
            nn_cfg = NNConfig(model=model_name, input_size=F, hidden_size=cfg.hidden_size,
                              num_layers=2, dropout=0.1, d_model=cfg.hidden_size, nhead=4,
                              epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr,
                              device="cpu")
            pred_va, pred_te = fit_predict_seq(nn_cfg, tr_seq_Xs, tr_seq_y, va_seq_Xs, va_seq_y, te_seq_Xs)

        # dates 使用序列末端时间步对应的 dva/dte（已由 build_sequences 返回）
        dva, dte = dva, dte

    else:
        raise ValueError(f"未知模型: {cfg.model}")

    metrics = {
        "val_rmse": rmse(yva, pred_va),
        "val_mae": mae(yva, pred_va),
        "val_mape": mape(yva, pred_va),
        "val_r2": r2(yva, pred_va),
        "val_dir": directional_accuracy(yva, pred_va),
        "test_rmse": rmse(yte, pred_te),
        "test_mae": mae(yte, pred_te),
        "test_mape": mape(yte, pred_te),
        "test_r2": r2(yte, pred_te),
        "test_dir": directional_accuracy(yte, pred_te),
    }
    # 合并 VAL+TEST 作为整体指标
    y_all = np.concatenate([yva, yte])
    yhat_all = np.concatenate([pred_va, pred_te])
    metrics.update({
        "all_rmse": rmse(y_all, yhat_all),
        "all_mae": mae(y_all, yhat_all),
        "all_mape": mape(y_all, yhat_all),
        "all_r2": r2(y_all, yhat_all),
        "all_dir": directional_accuracy(y_all, yhat_all),
    })

    # 创建本次运行目录
    if run_dir is None:
        run_dir = create_next_run_dir(cfg.out_dir)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
    # 输出预测文件
    out_df = pd.DataFrame({
        "date": np.concatenate([dva.to_numpy(), dte.to_numpy()]),
        "split": ["val"] * len(dva) + ["test"] * len(dte),
        "y_true": np.concatenate([yva, yte]),
        "y_pred": np.concatenate([pred_va, pred_te]),
    })
    fname = run_dir / f"pred_{cfg.model}_h{cfg.horizon}.csv"
    out_df.to_csv(fname, index=False)

    # 价格层面对比（序列模型因滑窗长度不同，默认跳过）
    if model_name not in {"lstm", "gru", "cnn", "tft"}:
        def to_price(c0: np.ndarray, y: np.ndarray) -> np.ndarray:
            t = cfg.target_type.lower()
            if t == "logret":
                return c0 * np.exp(y)
            if t == "pctret":
                return c0 * (1.0 + y)
            return y  # price

        price_true_va = to_price(c0_va, yva)
        price_pred_va = to_price(c0_va, pred_va)
        price_true_te = to_price(c0_te, yte)
        price_pred_te = to_price(c0_te, pred_te)

        price_df = pd.DataFrame({
            "date": np.concatenate([dva_fut.to_numpy(), dte_fut.to_numpy()]),
            "split": ["val"] * len(dva_fut) + ["test"] * len(dte_fut),
            "price_true": np.concatenate([price_true_va, price_true_te]),
            "price_pred": np.concatenate([price_pred_va, price_pred_te]),
        })
        price_df.to_csv(run_dir / f"price_{cfg.model}_h{cfg.horizon}.csv", index=False)

    # 保存指标
    with (run_dir / f"metrics_{cfg.model}_h{cfg.horizon}.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 可视化：时间序列、散点、残差、直方图
    # 验证
    rpt.plot_timeseries(dva, yva, pred_va, f"VAL {cfg.model} h={cfg.horizon}", run_dir / "val_timeseries.png")
    rpt.plot_scatter(yva, pred_va, f"VAL Scatter {cfg.model}", run_dir / "val_scatter.png")
    rpt.plot_residuals(dva, yva - pred_va, f"VAL Residuals {cfg.model}", run_dir / "val_residuals.png")
    rpt.plot_hist(yva - pred_va, f"VAL Residuals Hist {cfg.model}", run_dir / "val_residuals_hist.png")
    # 测试
    rpt.plot_timeseries(dte, yte, pred_te, f"TEST {cfg.model} h={cfg.horizon}", run_dir / "test_timeseries.png")
    rpt.plot_scatter(yte, pred_te, f"TEST Scatter {cfg.model}", run_dir / "test_scatter.png")
    rpt.plot_residuals(dte, yte - pred_te, f"TEST Residuals {cfg.model}", run_dir / "test_residuals.png")
    rpt.plot_hist(yte - pred_te, f"TEST Residuals Hist {cfg.model}", run_dir / "test_residuals_hist.png")

    # 价格层面的时间序列对比
    if model_name not in {"lstm", "gru", "cnn", "tft"}:
        try:
            rpt.plot_timeseries(dva_fut, price_true_va, price_pred_va, f"VAL Price {cfg.model}", run_dir / "val_timeseries_price.png")
            rpt.plot_timeseries(dte_fut, price_true_te, price_pred_te, f"TEST Price {cfg.model}", run_dir / "test_timeseries_price.png")
        except Exception:
            pass
    else:
        # 对序列模型还原价格并绘图（使用序列末端的 c0 与未来日期）
        try:
            def to_price(c0: np.ndarray, y: np.ndarray) -> np.ndarray:
                t = cfg.target_type.lower()
                if t == "logret":
                    return c0 * np.exp(y)
                if t == "pctret":
                    return c0 * (1.0 + y)
                return y
            price_true_va_s = to_price(c0_va_seq, yva)
            price_pred_va_s = to_price(c0_va_seq, pred_va)
            price_true_te_s = to_price(c0_te_seq, yte)
            price_pred_te_s = to_price(c0_te_seq, pred_te)

            price_df = pd.DataFrame({
                "date": np.concatenate([dva_fut_seq.to_numpy(), dte_fut_seq.to_numpy()]),
                "split": ["val"] * len(dva_fut_seq) + ["test"] * len(dte_fut_seq),
                "price_true": np.concatenate([price_true_va_s, price_true_te_s]),
                "price_pred": np.concatenate([price_pred_va_s, price_pred_te_s]),
            })
            price_df.to_csv(run_dir / f"price_{cfg.model}_h{cfg.horizon}.csv", index=False)

            rpt.plot_timeseries(dva_fut_seq, price_true_va_s, price_pred_va_s, f"VAL Price {cfg.model}", run_dir / "val_timeseries_price.png")
            rpt.plot_timeseries(dte_fut_seq, price_true_te_s, price_pred_te_s, f"TEST Price {cfg.model}", run_dir / "test_timeseries_price.png")
        except Exception:
            pass

    # 可选：特征重要性（xgb/lgbm/ridge/ols）；TFT注意力与变量权重
    try:
        if model_name in {"xgb", "lgbm"}:
            importances = getattr(model.model, "feature_importances_", None)
            if importances is not None and len(importances) == len(feat_cols):
                rpt.plot_feature_importance(feat_cols, importances, f"Feature Importance {cfg.model}", run_dir / "feature_importance.png")
        elif model_name in {"ridge", "ols"}:
            coefs = getattr(model, "coef_", None)
            if coefs is not None and len(coefs) == len(feat_cols):
                rpt.plot_feature_importance(feat_cols, np.abs(coefs), f"Coefficient Magnitude {cfg.model}", run_dir / "coef_magnitude.png")
        elif model_name == "tft":
            # 保存与可视化注意力/变量权重
            if 'tft_details' in locals() and tft_details is not None:
                for split in ['val', 'test']:
                    det = tft_details.get(split)
                    if not det:
                        continue
                    # 保存原始数组
                    for key in ['attn_mean', 'var_w_time', 'var_w_mean']:
                        arr = det.get(key)
                        if arr is not None:
                            np.save(run_dir / f"tft_{split}_{key}.npy", arr)
                    # 图
                    if det.get('attn_mean') is not None:
                        rpt.plot_attention_heatmap(det['attn_mean'], f"TFT Attention ({split.upper()})", run_dir / f"{split}_attention.png")
                    if det.get('var_w_mean') is not None:
                        rpt.plot_var_weights_bar(feat_cols, det['var_w_mean'], f"TFT Var Weights Mean ({split.upper()})", run_dir / f"{split}_var_weights_bar.png")
                    if det.get('var_w_time') is not None:
                        rpt.plot_var_weights_heatmap(det['var_w_time'], feat_cols, f"TFT Var Weights Heatmap ({split.upper()})", run_dir / f"{split}_var_weights_heatmap.png")
    except Exception:
        pass

    return metrics


def main_cli(cfg: TrainConfig):
    if cfg.model.lower() != "all":
        m = fit_and_eval(cfg)
        print("模型:", cfg.model)
        print("horizon:", cfg.horizon)
        # 打印本次运行目录（取 outputs 下最大的数字目录）
        try:
            subdirs = [p for p in config.OUT_DIR.iterdir() if p.is_dir() and p.name.isdigit()]
            if subdirs:
                run_dir = max(subdirs, key=lambda p: int(p.name))
                print("本次运行结果目录:", str(run_dir))
            else:
                print("输出目录:", str(config.OUT_DIR))
        except Exception:
            print("输出目录:", str(config.OUT_DIR))
        print("验证集:")
        print({k: v for k, v in m.items() if k.startswith("val_")})
        print("测试集:")
        print({k: v for k, v in m.items() if k.startswith("test_")})
        return

    # 运行全部模型
    all_models: List[str] = [
        "ridge", "ols", "svr", "xgb", "lgbm", "arima", "varx", "garch", "lstm", "gru", "cnn", "tft"
    ]
    session_dir = create_next_run_dir(config.OUT_DIR)
    rows = []
    print("批量运行，结果目录:", str(session_dir))
    for mdl in all_models:
        print(f"\n>>> 运行模型: {mdl}")
        sub_dir = session_dir / mdl
        sub_cfg = TrainConfig(
            data_path=cfg.data_path,
            horizon=cfg.horizon,
            train_end=cfg.train_end,
            val_end=cfg.val_end,
            model=mdl,
            ridge_alpha=cfg.ridge_alpha,
            out_dir=session_dir,  # 未使用（run_dir直接传入）
            seq_len=cfg.seq_len,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            hidden_size=cfg.hidden_size,
            lr=cfg.lr,
        )
        try:
            m = fit_and_eval(sub_cfg, run_dir=sub_dir)
            row = {"model": mdl, **m}
            rows.append(row)
            print("  完成.")
        except Exception as e:
            print(f"  失败: {e}")
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(session_dir / "summary_metrics.csv", index=False)
        with (session_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print("已写入:", str(session_dir / "summary_metrics.csv"))


if __name__ == "__main__":
    # 允许直接运行此脚本
    cfg = TrainConfig()
    main_cli(cfg)

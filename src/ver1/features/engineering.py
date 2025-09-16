#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def _logdiff(s: pd.Series) -> pd.Series:
    # 避免 log(0)
    return np.log(np.clip(s, 1e-8, None)).diff()


def _diff(s: pd.Series) -> pd.Series:
    return s.diff()


def _rol_mean(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(2, w // 2)).mean()


def _rol_std(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(2, w // 2)).std()


def make_features(raw: pd.DataFrame, h: int = 1, target_type: str = "logret") -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    仅使用连续数值特征（不含任何日历/事件虚拟变量）。
    - 目标 y:
      * logret: 下一期(h)对数收益 log(C_{t+h}) - log(C_t)
      * pctret: 下一期(h)简单收益 (C_{t+h}-C_t)/C_t
      * price:  下一期(h)的价格 C_{t+h}
    - 特征 X: BTC 技术面 + 宏观变量的变化与滞后
    返回: (X, y, dates_t, dates_future, c0_at_t)
    """
    df = raw.copy()

    # 目标：按 target_type 构建
    close = df["BTC_CLOSE"].astype(float)
    df["ret1"] = np.log(np.clip(close, 1e-8, None)).diff()
    ttype = (target_type or "logret").lower()
    if ttype == "logret":
        df["target"] = np.log(np.clip(close.shift(-h), 1e-8, None)) - np.log(np.clip(close, 1e-8, None))
    elif ttype == "pctret":
        df["target"] = (close.shift(-h) - close) / close
    elif ttype == "price":
        df["target"] = close.shift(-h)
    else:
        raise ValueError(f"未知 target_type: {target_type}")

    # BTC 技术特征（不引入虚拟变量）
    # 过去收益滞后
    for l in [1, 3, 7]:
        df[f"ret1_lag{l}"] = df["ret1"].shift(l)
    # 波动率与动量
    for w in [7, 14, 30]:
        df[f"ret_mean_{w}"] = _rol_mean(df["ret1"], w)
    for w in [7, 30]:
        df[f"ret_std_{w}"] = _rol_std(df["ret1"], w)
    # 成交量变化
    if "BTC_VOL" in df.columns:
        df["vol_chg"] = _logdiff(df["BTC_VOL"])  # log-volume diff
        for l in [1, 7]:
            df[f"vol_chg_lag{l}"] = df["vol_chg"].shift(l)

    # 价格相对均线
    for w in [7, 30]:
        ma = _rol_mean(df["BTC_CLOSE"], w)
        df[f"px_ma_ratio_{w}"] = df["BTC_CLOSE"] / ma

    # 宏观变量：变化 + 滞后（按类型）
    logdiff_vars = [c for c in ["VIX", "OIL", "DXY", "GOLD"] if c in df.columns]
    diff_vars    = [c for c in ["DGS5", "DFF"] if c in df.columns]

    for c in logdiff_vars:
        base = _logdiff(df[c])
        for l in [1, 3, 7, 14, 30]:
            df[f"{c}_ld_lag{l}"] = base.shift(l)

    for c in diff_vars:
        base = _diff(df[c])
        for l in [1, 3, 7, 14, 30]:
            df[f"{c}_d_lag{l}"] = base.shift(l)

    # 选择特征列
    drop_cols = ["target", "date"]
    feature_cols = [c for c in df.columns if c not in drop_cols and not c.startswith("BTC_")]
    # 也加入部分价格水平特征（避免泄漏：仅t时刻的）
    feature_cols += [c for c in ["BTC_CLOSE"] if c in df.columns]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    # 删除全空列与严重稀疏列
    nun = X.notna().sum(axis=0)
    keep_cols = [c for c in X.columns if nun[c] > 0 and nun[c] >= 0.5 * len(X)]
    X = X[keep_cols]
    y = df["target"]
    dates = df["date"].copy()

    # 清理空值：目标和特征均需有效
    valid = (~y.isna())
    for c in X.columns:
        valid &= ~X[c].isna()

    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    dates = dates.loc[valid].reset_index(drop=True)

    # 未来日期（与 target 对齐的日期）和基准价格 C_t
    fut_dates = (pd.to_datetime(dates) + pd.to_timedelta(h, unit="D")).reset_index(drop=True)
    c0 = close.loc[valid].reset_index(drop=True)

    return X, y, dates, fut_dates, c0

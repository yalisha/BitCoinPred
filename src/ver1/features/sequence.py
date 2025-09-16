#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd


def build_sequences(df: pd.DataFrame, seq_len: int) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, np.ndarray]:
    """
    接收列包含：
      - 必需：`date`, `target`
      - 可选：`date_future`, `c0`（用于从收益还原价格和价格对齐）
    构造长度为 seq_len 的滑动窗口：
      X_seq[i] = features[t-seq_len+1 : t]，y_seq[i] = target[t]
    返回：
      (X_seq: [N, seq_len, F], y_seq: [N], dates[t], date_future[t], c0[t])
    """
    drop_cols = ["date", "target"]
    if "date_future" in df.columns:
        drop_cols.append("date_future")
    if "c0" in df.columns:
        drop_cols.append("c0")

    features = df.drop(columns=drop_cols, errors="ignore").to_numpy(dtype=float)
    target = df["target"].to_numpy(dtype=float)
    dates = df.get("date")
    fut = df.get("date_future", df.get("date"))
    c0 = df.get("c0")
    dates = dates.copy() if dates is not None else pd.Series([])
    fut = fut.copy() if fut is not None else dates
    c0 = c0.to_numpy(dtype=float) if c0 is not None else np.full(len(target), np.nan, dtype=float)

    n, f = features.shape
    if n < seq_len:
        return np.zeros((0, seq_len, f)), np.zeros((0,)), dates.iloc[0:0]

    Xs, ys, ds, fs, cs = [], [], [], [], []
    for t in range(seq_len - 1, n):
        Xs.append(features[t - seq_len + 1: t + 1])
        ys.append(target[t])
        ds.append(dates.iloc[t])
        fs.append(fut.iloc[t])
        cs.append(c0[t])

    return np.asarray(Xs), np.asarray(ys), pd.Series(ds), pd.Series(fs), np.asarray(cs)

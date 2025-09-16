#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict

import pandas as pd


def load_macro_btc(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # 规范日期
    df["date"] = pd.to_datetime(df["date"])  # 保留时区无关的日期
    # 列顺序尽量固定
    cols = [
        "date", "BTC_OPEN", "BTC_HIGH", "BTC_LOW", "BTC_CLOSE", "BTC_VOL",
        "VIX", "OIL", "DXY", "GOLD", "DGS5", "DFF",
    ]
    # 若列不全，按现有列交集选择
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values("date").reset_index(drop=True)
    return df


def split_by_date(df: pd.DataFrame, train_end: str, val_end: str) -> Dict[str, pd.DataFrame]:
    train_m = df["date"] <= pd.to_datetime(train_end)
    val_m   = (df["date"] > pd.to_datetime(train_end)) & (df["date"] <= pd.to_datetime(val_end))
    test_m  = df["date"] > pd.to_datetime(val_end)
    return {
        "train": df.loc[train_m].reset_index(drop=True),
        "val":   df.loc[val_m].reset_index(drop=True),
        "test":  df.loc[test_m].reset_index(drop=True),
    }


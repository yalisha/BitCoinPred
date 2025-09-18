#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from . import config


def load_raw(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])  # naive date
    df = df.sort_values("date").reset_index(drop=True)

    # 数值列缺失值插补：先按时间插值，再前向/后向填补，避免后续取对数出现 NaN
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        df_num = df.set_index("date")[num_cols]
        df_num = df_num.interpolate(method="time", limit_direction="both")
        df_num = df_num.ffill().bfill()
        df_num = df_num.fillna(0.0)
        df[num_cols] = df_num.reset_index(drop=True)
    return df


def make_calendar_features(dates: pd.Series) -> pd.DataFrame:
    d = pd.to_datetime(dates)
    out = pd.DataFrame(index=d)
    out["dow"] = d.dt.weekday  # 0-6
    out["dom"] = d.dt.day
    out["month"] = d.dt.month
    out["doy"] = d.dt.dayofyear
    out["woy"] = d.dt.isocalendar().week.astype(int)
    out["is_month_end"] = d.dt.is_month_end.astype(int)
    out["is_quarter_end"] = d.dt.is_quarter_end.astype(int)
    out["is_year_end"] = d.dt.is_year_end.astype(int)
    out["is_month_start"] = d.dt.is_month_start.astype(int)
    out["is_quarter_start"] = d.dt.is_quarter_start.astype(int)
    out["is_year_start"] = d.dt.is_year_start.astype(int)
    out["is_weekend"] = (d.dt.weekday >= 5).astype(int)

    # 周期性编码（已知未来）
    two_pi = 2 * np.pi
    out["doy_sin"] = np.sin(two_pi * out["doy"] / 366.0)
    out["doy_cos"] = np.cos(two_pi * out["doy"] / 366.0)
    out["woy_sin"] = np.sin(two_pi * out["woy"] / 53.0)
    out["woy_cos"] = np.cos(two_pi * out["woy"] / 53.0)

    # One-hot for dow & month
    dow_oh = pd.get_dummies(out["dow"], prefix="dow")
    mon_oh = pd.get_dummies(out["month"], prefix="mon")
    out = pd.concat([out.drop(columns=["dow","month"]), dow_oh, mon_oh], axis=1)
    out.reset_index(drop=True, inplace=True)
    return out


def add_known_events(cal_df: pd.DataFrame, dates: pd.Series, window_days: int = 14) -> pd.DataFrame:
    """加入可提前知道的事件：
    - 比特币减半窗口（±window_days）
    - 常见月中节点（dom_1/dom_10/dom_12/dom_15/dom_20/dom_25）
    - 可选文件 data/events_known.csv: 列 [date,event_name]
    """
    d = pd.to_datetime(dates).reset_index(drop=True)
    df = cal_df.copy()

    # 减半日期
    halvings = [
        pd.Timestamp('2012-11-28'),
        pd.Timestamp('2016-07-09'),
        pd.Timestamp('2020-05-11'),
        pd.Timestamp('2024-04-20'),
    ]
    for i, h in enumerate(halvings, start=1):
        df[f"event_halving{i}"] = ((d >= h - pd.Timedelta(days=window_days)) & (d <= h + pd.Timedelta(days=window_days))).astype(int)

    # 常见月中节点（一些经济数据常于这些日子发布；近似占位）
    for k in [1, 10, 12, 15, 20, 25]:
        df[f"event_dom_{k}"] = (d.dt.day == k).astype(int)

    # 自定义事件（若存在）
    events_csv = Path(__file__).resolve().parents[2] / 'data' / 'events_known.csv'
    if events_csv.exists():
        # 允许 CSV 中包含以 # 开头的注释行；严格使用 YYYY-MM-DD 格式
        ev = pd.read_csv(events_csv, comment='#', keep_default_na=False)
        if {'date','event_name'}.issubset(ev.columns):
            ev['date'] = pd.to_datetime(ev['date'].astype(str).str.strip(), format='%Y-%m-%d', errors='coerce')
            ev['event_name'] = ev['event_name'].astype(str).str.strip().str.lower()
            ev = ev.dropna(subset=['date'])
            if not ev.empty:
                ev['v'] = 1
                piv = ev.pivot_table(index='date', columns='event_name', values='v', aggfunc='max', fill_value=0)
                piv = piv.reindex(d, fill_value=0)
                piv.columns = [f"event_{str(c).lower()}" for c in piv.columns]
                piv.reset_index(drop=True, inplace=True)
                df = pd.concat([df, piv], axis=1)
    return df


def _logdiff(s: pd.Series) -> pd.Series:
    return np.log(np.clip(s, 1e-8, None)).diff()


def build_feature_frame(raw: pd.DataFrame, target_type: str = "price") -> pd.DataFrame:
    """
    构造特征并标注三类变量：
      - 观测类 observed_*（仅过去可用）
      - 已知未来 known_*（日历虚拟变量等）
      - 静态类 static_*（单资产：常数1）
    并生成目标：price/logprice/rel_logprice/logret/pctret
    """
    df = raw.copy()
    close = df["BTC_CLOSE"].astype(float)

    # 目标
    if target_type == "price":
        df["target"] = close
    elif target_type == "logprice":
        df["target"] = np.log(np.clip(close, 1e-8, None))
    elif target_type == "rel_logprice":
        # 相对对数价格：在 make_sequences 按窗口构造
        pass
    elif target_type == "logret":
        df["target"] = _logdiff(close).shift(-1)
    elif target_type == "pctret":
        df["target"] = (close.pct_change()).shift(-1)
    else:
        raise ValueError("unknown target_type")

    # 观测类：BTC 技术 + 宏观差分
    df["ret1"] = _logdiff(close)
    for l in [1,3,7,14,30]:
        df[f"obs_ret1_lag{l}"] = df["ret1"].shift(l)
    for w in [7,14,30]:
        df[f"obs_ret_mean_{w}"] = df["ret1"].rolling(w, min_periods=w//2).mean()
        df[f"obs_ret_std_{w}"]  = df["ret1"].rolling(w, min_periods=w//2).std()
    if "BTC_VOL" in df.columns:
        df["obs_vol_ld"] = _logdiff(df["BTC_VOL"]) 
        for l in [1,7,14]:
            df[f"obs_vol_ld_lag{l}"] = df["obs_vol_ld"].shift(l)

    logdiff_vars = [c for c in ["VIX","OIL","DXY","GOLD"] if c in df.columns]
    diff_vars    = [c for c in ["DGS5","DFF"] if c in df.columns]
    for c in logdiff_vars:
        base = _logdiff(df[c])
        for l in [1,3,7,14,30]:
            df[f"obs_{c}_ld_lag{l}"] = base.shift(l)
    for c in diff_vars:
        base = df[c].diff()
        for l in [1,3,7,14,30]:
            df[f"obs_{c}_d_lag{l}"] = base.shift(l)

    # 已知未来：日历特征 + 事件
    cal = make_calendar_features(df["date"])  # index对齐
    cal = add_known_events(cal, df["date"], window_days=14)
    cal.columns = [f"kn_{c}" for c in cal.columns]
    df = pd.concat([df.reset_index(drop=True), cal], axis=1)

    # 静态特征：常数1
    df["st_const"] = 1.0

    return df


@dataclass
class SeqData:
    X_obs: np.ndarray   # [N, T_enc, F_obs]
    X_known: np.ndarray # [N, T_dec, F_kn]
    X_static: np.ndarray# [N, F_st]
    y: np.ndarray       # [N, T_dec] (price or ret)
    dates_t: pd.Series  # 观测末端日期 t
    dates_fut: pd.Series# 未来对齐日期 t+1..t+H (仅保存末端日期方便对齐)
    c0: np.ndarray      # 基准价格 C_t，用于从收益还原价格
    feat_names: Dict[str, list]


def make_sequences(df: pd.DataFrame,
                   seq_len: int,
                   horizon: int,
                   target_type: str,
                   anchor_lag: Optional[int] = None) -> SeqData:
    # 列分组
    obs_cols = [c for c in df.columns if c.startswith("obs_")]
    kn_cols  = [c for c in df.columns if c.startswith("kn_")]
    st_cols  = [c for c in df.columns if c.startswith("st_")]

    # 为避免窗口全被 NaN 跳过：对观测特征做前向/后向填充
    obs_df = df[obs_cols].copy()
    # 使用新API避免FutureWarning
    obs_df = obs_df.ffill().bfill()
    X_obs = obs_df.to_numpy(dtype=float)

    kn_df = df[kn_cols].copy()
    kn_df = kn_df.fillna(0.0)
    X_kn  = kn_df.to_numpy(dtype=float)

    st_df = df[st_cols].copy()
    st_df = st_df.fillna(0.0)
    X_st  = st_df.to_numpy(dtype=float)
    y_all = df.get("target", pd.Series([np.nan]*len(df))).to_numpy(dtype=float)
    dates = df["date"].copy()
    c0_all= df["BTC_CLOSE"].to_numpy(dtype=float)
    log_close = np.log(np.clip(c0_all, 1e-8, None))

    n = len(df)
    T_enc = seq_len
    T_dec = horizon

    Xo_list, Xk_list, Xs_list, y_list, dt_list, dfut_list, c0_list = [], [], [], [], [], [], []
    anchor_lag_value = anchor_lag
    if anchor_lag_value is None:
        anchor_cfg = getattr(config, "ANCHOR_LAG", None)
        if anchor_cfg is None:
            anchor_lag_value = None
        else:
            anchor_lag_value = int(anchor_cfg)
    if anchor_lag_value is not None and anchor_lag_value < 0:
        anchor_lag_value = None

    for t in range(T_enc-1, n - T_dec):
        # 编码器窗口: [t-seq_len+1, t]
        Xo = X_obs[t-T_enc+1: t+1]
        # 解码器窗口: 已知未来 [t+1, t+H]
        Xk = X_kn[t+1: t+1+T_dec]
        # 可选：加入滞后 log 价格锚点
        anchors = []
        if anchor_lag_value is not None:
            lag_idx = max(0, t - anchor_lag_value)
            logc_anchor = log_close[lag_idx]
            anchors.append((f"kn_anchor_logc0_lag{anchor_lag_value}", np.full((T_dec, 1), logc_anchor, dtype=float)))
        if anchors:
            for _, arr in anchors:
                Xk = np.concatenate([Xk, arr], axis=1)
        # 静态
        Xs = X_st[t]
        # 输出目标: [t+1, t+H]
        if target_type == "rel_logprice":
            y = log_close[t+1: t+1+T_dec] - log_close[t]
        else:
            y  = y_all[t+1: t+1+T_dec]
        # 日期
        d_t = dates.iloc[t]
        d_f = dates.iloc[t+T_dec]
        c0  = c0_all[t]

        if np.any(np.isnan(y)):
            continue

        Xo_list.append(Xo)
        Xk_list.append(Xk)
        Xs_list.append(Xs)
        y_list.append(y)
        dt_list.append(d_t)
        dfut_list.append(d_f)
        c0_list.append(c0)

    kn_cols_aug = list(kn_cols)
    if anchor_lag_value is not None:
        kn_cols_aug.append(f"kn_anchor_logc0_lag{anchor_lag_value}")

    seq = SeqData(
        X_obs=np.asarray(Xo_list),
        X_known=np.asarray(Xk_list),
        X_static=np.asarray(Xs_list),
        y=np.asarray(y_list),
        dates_t=pd.Series(dt_list),
        dates_fut=pd.Series(dfut_list),
        c0=np.asarray(c0_list),
        feat_names={"obs": obs_cols, "known": kn_cols_aug, "static": st_cols},
    )
    # 容错：若无可用样本，给出友好提示
    if seq.X_obs.ndim != 3 or seq.X_obs.shape[0] == 0:
        raise ValueError(
            "No valid sequences constructed. Try reducing SEQ_LEN/HORIZON or ensure features have fewer NaNs."
        )
    return seq


def split_by_date(df: pd.DataFrame, train_end: str, val_end: str) -> Dict[str, pd.DataFrame]:
    m_tr = df["date"] <= pd.to_datetime(train_end)
    m_va = (df["date"] > pd.to_datetime(train_end)) & (df["date"] <= pd.to_datetime(val_end))
    m_te = df["date"] > pd.to_datetime(val_end)
    return {
        "train": df.loc[m_tr].reset_index(drop=True),
        "val":   df.loc[m_va].reset_index(drop=True),
        "test":  df.loc[m_te].reset_index(drop=True),
    }

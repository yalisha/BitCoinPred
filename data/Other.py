#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一抓取（2014-01-01 ~ 2025-09-14，UTC日频）：
  宏观变量（全部 FRED）：
    - VIX = VIXCLS
    - OIL = DCOILWTICO
    - DXY = DTWEXBGS
    - GOLD = GOLDAMGBD228NLBM
    - DGS5 = DGS5
    - DFF  = DFF
  BTC 价格（Bitstamp 2014~2017-08-16 + Binance 2017-08-17~；若 Binance 失败则回退 Coinbase Pro）
导出 CSV/Excel：macro_btc_2014_2025_daily.*

依赖: requests, pandas, python-dateutil
环境变量: FRED_API_KEY=你的key
"""

import os
import time
from datetime import datetime, timedelta, timezone
import requests
import pandas as pd

# ===================== 配置 =====================
START_DATE = "2014-01-01"
END_DATE   = "2025-09-14"

OUT_CSV   = "macro_btc_2014_2025_daily.csv"
OUT_XLSX  = "macro_btc_2014_2025_daily.xlsx"

# FRED 序列映射（全部走 FRED）
FRED_SERIES = {
    "VIX":   "VIXCLS",               # CBOE Volatility Index
    "OIL":   "DCOILWTICO",           # WTI 原油现货
    "DXY":   "DTWEXBGS",             # Broad Dollar Index
    "GOLD":  "GOLDAMGBD228NLBM",     # LBMA Gold AM USD/oz
    "DGS5":  "DGS5",                 # 5Y UST Yield
    "DFF":   "DFF",                  # Fed Funds Effective Rate
}

# ===================== 工具函数 =====================
def to_unix(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

def backoff(i: int):
    time.sleep(min(1.5 ** i, 10))

def retry_get(url, params=None, headers=None, max_retry=6, timeout=30):
    for i in range(max_retry):
        try:
            r = requests.get(url, params=params, headers=headers or {"Accept":"application/json"}, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (418, 429, 500, 502, 503, 504):
                backoff(i); continue
            backoff(i)
        except requests.RequestException:
            backoff(i)
    return None

# ===================== FRED 段 =====================
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

def fetch_fred_series(series_id: str, start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
    }
    r = retry_get(FRED_BASE, params=params)
    if r is None:
        print(f"⚠️ FRED 获取失败：{series_id}")
        return pd.DataFrame(columns=["date", series_id])
    obs = (r.json() or {}).get("observations", [])
    if not obs:
        return pd.DataFrame(columns=["date", series_id])
    df = pd.DataFrame(obs)[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
    df["date"]  = pd.to_datetime(df["date"]).dt.date
    return df.rename(columns={"value": series_id})

def get_all_fred(series_map: dict, start_date: str, end_date: str) -> pd.DataFrame:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到 FRED_API_KEY，请先设置环境变量。")
    frames = []
    for name, sid in series_map.items():
        print(f"FRED: 抓取 {name} ({sid}) …")
        df = fetch_fred_series(sid, start_date, end_date, api_key)
        df = df.rename(columns={sid: name})
        frames.append(df)
        time.sleep(0.2)
    # 外连接并用完整日历前向填充
    out = None
    for df in frames:
        out = df if out is None else pd.merge(out, df, on="date", how="outer")
    cal = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="D").date})
    out = pd.merge(cal, out, on="date", how="left").sort_values("date").reset_index(drop=True)
    cols = [c for c in out.columns if c != "date"]
    out[cols] = out[cols].ffill()
    return out

# ===================== BTC 段（多源兜底） =====================
# Bitstamp：2014-01-01 ~ 2017-08-16
BITSTAMP_URL = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"

def fetch_bitstamp_range(start_unix: int, end_unix: int) -> pd.DataFrame:
    chunk_days = 900
    frames = []
    cur = start_unix
    while cur <= end_unix:
        seg_end = min(end_unix, cur + (chunk_days - 1)*86400)
        params = {"step": 86400, "limit": 1000, "start": cur, "end": seg_end}
        r = retry_get(BITSTAMP_URL, params=params)
        if r is not None:
            j = r.json()
            rows = (j or {}).get("data", {}).get("ohlc", [])
            if rows:
                df = pd.DataFrame(rows)
                for c in ["open","high","low","close","volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True).dt.tz_convert(None).dt.date
                frames.append(df[["date","open","high","low","close","volume"]])
        cur = seg_end + 86400
        time.sleep(0.4)
    if not frames:
        return pd.DataFrame(columns=["date","open","high","low","close","volume"])
    out = (pd.concat(frames, ignore_index=True)
             .drop_duplicates(subset=["date"])
             .sort_values("date")
             .reset_index(drop=True))
    return out

# Binance：2017-08-17 ~ 结束；失败回退 Coinbase Pro
BINANCE_BASES = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]

def fetch_binance_klines_range(start_ms: int, end_ms: int) -> pd.DataFrame:
    interval_ms = 24*3600*1000
    frames = []
    cur = start_ms
    while cur <= end_ms:
        nxt = min(end_ms, cur + 900*interval_ms - 1)
        got = False
        for base in BINANCE_BASES:
            url = f"{base}/api/v3/klines"
            params = {"symbol":"BTCUSDT","interval":"1d","startTime":cur,"endTime":nxt,"limit":1000}
            r = retry_get(url, params=params)
            if r is None:
                continue
            data = r.json()
            if not isinstance(data, list) or not data:
                continue
            cols = ["open_time","open","high","low","close","volume","close_time",
                    "qav","trades","taker_base","taker_quote","ignore"]
            df = pd.DataFrame(data, columns=cols)
            for c in ["open","high","low","close","volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None).dt.date
            frames.append(df[["date","open","high","low","close","volume","open_time"]])
            last_open = int(df["open_time"].iloc[-1])
            cur = last_open + interval_ms   # 无缝翻页
            got = True
            break
        if not got:
            cur = nxt + 1
        time.sleep(0.3)
    if not frames:
        return pd.DataFrame(columns=["date","open","high","low","close","volume"])
    out = (pd.concat(frames, ignore_index=True)
             .drop_duplicates(subset=["date"])
             .sort_values("date")
             .reset_index(drop=True))
    return out[["date","open","high","low","close","volume"]]

# Coinbase Pro 回退
CBP_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"

def iso_utc(ts: int) -> str:
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")

def fetch_coinbase_candles_range(start_unix: int, end_unix: int) -> pd.DataFrame:
    step = 300  # 300 天/页
    frames = []
    cur = start_unix
    while cur <= end_unix:
        nxt = min(end_unix, cur + (step - 1)*86400)
        params = {"granularity":86400,"start":iso_utc(cur),"end":iso_utc(nxt + 86399)}
        r = retry_get(CBP_URL, params=params)
        if r is not None:
            data = r.json()
            if isinstance(data, list) and data:
                df = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
                df = df.sort_values("time")
                for c in ["open","high","low","close","volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None).dt.date
                frames.append(df[["date","open","high","low","close","volume"]])
        cur = nxt + 86400
        time.sleep(0.3)
    if not frames:
        return pd.DataFrame(columns=["date","open","high","low","close","volume"])
    out = (pd.concat(frames, ignore_index=True)
             .drop_duplicates(subset=["date"])
             .sort_values("date")
             .reset_index(drop=True))
    return out

def get_btc_daily(start_date: str, end_date: str) -> pd.DataFrame:
    split_date = "2017-08-17"
    print("BTC: 抓取 Bitstamp 2014-01-01 ~ 2017-08-16 …")
    bs = fetch_bitstamp_range(to_unix(start_date), to_unix(split_date) - 86400)
    print(f"  Bitstamp 行数: {len(bs)}")

    print("BTC: 抓取 Binance 2017-08-17 ~ 结束 …")
    bn = fetch_binance_klines_range(to_unix(split_date)*1000,
                                    to_unix(end_date)*1000 + (24*3600*1000 - 1))
    print(f"  Binance 行数: {len(bn)}")

    if bn.empty:
        print("  Binance 为空，回退 Coinbase Pro …")
        cb = fetch_coinbase_candles_range(to_unix(split_date), to_unix(end_date))
        print(f"  Coinbase Pro 行数: {len(cb)}")
        post = cb
    else:
        post = bn

    all_df = pd.concat([bs, post], ignore_index=True)
    if all_df.empty:
        raise RuntimeError("BTC 价格未获取到，请检查网络/代理。")

    daily = (all_df.groupby("date", as_index=False)
                   .agg(open=("open","first"),
                        high=("high","max"),
                        low=("low","min"),
                        close=("close","last"),
                        volume=("volume","sum")))

    m = (pd.to_datetime(daily["date"]) >= pd.to_datetime(start_date)) & \
        (pd.to_datetime(daily["date"]) <= pd.to_datetime(end_date))
    daily = daily.loc[m].sort_values("date").reset_index(drop=True)

    return daily.rename(columns={
        "open":"BTC_OPEN","high":"BTC_HIGH","low":"BTC_LOW","close":"BTC_CLOSE","volume":"BTC_VOL"
    })

# ===================== 主流程 =====================
def main():
    # 1) FRED 宏观
    fred_df = get_all_fred(FRED_SERIES, START_DATE, END_DATE)

    # 2) BTC
    btc_df = get_btc_daily(START_DATE, END_DATE)

    # 3) 合并对齐（完整日历为基准）
    cal = pd.DataFrame({"date": pd.date_range(START_DATE, END_DATE, freq="D").date})
    out = cal.merge(btc_df, on="date", how="left").merge(fred_df, on="date", how="left")

    # 4) 导出
    out.to_csv(OUT_CSV, index=False)
    with pd.ExcelWriter(OUT_XLSX) as w:
        out.to_excel(w, index=False, sheet_name="Daily")

    print(f"\n完成：{len(out)} 天")
    print(f"- {OUT_CSV}\n- {OUT_XLSX}")
    print("\n预览：")
    print(out.head(3))
    print("…")
    print(out.tail(3))

if __name__ == "__main__":
    main()
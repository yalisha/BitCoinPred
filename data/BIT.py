#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
拉取 2014-01-01 ~ 2025-09-14 的 BTC 日线 OHLCV（UTC），三源兜底：
- 2014-01-01 ~ 2017-08-16：Bitstamp BTCUSD
- 2017-08-17 ~ 2025-09-14：优先 Binance BTCUSDT；若失败回退 Coinbase Pro BTC-USD
无需 API Key。自动分页与限流重试。
"""

import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import requests
import pandas as pd

START_DATE = "2014-01-01"
SPLIT_DATE = "2017-08-17"   # Binance 最早K线（含当日）
END_DATE   = "2025-09-14"

OUT_CSV   = "btc_usd_daily_2014_2025.csv"
OUT_XLSX  = "btc_usd_daily_2014_2025.xlsx"

# ---------------- 工具 ----------------
def to_unix(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

def clamp(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if df.empty: return df
    m = (pd.to_datetime(df["date"]) >= pd.to_datetime(start_date)) & \
        (pd.to_datetime(df["date"]) <= pd.to_datetime(end_date))
    return df.loc[m].sort_values("date").reset_index(drop=True)

def retry_get(url, params=None, headers=None, max_retry=6, backoff_base=1.2):
    for i in range(max_retry):
        try:
            r = requests.get(url, params=params, headers=headers or {"Accept": "application/json"}, timeout=30)
            if r.status_code == 200:
                return r
            # 常见限流/网关
            if r.status_code in (418, 429, 500, 502, 503, 504):
                time.sleep(min(backoff_base ** i, 10))
                continue
            # 其他错误也稍微重试
            time.sleep(min(backoff_base ** i, 5))
        except requests.RequestException:
            time.sleep(min(backoff_base ** i, 10))
    return None

# ---------------- Bitstamp（日线 OHLCV，免 Key） ----------------
# API: https://www.bitstamp.net/api/v2/ohlc/btcusd/?step=86400&limit=1000&start=...&end=...
BITSTAMP_URL = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"

def fetch_bitstamp_ohlc_range(start_unix: int, end_unix: int) -> pd.DataFrame:
    """分块抓取，最多 1000 根/次，这里用 900 天窗口更稳妥。"""
    chunk_days = 900
    frames = []
    cur = start_unix
    while cur <= end_unix:
        seg_end = min(end_unix, cur + (chunk_days - 1) * 86400)
        params = {"step": 86400, "limit": 1000, "start": cur, "end": seg_end}
        r = retry_get(BITSTAMP_URL, params=params)
        if r is None:
            cur = seg_end + 86400
            continue
        j = r.json()
        rows = (j or {}).get("data", {}).get("ohlc", [])
        if rows:
            df = pd.DataFrame(rows)
            for col in ["open","high","low","close","volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True).dt.tz_convert(None).dt.date
            frames.append(df[["date","open","high","low","close","volume"]])
        cur = seg_end + 86400
        time.sleep(0.5)
    if frames:
        out = (pd.concat(frames, ignore_index=True)
                 .drop_duplicates(subset=["date"])
                 .sort_values("date")
                 .reset_index(drop=True))
        return out
    return pd.DataFrame(columns=["date","open","high","low","close","volume"])

# ---------------- Binance（日线 klines，免 Key） ----------------
# API: GET /api/v3/klines?symbol=BTCUSDT&interval=1d&startTime&endTime&limit=1000
BINANCE_BASES = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]

def fetch_binance_klines_range(start_ms: int, end_ms: int) -> pd.DataFrame:
    """用多个域名兜底；严格用 open_time 分页，避免窗口缝隙导致漏数据。"""
    interval_ms = 24*3600*1000
    frames = []
    cur = start_ms
    while cur <= end_ms:
        nxt = min(end_ms, cur + 900*interval_ms - 1)  # 900 根一页
        ok = False
        for base in BINANCE_BASES:
            url = f"{base}/api/v3/klines"
            params = {
                "symbol":"BTCUSDT",
                "interval":"1d",
                "startTime":cur,
                "endTime":nxt,
                "limit":1000
            }
            r = retry_get(url, params=params)
            if r is None:  # 换下一个域名
                continue
            data = r.json()
            if not isinstance(data, list) or len(data) == 0:
                # 这页可能没有数据，尝试下一个域名；都没有就跳页
                continue
            cols = ["open_time","open","high","low","close","volume","close_time",
                    "qav","trades","taker_base","taker_quote","ignore"]
            df = pd.DataFrame(data, columns=cols)
            for c in ["open","high","low","close","volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None).dt.date
            frames.append(df[["date","open","high","low","close","volume","open_time"]])
            # 用最后一根的 open_time + 1 天作为下一窗口开头（无缝衔接）
            last_open = int(df["open_time"].iloc[-1])
            cur = last_open + interval_ms
            ok = True
            break  # 本页成功就不再换域名
        if not ok:
            # 这一页所有域名都失败：推进一个窗口，防卡死
            cur = nxt + 1
        time.sleep(0.4)
    if frames:
        out = (pd.concat(frames, ignore_index=True)
                 .drop_duplicates(subset=["date"])
                 .sort_values("date")
                 .reset_index(drop=True))
        return out[["date","open","high","low","close","volume"]]
    return pd.DataFrame(columns=["date","open","high","low","close","volume"])

# ---------------- Coinbase Pro（日线 candles，免 Key，回退用） ----------------
# API: https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=86400&start=ISO&end=ISO
CBP_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"

def iso_utc(ts: int) -> str:
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")

def fetch_coinbase_candles_range(start_unix: int, end_unix: int) -> pd.DataFrame:
    """Coinbase Pro 一次最多返回 300 根，按 300 天分页；返回 [time, low, high, open, close, volume]，time 为 bucket 起始秒（UTC）。"""
    step = 300  # 天
    frames = []
    cur = start_unix
    while cur <= end_unix:
        nxt = min(end_unix, cur + (step - 1)*86400)
        params = {"granularity": 86400, "start": iso_utc(cur), "end": iso_utc(nxt + 86399)}
        r = retry_get(CBP_URL, params=params, headers={"Accept": "application/json"})
        if r is None:
            cur = nxt + 86400
            continue
        data = r.json()
        if isinstance(data, list) and data:
            # Coinbase 返回是乱序（新在前），需要排序
            df = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
            df = df.sort_values("time")
            for c in ["open","high","low","close","volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None).dt.date
            frames.append(df[["date","open","high","low","close","volume"]])
        cur = nxt + 86400
        time.sleep(0.4)
    if frames:
        out = (pd.concat(frames, ignore_index=True)
                 .drop_duplicates(subset=["date"])
                 .sort_values("date")
                 .reset_index(drop=True))
        return out
    return pd.DataFrame(columns=["date","open","high","low","close","volume"])

# ---------------- 主流程 ----------------
def main():
    start_unix = to_unix(START_DATE)
    split_unix = to_unix(SPLIT_DATE)
    end_unix   = to_unix(END_DATE)

    print("① Bitstamp 2014-01-01 ~ 2017-08-16 …")
    bs = fetch_bitstamp_ohlc_range(start_unix, split_unix - 86400)
    print(f"   Bitstamp 行数: {len(bs)}")

    print("② Binance 2017-08-17 ~ 2025-09-14 …")
    bn = fetch_binance_klines_range(split_unix*1000, end_unix*1000 + (24*3600*1000 - 1))
    print(f"   Binance 行数: {len(bn)}")

    if bn.empty:
        print("   Binance 为空，回退 Coinbase Pro …")
        cb = fetch_coinbase_candles_range(split_unix, end_unix)
        print(f"   Coinbase Pro 行数: {len(cb)}")
        post = cb
    else:
        post = bn

    # 合并、聚合（日重叠去重）
    all_df = pd.concat([bs, post], ignore_index=True)
    if all_df.empty:
        raise RuntimeError("三个数据源都未取到数据，请检查网络/代理后再试。")

    daily = (all_df.groupby("date", as_index=False)
                   .agg(open=("open","first"),
                        high=("high","max"),
                        low=("low","min"),
                        close=("close","last"),
                        volume=("volume","sum")))

    daily = clamp(daily, START_DATE, END_DATE)

    # 导出
    daily.to_csv(OUT_CSV, index=False)
    with pd.ExcelWriter(OUT_XLSX) as w:
        daily.to_excel(w, index=False, sheet_name="BTC-USD-Daily")

    print(f"\n完成：{len(daily)} 天")
    print(f"- {OUT_CSV}\n- {OUT_XLSX}")
    print("\n预览：")
    print(daily.head(3))
    print("…")
    print(daily.tail(3))

if __name__ == "__main__":
    main()
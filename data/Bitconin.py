#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
抓取 2014-01-01 ~ 2025-09-14 BTC 日线 OHLCV（UTC）并导出：
- 2014-01-01 ~ 2017-08-16：Bitstamp BTCUSD（无需key）
- 2017-08-17 ~ 2025-09-14：Binance BTCUSDT（无需key）
合并去重、按日聚合，导出 CSV/Excel。
"""

import time
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Optional

import requests
import pandas as pd

# ---------------- 配置 ----------------
START_DATE = "2014-01-01"
SPLIT_DATE = "2017-08-17"  # Binance BTCUSDT 最早K线：2017-08-17
END_DATE   = "2025-09-14"

OUT_CSV   = "btc_usd_daily_2014_2025_combined.csv"
OUT_EXCEL = "btc_usd_daily_2014_2025_combined.xlsx"

# ---------------- 工具 ----------------
def to_unix(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

def parse_date(ts_ms_or_s: int) -> datetime.date:
    # 自动识别毫秒/秒
    if ts_ms_or_s > 10**12:
        dt = datetime.utcfromtimestamp(ts_ms_or_s / 1000.0)
    else:
        dt = datetime.utcfromtimestamp(ts_ms_or_s)
    return dt.date()

def daterange(start: datetime, end: datetime, step_days: int):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=step_days)

def backoff(i: int):
    time.sleep(min(2 ** i, 10))

# ---------------- Bitstamp：2014-01-01 ~ 2017-08-16 ----------------
# API: https://www.bitstamp.net/api/v2/ohlc/btcusd/?step=86400&limit=1000&start=...&end=...
BITSTAMP_URL = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"

def fetch_bitstamp_ohlc_day(start_unix: int, end_unix: int, max_retries: int = 5) -> pd.DataFrame:
    params = {"step": 86400, "limit": 1000, "start": start_unix, "end": end_unix}
    for i in range(max_retries):
        try:
            r = requests.get(BITSTAMP_URL, params=params, timeout=30)
            if r.status_code != 200:
                backoff(i); continue
            j = r.json()
            data = j.get("data", {})
            ohlc = data.get("ohlc", [])
            if not ohlc:
                return pd.DataFrame()
            df = pd.DataFrame(ohlc)
            # Bitstamp 返回为字符串，需要转数值
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True).dt.tz_convert(None).dt.date
            df = df[["date", "open", "high", "low", "close", "volume"]]
            return df
        except requests.RequestException:
            backoff(i)
    return pd.DataFrame()

def get_bitstamp_range(start_date: str, end_date: str) -> pd.DataFrame:
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt   = datetime.strptime(end_date,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
    # Bitstamp 每次最多 1000 根日K，安全起见按 ~2.5 年一块分段
    chunk_days = 900
    pieces = []
    cur = start_dt
    while cur <= end_dt:
        seg_end = min(end_dt, cur + timedelta(days=chunk_days-1))
        df = fetch_bitstamp_ohlc_day(int(cur.timestamp()), int(seg_end.timestamp()))
        if not df.empty:
            pieces.append(df)
        cur = seg_end + timedelta(days=1)
        time.sleep(0.7)  # 轻微限频
    if pieces:
        out = pd.concat(pieces, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
        return out
    return pd.DataFrame(columns=["date","open","high","low","close","volume"])

# ---------------- Binance：2017-08-17 ~ 现在 ----------------
# API: https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&startTime=...&endTime=...&limit=1000
BINANCE_URL = "https://api.binance.com/api/v3/klines"

def fetch_binance_klines(symbol: str, interval: str, start_ms: int, end_ms: int, limit: int = 1000, max_retries: int = 5) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "startTime": start_ms, "endTime": end_ms, "limit": limit}
    for i in range(max_retries):
        try:
            r = requests.get(BINANCE_URL, params=params, timeout=30)
            if r.status_code != 200:
                backoff(i); continue
            arr = r.json()
            if not isinstance(arr, list) or not arr:
                return pd.DataFrame()
            cols = ["open_time","open","high","low","close","volume","close_time",
                    "quote_asset_volume","number_of_trades","taker_buy_base","taker_buy_quote","ignore"]
            df = pd.DataFrame(arr, columns=cols)
            for col in ["open","high","low","close","volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None).dt.date
            return df[["date","open","high","low","close","volume","open_time","close_time"]]
        except requests.RequestException:
            backoff(i)
    return pd.DataFrame()

def get_binance_range(start_date: str, end_date: str) -> pd.DataFrame:
    start_ms = to_unix(start_date) * 1000
    end_ms   = to_unix(end_date) * 1000 + 24*3600*1000 - 1  # 包含 end 当天
    pieces = []
    cur = start_ms
    while cur <= end_ms:
        # Binance 每次最多 1000 根，1d 间隔 -> 约 1000 天
        # 我们按 900 天窗口分页更保险
        window_days = 900
        nxt = min(end_ms, cur + window_days*24*3600*1000 - 1)
        df = fetch_binance_klines("BTCUSDT", "1d", cur, nxt, limit=1000)
        if df.empty:
            break
        pieces.append(df)
        # 下一个窗口从最后一根k线的 close_time+1ms 开始
        last_close_ms = int(pd.to_datetime(df["date"].iloc[-1]).to_datetime64().astype("datetime64[ms]").astype(int)) + (24*3600*1000 - 1)
        cur = last_close_ms + 1
        time.sleep(0.7)
    if pieces:
        out = pd.concat(pieces, ignore_index=True).drop_duplicates(subset=["date"]).sort_values("date")
        return out[["date","open","high","low","close","volume"]]
    return pd.DataFrame(columns=["date","open","high","low","close","volume"])

# ---------------- 合并与导出 ----------------
def clamp(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if df.empty: return df
    m = (pd.to_datetime(df["date"]) >= pd.to_datetime(start_date)) & \
        (pd.to_datetime(df["date"]) <= pd.to_datetime(end_date))
    out = df.loc[m].sort_values("date").reset_index(drop=True)
    # 若某些源缺失值，前向填充 close 作为 open/high/low（可选）
    return out

def main():
    print("① 抓取 Bitstamp (BTCUSD) 2014-01-01 ~ 2017-08-16 …")
    bs_df = get_bitstamp_range(START_DATE, (datetime.strptime(SPLIT_DATE, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d"))
    print(f"   Bitstamp 行数: {len(bs_df)}")

    print("② 抓取 Binance (BTCUSDT) 2017-08-17 ~ 2025-09-14 …")
    bn_df = get_binance_range(SPLIT_DATE, END_DATE)
    print(f"   Binance 行数: {len(bn_df)}")

    # 统一列并合并
    all_df = pd.concat([bs_df, bn_df], ignore_index=True)
    if all_df.empty:
        raise RuntimeError("未获取到任何数据，请检查网络/代理或稍后重试。")

    # 按日聚合（双源重叠的边界日做去重与聚合）
    daily = (all_df.groupby("date", as_index=False)
                   .agg(open=("open","first"),
                        high=("high","max"),
                        low=("low","min"),
                        close=("close","last"),
                        volume=("volume","sum")))

    daily = clamp(daily, START_DATE, END_DATE)

    # 导出
    daily.to_csv(OUT_CSV, index=False)
    with pd.ExcelWriter(OUT_EXCEL) as w:
        daily.to_excel(w, index=False, sheet_name="BTC-USD-Daily")

    print(f"\n完成：{len(daily)} 天")
    print(f"- {OUT_CSV}\n- {OUT_EXCEL}")
    print("\n预览：")
    print(daily.head(3))
    print("…")
    print(daily.tail(3))

if __name__ == "__main__":
    main()
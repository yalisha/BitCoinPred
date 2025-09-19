from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Dict

import pandas as pd

from ..config import CONFIG, DataSourceConfig


def load_gold_history(config: DataSourceConfig | None = None) -> pd.DataFrame:
    """读取黄金历史价格数据，并标准化列名与索引。"""

    cfg = config or CONFIG.data
    loader = _get_loader(cfg.provider)
    raw_df = loader(cfg)
    df = _normalise_dataframe(raw_df, cfg)
    df = df.loc[(df.index >= cfg.start_date)]
    if cfg.end_date:
        df = df.loc[(df.index <= cfg.end_date)]
    return df


def _get_loader(provider: str) -> Callable[[DataSourceConfig], pd.DataFrame]:
    registry: Dict[str, Callable[[DataSourceConfig], pd.DataFrame]] = {
        "local_csv": _load_from_csv,
        "yfinance": _load_from_yfinance,
        "fred": _load_from_fred,
    }
    if provider not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unsupported data provider '{provider}'. Available: {available}")
    return registry[provider]


def _load_from_csv(config: DataSourceConfig) -> pd.DataFrame:
    if not config.csv_path.exists():
        raise FileNotFoundError(
            f"Local CSV '{config.csv_path}' 不存在，请先下载或放置 44 年黄金价格数据。"
        )
    df = pd.read_csv(config.csv_path, parse_dates=True, index_col=0)
    if not isinstance(df.index, pd.DatetimeIndex):
        datetime_col = None
        for candidate in ("Date", "DATE", "date"):
            if candidate in df.columns:
                datetime_col = candidate
                break
        if datetime_col:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            df = df.set_index(datetime_col)
        else:
            raise ValueError("CSV 文件缺少日期列，无法构建时序索引。")
    return df


def _load_from_yfinance(config: DataSourceConfig) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - 仅在未安装库时触发
        raise RuntimeError("请先安装 yfinance 才能从雅虎财经下载数据。") from exc

    ticker = yf.Ticker(config.symbol)
    df = ticker.history(
        start=config.start_date,
        end=config.end_date,
        interval=_frequency_to_interval(config.frequency),
        auto_adjust=config.adjust_prices,
    )
    if df.empty:
        raise ValueError(f"从 yfinance 未获取到 {config.symbol} 的历史数据，参数: {asdict(config)}")
    return df


def _load_from_fred(config: DataSourceConfig) -> pd.DataFrame:
    try:
        from pandas_datareader import data as web
    except ImportError as exc:  # pragma: no cover - 仅在未安装库时触发
        raise RuntimeError("请先安装 pandas-datareader 才能从 FRED 下载数据。") from exc

    series = web.DataReader(config.symbol, "fred", config.start_date, config.end_date)
    if series.empty:
        raise ValueError(f"FRED 未返回 {config.symbol} 的历史数据。")
    df = series.to_frame(name="Close")
    return df


def _normalise_dataframe(df: pd.DataFrame, config: DataSourceConfig) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("数据索引必须是 DatetimeIndex。")
    df = df.sort_index()
    rename_map = {
        "Adj Close": "close",
        "Close": "close",
        "close": "close",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Volume": "volume",
    }
    normalised_columns = {}
    for col in df.columns:
        if col in rename_map:
            normalised_columns[col] = rename_map[col]
    df = df.rename(columns=normalised_columns)
    if "close" not in df.columns:
        df["close"] = df.iloc[:, 0]
    # 简单补全缺失值，保留原始列名
    df = df.ffill()
    return df


def _frequency_to_interval(freq: str) -> str:
    mapping = {
        "D": "1d",
        "W": "1wk",
        "M": "1mo",
    }
    if freq not in mapping:
        raise ValueError(f"frequency '{freq}' 不受支持，仅支持 {tuple(mapping)}")
    return mapping[freq]


__all__ = ["load_gold_history"]

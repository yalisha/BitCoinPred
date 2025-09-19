from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CONFIG, FeatureConfig


def build_feature_frame(
    price_df: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """为黄金价格构建技术分析特征矩阵。"""

    cfg = config or CONFIG.features
    df = price_df.copy()
    features = pd.DataFrame(index=df.index)
    close = df["close"]

    if cfg.include_returns:
        features["return_pct"] = close.pct_change()
        features["return_log"] = np.log(close / close.shift(1))

    if cfg.include_volatility:
        features["rolling_vol"] = features["return_log"].rolling(cfg.lookback_window).std()

    if cfg.include_technical_indicators:
        for window in cfg.moving_average_windows:
            features[f"sma_{window}"] = close.rolling(window).mean()
            features[f"ema_{window}"] = close.ewm(span=window, adjust=False).mean()
            features[f"momentum_{window}"] = close / close.shift(window) - 1
        for window in cfg.oscillator_windows:
            features[f"rsi_{window}"] = _rsi(close, window)

    features["close"] = close
    features = features.dropna()
    return features


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


__all__ = ["build_feature_frame"]

"""
Price/return features computed purely from OHLCV — no look-ahead: every value
at row T is computable using only bars[0..T] (rolling windows look backward
only; see tests/test_leakage.py for the automated check).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACKS = [5, 10, 20, 50, 100, 200]


def add_price_features(bars: pd.DataFrame) -> pd.DataFrame:
    """bars: OHLCV_COLUMNS dataframe (date, open, high, low, close, adj_close, volume),
    sorted ascending by date, for a single symbol. Uses adj_close as "the" price
    series so splits/dividends don't create fake return spikes."""
    df = bars.copy().reset_index(drop=True)
    px = df["adj_close"]

    df["daily_return"] = px.pct_change()
    df["log_return"] = np.log(px / px.shift(1))
    df["gap_return"] = df["open"] / df["close"].shift(1) - 1
    df["intraday_return"] = df["close"] / df["open"] - 1
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    df["close_open_return"] = df["close"] / df["open"] - 1

    df["dollar_volume"] = df["close"] * df["volume"]
    df["volume_change"] = df["volume"].pct_change()
    df["dollar_volume_change"] = df["dollar_volume"].pct_change()

    # Rolling VWAP-ish proxy (true intraday VWAP needs tick data; this is a
    # typical-price rolling approximation, documented as such).
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap_20d"] = (typical_price * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()

    for window in LOOKBACKS:
        df[f"rolling_return_{window}d"] = px.pct_change(window)
        df[f"rolling_volatility_{window}d"] = df["daily_return"].rolling(window).std() * np.sqrt(252)

    # ATR% (True Range based, expressed as % of close) — also used by labels/.
    prev_close = df["close"].shift(1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(14).mean()
    df["atr_percent"] = df["atr_14"] / df["close"]

    df["avg_dollar_volume_20d"] = df["dollar_volume"].rolling(20).mean()
    df["avg_dollar_volume_60d"] = df["dollar_volume"].rolling(60).mean()

    return df

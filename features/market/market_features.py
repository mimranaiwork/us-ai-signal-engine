"""
US market-context features from the liquid ETF proxies for the major indices:

    SPY -> S&P 500          QQQ -> Nasdaq-100
    IWM -> Russell 2000     DIA -> Dow Jones Industrial Average

Also computes stock-vs-benchmark relative strength, since "a stock up 3% while
the market is flat" is a very different signal from "a stock up 3% because the
whole market is up 3%".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.technical.indicators import rsi

BENCHMARKS = ["SPY", "QQQ", "IWM", "DIA"]


def compute_benchmark_features(bars: pd.DataFrame) -> pd.DataFrame:
    """bars: OHLCV_COLUMNS for one benchmark ETF, sorted ascending by date."""
    df = bars[["date"]].copy()
    px = bars["adj_close"]

    df["return_1d"] = px.pct_change()
    df["rsi_14"] = rsi(px, 14)
    df["volatility_20d"] = df["return_1d"].rolling(20).std() * np.sqrt(252)

    sma50 = px.rolling(50).mean()
    sma200 = px.rolling(200).mean()
    trend = pd.Series("FLAT", index=df.index)
    trend = trend.mask((px > sma50) & (px > sma200), "UP")
    trend = trend.mask((px < sma50) & (px < sma200), "DOWN")
    df["trend"] = trend
    return df


def compute_all_benchmark_features(bars_by_benchmark: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {b: compute_benchmark_features(bars) for b, bars in bars_by_benchmark.items()}


def add_relative_strength(stock_df: pd.DataFrame, benchmark_features: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Joins one benchmark's return_1d/trend onto the stock frame (by date) and adds
    stock_return - benchmark_return relative-strength columns at several windows.
    `stock_df` must already have `daily_return` and `adj_close` (see price_features).
    """
    df = stock_df.merge(
        benchmark_features[["date", "return_1d", "trend"]].rename(
            columns={"return_1d": f"{prefix}_return_1d", "trend": f"{prefix}_trend"}
        ),
        on="date", how="left",
    )
    df[f"relative_strength_vs_{prefix}_1d"] = df["daily_return"] - df[f"{prefix}_return_1d"]

    stock_cum = (1 + df["daily_return"].fillna(0)).cumprod()
    bench_cum = (1 + df[f"{prefix}_return_1d"].fillna(0)).cumprod()
    for w in (5, 21, 63):
        df[f"relative_strength_vs_{prefix}_{w}d"] = (
            stock_cum / stock_cum.shift(w) - bench_cum / bench_cum.shift(w)
        )
    return df

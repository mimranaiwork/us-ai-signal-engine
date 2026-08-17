"""
Technical indicators: moving averages, momentum, volatility, volume.
All backward-looking (rolling/ewm windows only) — no future data at row T.
Expects the output of features.price.price_features.add_price_features
(needs `daily_return`, `dollar_volume`, `atr_percent`, `atr_14` already present)
and adj_close as the canonical price series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.technical.pivot_points import add_pivot_points

MA_WINDOWS = [5, 10, 20, 50, 100, 200]


# ── Moving averages ──────────────────────────────────────────────────────────

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    px = df["adj_close"]

    for w in MA_WINDOWS:
        df[f"sma_{w}"] = px.rolling(w).mean()
        df[f"ema_{w}"] = px.ewm(span=w, adjust=False).mean()

    df["price_vs_sma20"] = px / df["sma_20"] - 1
    df["price_vs_sma50"] = px / df["sma_50"] - 1
    df["price_vs_sma200"] = px / df["sma_200"] - 1

    df["sma20_slope"] = df["sma_20"].pct_change(5)
    df["sma50_slope"] = df["sma_50"].pct_change(10)
    df["sma200_slope"] = df["sma_200"].pct_change(20)

    df["ema_crossover_10_20"] = np.sign(df["ema_10"] - df["ema_20"])
    df["ema_crossover_50_200"] = np.sign(df["ema_50"] - df["ema_200"])  # "golden/death cross" sign
    return df


# ── Momentum ─────────────────────────────────────────────────────────────────

def rsi(px: pd.Series, window: int) -> pd.Series:
    delta = px.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_value = 100 - (100 / (1 + rs))
    return rsi_value.fillna(50)  # neutral when undefined (e.g. avg_loss == 0)


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    px = df["adj_close"]

    for w in (7, 14, 21):
        df[f"rsi_{w}"] = rsi(px, w)

    ema12 = px.ewm(span=12, adjust=False).mean()
    ema26 = px.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    df["roc_10"] = px.pct_change(10) * 100
    df["momentum_10"] = px - px.shift(10)

    low14 = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    denom = (high14 - low14).replace(0, np.nan)
    df["stochastic_k"] = ((df["close"] - low14) / denom * 100).fillna(50)
    df["stochastic_d"] = df["stochastic_k"].rolling(3).mean()

    df["williams_r"] = ((high14 - df["close"]) / denom * -100).fillna(-50)

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = typical_price.rolling(20).mean()
    mean_dev = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci_20"] = (typical_price - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))

    return df


def add_trend_strength(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """ADX (Average Directional Index) + its +DI/-DI components — spec
    section 24's stock-specific regime detection needs this (trend
    STRENGTH, distinct from RSI/MACD which measure direction). Standard
    Wilder formulation: directional movement -> Wilder-smoothed (EMA with
    alpha=1/window, matching how ATR/RSI are already smoothed in this
    module) -> +DI/-DI -> DX -> ADX (a second Wilder smoothing of DX)."""
    df = df.copy()
    prev_close = df["close"].shift(1)
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)

    up_move = df["high"] - prev_high
    down_move = prev_low - df["low"]
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

    smoothed_tr = true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    smoothed_plus_dm = plus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    plus_di = 100 * smoothed_plus_dm / smoothed_tr.replace(0, np.nan)
    minus_di = 100 * smoothed_minus_dm / smoothed_tr.replace(0, np.nan)
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum

    df["plus_di_14"] = plus_di.fillna(0)
    df["minus_di_14"] = minus_di.fillna(0)
    df["adx_14"] = dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean().fillna(0)
    return df


# ── Volatility ───────────────────────────────────────────────────────────────

def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    px = df["adj_close"]

    mid = df["sma_20"] if "sma_20" in df.columns else px.rolling(20).mean()
    std20 = px.rolling(20).std()
    df["bollinger_mid"] = mid
    df["bollinger_upper"] = mid + 2 * std20
    df["bollinger_lower"] = mid - 2 * std20
    df["bollinger_band_width"] = (df["bollinger_upper"] - df["bollinger_lower"]) / mid

    ret = df["daily_return"] if "daily_return" in df.columns else px.pct_change()
    for w in (20, 60, 120):
        df[f"volatility_{w}d"] = ret.rolling(w).std() * np.sqrt(252)

    vol20 = df["volatility_20d"]
    df["volatility_percentile"] = vol20.rolling(252, min_periods=60).apply(
        lambda x: (x < x.iloc[-1]).mean(), raw=False
    )
    return df


# ── Volume ───────────────────────────────────────────────────────────────────

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
    df["relative_volume"] = df["volume_ratio"]  # alias, common naming in the spec

    direction = np.sign(df["close"].diff()).fillna(0)
    df["obv"] = (direction * df["volume"]).cumsum()

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    money_flow = typical_price * df["volume"]
    tp_diff = typical_price.diff()
    positive_flow = money_flow.where(tp_diff > 0, 0.0).rolling(14).sum()
    negative_flow = money_flow.where(tp_diff < 0, 0.0).rolling(14).sum()
    mfr = positive_flow / negative_flow.replace(0, np.nan)
    df["mfi_14"] = (100 - (100 / (1 + mfr))).fillna(50)

    if "vwap_20d" not in df.columns:
        df["vwap_20d"] = (typical_price * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()

    return df


def add_all_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = add_moving_averages(df)
    df = add_momentum(df)
    df = add_volatility(df)
    df = add_volume_features(df)
    df = add_trend_strength(df)
    df = add_pivot_points(df)
    return df

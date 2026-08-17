from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from features.price.price_features import add_price_features
from features.technical.indicators import add_all_technical_indicators, add_trend_strength
from ingestion.ohlcv_source import SampleCSVSource


@pytest.fixture(scope="module")
def features_df():
    src = SampleCSVSource()
    bars = src.get_bars("AAPL", date(2018, 1, 1), date(2026, 8, 8))
    df = add_price_features(bars)
    df = add_all_technical_indicators(df)
    return df


def test_sma_matches_manual_rolling_mean(features_df):
    manual = features_df["adj_close"].rolling(20).mean()
    assert np.allclose(features_df["sma_20"].dropna(), manual.dropna())


def test_rsi_bounded_0_100(features_df):
    rsi = features_df["rsi_14"].dropna()
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_stochastic_bounded_0_100(features_df):
    k = features_df["stochastic_k"].dropna()
    assert (k >= 0).all() and (k <= 100).all()


def test_williams_r_bounded(features_df):
    wr = features_df["williams_r"].dropna()
    assert (wr >= -100).all() and (wr <= 0).all()


def test_mfi_bounded_0_100(features_df):
    mfi = features_df["mfi_14"].dropna()
    assert (mfi >= 0).all() and (mfi <= 100).all()


def test_atr_percent_non_negative(features_df):
    atr = features_df["atr_percent"].dropna()
    assert (atr >= 0).all()


def test_bollinger_upper_above_lower(features_df):
    valid = features_df.dropna(subset=["bollinger_upper", "bollinger_lower"])
    assert (valid["bollinger_upper"] >= valid["bollinger_lower"]).all()


def test_macd_histogram_equals_macd_minus_signal(features_df):
    valid = features_df.dropna(subset=["macd", "macd_signal", "macd_histogram"])
    assert np.allclose(valid["macd_histogram"], valid["macd"] - valid["macd_signal"])


def test_ema_windows_present(features_df):
    for w in (5, 10, 20, 50, 100, 200):
        assert f"sma_{w}" in features_df.columns
        assert f"ema_{w}" in features_df.columns


def test_adx_bounded_0_100(features_df):
    adx = features_df["adx_14"].dropna()
    assert (adx >= 0).all() and (adx <= 100).all()


def test_plus_minus_di_bounded_0_100(features_df):
    for col in ("plus_di_14", "minus_di_14"):
        valid = features_df[col].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()


def _synthetic_bars(prices: np.ndarray) -> pd.DataFrame:
    n = len(prices)
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "open": prices, "high": prices * 1.002, "low": prices * 0.998,
        "close": prices, "adj_close": prices,
        "volume": np.full(n, 1_000_000),
    })


def test_adx_higher_for_strong_trend_than_choppy_sideways():
    """ADX measures trend STRENGTH, not direction — a steadily trending
    series should show a materially higher ADX than a flat, noisy one over
    the same window count."""
    n = 100
    trending = _synthetic_bars(100 + np.arange(n) * 1.0)  # steady +1/day trend, no noise

    rng = np.random.default_rng(0)
    choppy = _synthetic_bars(100 + rng.normal(0, 0.5, n).cumsum() * 0 + rng.normal(0, 1.5, n))  # pure noise, no drift

    trending_adx = add_trend_strength(trending)["adx_14"].iloc[-1]
    choppy_adx = add_trend_strength(choppy)["adx_14"].iloc[-1]

    assert trending_adx > choppy_adx
    assert trending_adx > 40  # strong, clean trend should read as a high ADX


def test_adx_plus_di_dominant_in_uptrend():
    n = 60
    uptrend = _synthetic_bars(100 + np.arange(n) * 1.0)
    result = add_trend_strength(uptrend)
    assert result["plus_di_14"].iloc[-1] > result["minus_di_14"].iloc[-1]

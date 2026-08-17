import numpy as np
import pandas as pd
import pytest

from features.macro.macro_features import compute_macro_series_features, merge_macro_features, vix_regime


def _synthetic_series_bars(n=60, level=20.0, seed=0):
    rng = np.random.default_rng(seed)
    levels = level + rng.normal(0, 1, n).cumsum()
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "date": dates, "open": levels, "high": levels, "low": levels,
        "close": levels, "adj_close": levels, "volume": np.zeros(n),
    })


def test_compute_macro_series_features_shape_and_naming():
    bars = _synthetic_series_bars()
    result = compute_macro_series_features(bars, "VIX")
    for suffix in ("level", "change_1d", "change_1w", "change_1m", "volatility_20d"):
        assert f"vix_{suffix}" in result.columns
    assert len(result) == len(bars)


def test_compute_macro_series_features_uses_series_name_prefix():
    bars = _synthetic_series_bars()
    result = compute_macro_series_features(bars, "GOLD")
    assert "gold_level" in result.columns
    assert "vix_level" not in result.columns


def test_merge_macro_features_joins_on_date():
    vix_bars = _synthetic_series_bars(level=20)
    gold_bars = _synthetic_series_bars(level=180, seed=1)
    macro_frames = {
        "VIX": compute_macro_series_features(vix_bars, "VIX"),
        "GOLD": compute_macro_series_features(gold_bars, "GOLD"),
    }
    stock_df = pd.DataFrame({"date": vix_bars["date"], "daily_return": np.zeros(len(vix_bars))})
    merged = merge_macro_features(stock_df, macro_frames)
    assert "vix_level" in merged.columns
    assert "gold_level" in merged.columns
    assert len(merged) == len(stock_df)


def test_merge_macro_features_handles_misaligned_dates():
    """A macro series (e.g. Bitcoin) with extra/missing dates vs the stock's
    NYSE trading calendar shouldn't break the merge or duplicate rows."""
    vix_bars = _synthetic_series_bars(n=60)
    btc_bars = _synthetic_series_bars(n=80, seed=2)  # different date range (weekends included)
    btc_bars["date"] = pd.date_range("2024-01-01", periods=80, freq="D")  # calendar days, not business days

    macro_frames = {"BITCOIN": compute_macro_series_features(btc_bars, "BITCOIN")}
    stock_df = pd.DataFrame({"date": vix_bars["date"]})
    merged = merge_macro_features(stock_df, macro_frames)
    assert len(merged) == len(stock_df)  # left join preserves the stock's own trading-session rows


def test_vix_regime_bands():
    assert vix_regime(10) == "LOW"
    assert vix_regime(20) == "NORMAL"
    assert vix_regime(30) == "ELEVATED"
    assert vix_regime(50) == "EXTREME"
    assert vix_regime(None) is None
    assert vix_regime(float("nan")) is None

import pandas as pd
import pytest

from signals.stock_regime import detect_stock_regime


def _row(**kwargs):
    base = {"adx_14": 10.0, "price_vs_sma200": 0.0, "roc_10": 0.0, "volatility_20d": 0.20}
    base.update(kwargs)
    return pd.Series(base)


def test_high_volatility_overrides_everything():
    row = _row(adx_14=40, price_vs_sma200=0.1, roc_10=5, volatility_20d=0.60)
    result = detect_stock_regime(row)
    assert result.regime == "HIGH_VOLATILITY"


def test_low_volatility_when_not_trending_and_calm():
    row = _row(adx_14=15, volatility_20d=0.10)
    result = detect_stock_regime(row)
    assert result.regime == "LOW_VOLATILITY"


def test_sideways_when_not_trending_and_not_especially_calm():
    row = _row(adx_14=15, volatility_20d=0.25)
    result = detect_stock_regime(row)
    assert result.regime == "SIDEWAYS"


def test_bullish_when_trending_up():
    row = _row(adx_14=35, price_vs_sma200=0.08, roc_10=3.0, volatility_20d=0.20)
    result = detect_stock_regime(row)
    assert result.regime == "BULLISH"
    assert result.confidence == pytest.approx(1.0)  # both signals agree


def test_bearish_when_trending_down():
    row = _row(adx_14=35, price_vs_sma200=-0.08, roc_10=-3.0, volatility_20d=0.20)
    result = detect_stock_regime(row)
    assert result.regime == "BEARISH"


def test_market_and_sector_trend_included_in_vote():
    row = _row(adx_14=35, price_vs_sma200=0.05, roc_10=1.0, volatility_20d=0.20)
    bullish = detect_stock_regime(row, market_trend="UP", sector_trend="UP")
    assert bullish.regime == "BULLISH"
    assert bullish.confidence == pytest.approx(1.0)  # all 4 signals agree

    # Now with a dissenting market trend, confidence should drop but regime unchanged.
    mixed = detect_stock_regime(row, market_trend="DOWN", sector_trend="UP")
    assert mixed.regime == "BULLISH"  # still 3-vs-1 in favor
    assert mixed.confidence < bullish.confidence


def test_missing_data_degrades_to_sideways_not_a_crash():
    row = pd.Series({"adx_14": None, "price_vs_sma200": None, "roc_10": None, "volatility_20d": None})
    result = detect_stock_regime(row)
    assert result.regime == "SIDEWAYS"
    assert result.confidence == 0.5


def test_trending_but_evenly_split_votes_is_sideways():
    row = _row(adx_14=35, price_vs_sma200=0.05, roc_10=-2.0, volatility_20d=0.20)  # conflicting signals
    result = detect_stock_regime(row, market_trend="UP", sector_trend="DOWN")
    assert result.regime == "SIDEWAYS"

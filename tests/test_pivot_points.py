import pandas as pd
import pytest

from features.technical.pivot_points import BREAKOUT_PROXIMITY_PCT, add_pivot_points, breakout_assessment


def _bars(high, low, close, adj_close=None):
    """Two rows: a 'prior session' (H/L/C given) and a 'today' row whose
    adj_close is what we're testing breakout_assessment against. Pivot
    levels are derived FROM the prior row, so only row 1 (today) has them."""
    adj_close = adj_close if adj_close is not None else close
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=2),
        "open": [close, close], "high": [high, high], "low": [low, low],
        "close": [close, close], "adj_close": [close, adj_close], "volume": [1_000_000, 1_000_000],
    })


def test_pivot_formula_matches_classic_definition():
    df = _bars(high=110, low=90, close=100)
    result = add_pivot_points(df)
    today = result.iloc[1]

    pivot = (110 + 90 + 100) / 3
    assert today["pivot"] == pytest.approx(pivot)
    assert today["resistance_1"] == pytest.approx(2 * pivot - 90)
    assert today["support_1"] == pytest.approx(2 * pivot - 110)
    assert today["resistance_2"] == pytest.approx(pivot + (110 - 90))
    assert today["support_2"] == pytest.approx(pivot - (110 - 90))
    assert today["resistance_3"] == pytest.approx(110 + 2 * (pivot - 90))
    assert today["support_3"] == pytest.approx(90 - 2 * (110 - pivot))


def test_pivot_points_are_nan_on_first_row_no_leakage():
    df = _bars(high=110, low=90, close=100)
    result = add_pivot_points(df)
    assert pd.isna(result.iloc[0]["pivot"])  # no prior session for row 0


def test_breakout_assessment_unavailable_without_history():
    df = _bars(high=110, low=90, close=100)
    result = add_pivot_points(df)
    assessment = breakout_assessment(result.iloc[0])  # first row, no pivot levels yet
    assert assessment["available"] is False


def test_breakout_assessment_near_breakout_true_just_below_resistance():
    df = _bars(high=110, low=90, close=100)
    result = add_pivot_points(df)
    pivot = (110 + 90 + 100) / 3
    r1 = 2 * pivot - 90

    df2 = _bars(high=110, low=90, close=100, adj_close=r1 * (1 - BREAKOUT_PROXIMITY_PCT / 2))
    result2 = add_pivot_points(df2)
    assessment = breakout_assessment(result2.iloc[1])

    assert assessment["available"] is True
    assert assessment["near_breakout"] is True
    assert assessment["nearest_resistance"] == pytest.approx(round(r1, 2))
    assert assessment["target"] == pytest.approx(round(r1, 2))
    assert assessment["expected_profit_pct"] > 0


def test_breakout_assessment_not_near_when_far_from_resistance():
    df = _bars(high=110, low=90, close=100, adj_close=95)  # well below R1
    result = add_pivot_points(df)
    assessment = breakout_assessment(result.iloc[1])
    assert assessment["available"] is True
    assert assessment["near_breakout"] is False


def test_breakout_assessment_reports_all_three_levels_each_side():
    df = _bars(high=110, low=90, close=100)
    result = add_pivot_points(df)
    assessment = breakout_assessment(result.iloc[1])
    assert set(assessment["resistance"]) == {"r1", "r2", "r3"}
    assert set(assessment["support"]) == {"s1", "s2", "s3"}
    assert assessment["resistance"]["r1"] < assessment["resistance"]["r2"] < assessment["resistance"]["r3"]
    assert assessment["support"]["s1"] > assessment["support"]["s2"] > assessment["support"]["s3"]


def test_breakout_assessment_handles_price_above_all_resistance_levels():
    df = _bars(high=110, low=90, close=100, adj_close=500)  # blew past R3 already
    result = add_pivot_points(df)
    assessment = breakout_assessment(result.iloc[1])
    assert assessment["nearest_resistance"] is None
    assert assessment["target"] is None
    assert assessment["expected_profit_pct"] is None
    assert assessment["near_breakout"] is False

import os

import pandas as pd
import pytest

from features.market.market_breadth import compute_market_breadth, latest_breadth_snapshot


@pytest.fixture
def processed_dir(tmp_path):
    """Three synthetic symbols (>=20 rows so the new-high/low rolling window's
    min_periods=20 is actually satisfied) with known advance/decline/SMA/
    new-high outcomes on the last shared date, so the aggregation math is
    checkable by hand."""
    n = 20
    dates = pd.date_range("2024-01-01", periods=n)

    # Symbol A: flat for 19 days then jumps to a new high on the last day.
    a_close = [100.0] * (n - 1) + [105.0]
    a_return = [0.0] * (n - 1) + [0.05]
    a = pd.DataFrame({
        "date": dates, "adj_close": a_close, "daily_return": a_return,
        "price_vs_sma50": [0.01] * n, "price_vs_sma200": [0.02] * n,
    })
    # Symbol B: flat for 19 days then drops to a new low on the last day.
    b_close = [50.0] * (n - 1) + [45.0]
    b_return = [0.0] * (n - 1) + [-0.10]
    b = pd.DataFrame({
        "date": dates, "adj_close": b_close, "daily_return": b_return,
        "price_vs_sma50": [-0.01] * n, "price_vs_sma200": [-0.02] * n,
    })
    # Symbol C: peaks early then settles lower — by the last day it's neither
    # at a new high nor a new low (a perfectly flat series would trivially
    # register as "at its own high" every day, which isn't what this test
    # wants to isolate).
    c_close = [20.0, 22.0] + [21.0] * (n - 2)
    c = pd.DataFrame({
        "date": dates, "adj_close": c_close, "daily_return": [0.0] * n,
        "price_vs_sma50": [0.005] * n, "price_vs_sma200": [-0.005] * n,
    })

    for symbol, df in [("A", a), ("B", b), ("C", c)]:
        df.to_csv(os.path.join(tmp_path, f"{symbol}_features.csv"), index=False)
    return str(tmp_path)


def test_advancers_decliners_counted_correctly(processed_dir):
    breadth = compute_market_breadth(["A", "B", "C"], processed_dir=processed_dir)
    last_day = breadth.iloc[-1]
    assert last_day["advancers"] == 1  # only A
    assert last_day["decliners"] == 1  # only B
    assert last_day["n_symbols"] == 3


def test_pct_above_sma_correct(processed_dir):
    breadth = compute_market_breadth(["A", "B", "C"], processed_dir=processed_dir)
    last_day = breadth.iloc[-1]
    # Above SMA50: A and C -> 2/3. Above SMA200: only A -> 1/3.
    assert last_day["pct_above_sma50"] == pytest.approx(2 / 3)
    assert last_day["pct_above_sma200"] == pytest.approx(1 / 3)


def test_new_highs_and_lows_detected(processed_dir):
    breadth = compute_market_breadth(["A", "B", "C"], processed_dir=processed_dir)
    last_day = breadth.iloc[-1]
    assert last_day["new_highs"] == 1  # A
    assert last_day["new_lows"] == 1   # B
    assert last_day["net_new_highs_lows"] == 0  # 1 - 1


def test_advance_decline_ratio(processed_dir):
    breadth = compute_market_breadth(["A", "B", "C"], processed_dir=processed_dir)
    last_day = breadth.iloc[-1]
    assert last_day["advance_decline_ratio"] == pytest.approx(1.0)  # 1 advancer / 1 decliner


def test_missing_symbol_files_are_skipped_not_fatal(processed_dir):
    breadth = compute_market_breadth(["A", "B", "NOT_A_REAL_SYMBOL"], processed_dir=processed_dir)
    assert breadth.iloc[-1]["n_symbols"] == 2


def test_raises_when_no_symbols_have_data(tmp_path):
    with pytest.raises(RuntimeError):
        compute_market_breadth(["NOPE1", "NOPE2"], processed_dir=str(tmp_path))


def test_latest_breadth_snapshot_shape(processed_dir):
    breadth = compute_market_breadth(["A", "B", "C"], processed_dir=processed_dir)
    snapshot = latest_breadth_snapshot(breadth)
    assert "market_breadth_score" in snapshot
    assert "advance_decline_ratio" in snapshot
    assert -1.5 <= snapshot["market_breadth_score"] <= 1.5

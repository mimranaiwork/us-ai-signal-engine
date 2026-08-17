from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from backtesting.costs import TransactionCostModel
from backtesting.engine import buy_and_hold_curve, run_backtest, simulate_long_only
from backtesting.metrics import classification_metrics, trading_metrics
from backtesting.walk_forward import generate_folds


def _synthetic_price_series(n=100, start_price=100.0, seed=0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.01, n)
    prices = start_price * (1 + returns).cumprod()
    dates = pd.Series([date(2024, 1, 1) + timedelta(days=i) for i in range(n)])
    return dates, pd.Series(prices)


def test_always_long_from_day_zero_costs_less_than_buy_and_hold():
    """Regression test: entering a position on day 0 must incur a transaction
    cost, so an always-long strategy (BUY signal every day, position from the
    very first bar) should trail raw buy & hold by the entry cost — they must
    NOT be identical. This is the exact bug caught during Phase 4 development:
    the first trade's cost was silently skipped by an off-by-one in the
    equity-accumulation loop, making the two curves match to float precision."""
    dates, prices = _synthetic_price_series()
    signals = pd.Series(["BUY"] * len(prices))
    cost_model = TransactionCostModel(commission_per_share=0, commission_minimum=0, spread_bps=5, slippage_bps=5)

    equity, positions, trades = simulate_long_only(dates, prices, signals, cost_model)
    bh = buy_and_hold_curve(dates, prices)

    assert (positions == 1).all()
    assert len(trades) == 1  # one entry, no exit
    assert equity.iloc[-1] < bh.iloc[-1]
    assert equity.iloc[-1] == pytest.approx(bh.iloc[-1] * (1 - cost_model.entry_or_exit_cost_pct()))


def test_always_flat_never_trades_and_stays_at_initial_capital():
    dates, prices = _synthetic_price_series()
    signals = pd.Series(["HOLD"] * len(prices))  # never opens a position
    equity, positions, trades = simulate_long_only(dates, prices, signals)
    assert (positions == 0).all()
    assert len(trades) == 0
    assert equity.iloc[-1] == pytest.approx(equity.iloc[0])


def test_hold_maintains_current_position_without_new_trade():
    dates, prices = _synthetic_price_series(n=10)
    signals = pd.Series(["BUY", "HOLD", "HOLD", "HOLD", "SELL", "HOLD", "BUY", "HOLD", "SELL", "HOLD"])
    equity, positions, trades = simulate_long_only(dates, prices, signals)
    # Position changes only at the BUY/SELL indices — 4 trades total (2 entries, 2 exits).
    assert len(trades) == 4
    assert list(positions) == [1, 1, 1, 1, 0, 0, 1, 1, 0, 0]


def test_two_round_trips_cost_more_than_one():
    dates, prices = _synthetic_price_series(n=50)
    cost_model = TransactionCostModel(commission_per_share=0, commission_minimum=0, spread_bps=10, slippage_bps=10)

    one_trip = pd.Series(["BUY"] * 25 + ["SELL"] * 25)
    two_trips = pd.Series(["BUY"] * 10 + ["SELL"] * 5 + ["BUY"] * 10 + ["SELL"] * 25)

    equity_one, _, trades_one = simulate_long_only(dates, prices, one_trip, cost_model)
    equity_two, _, trades_two = simulate_long_only(dates, prices, two_trips, cost_model)

    assert len(trades_two) > len(trades_one)


def test_run_backtest_returns_all_comparison_curves():
    dates, prices = _synthetic_price_series(n=60)
    _, spy_prices = _synthetic_price_series(n=60, seed=1)
    signals = pd.Series(["BUY"] * 30 + ["SELL"] * 30)

    result = run_backtest(dates, prices, signals, {"SPY": spy_prices})
    assert len(result.equity_curve) == len(dates)
    assert "SPY" in result.benchmark_equity_curves
    assert len(result.buy_and_hold_equity_curve) == len(dates)


def test_classification_metrics_shapes():
    y_true = np.array([0, 1, 2, 1, 0, 2, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 2, 2])
    y_proba = np.array([
        [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.2, 0.6, 0.2], [0.1, 0.7, 0.2],
        [0.7, 0.2, 0.1], [0.1, 0.1, 0.8], [0.2, 0.3, 0.5],
    ])
    metrics = classification_metrics(y_true, y_pred, y_proba)
    assert 0 <= metrics["accuracy"] <= 1
    assert len(metrics["confusion_matrix"]) == 3
    assert metrics["roc_auc_macro"] is None or 0 <= metrics["roc_auc_macro"] <= 1


def test_trading_metrics_sane_on_known_curve():
    # Equity doubles smoothly over a year -> total_return should be ~1.0.
    equity = pd.Series(np.linspace(100_000, 200_000, 253),
                        index=pd.date_range("2024-01-01", periods=253, freq="B"))
    metrics = trading_metrics(equity)
    assert metrics["total_return"] == pytest.approx(1.0, abs=0.01)
    assert metrics["max_drawdown"] == pytest.approx(0.0, abs=1e-6)  # monotonically increasing -> no drawdown


def test_trading_metrics_detects_drawdown():
    values = [100, 110, 90, 95, 120]
    equity = pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="D"))
    metrics = trading_metrics(equity)
    # Peak 110 -> trough 90 => drawdown of -18.18%
    assert metrics["max_drawdown"] == pytest.approx((90 - 110) / 110, abs=1e-6)


def test_generate_folds_expanding_window():
    folds = generate_folds(date(2018, 1, 1), date(2026, 8, 8), min_train_years=3, val_years=1, test_years=1)
    assert len(folds) >= 3
    # Train window always starts at data_start and grows across folds.
    assert all(f.train_start == date(2018, 1, 1) for f in folds)
    train_ends = [f.train_end for f in folds]
    assert train_ends == sorted(train_ends)
    for f in folds:
        assert f.train_end < f.val_end < f.test_end
        assert f.test_end <= date(2026, 8, 8)


def test_generate_folds_respects_data_end():
    folds = generate_folds(date(2020, 1, 1), date(2021, 6, 1), min_train_years=3)
    assert folds == []  # not even enough data for one fold

"""AI-signal walk-forward backtest: purging, timing lag, costs, idle cash,
scoring, and signal-quality diagnostics. Synthetic data only."""

import numpy as np
import pandas as pd

from backtesting.ai_signal_backtest import (
    composite, decile_spread, purged_splits, rank_ic, simulate, target_weights, weekly_rebalance_days,
)


def test_purging_never_trains_on_labels_that_see_the_traded_year():
    dates = pd.bdate_range("2018-01-01", "2021-12-31")
    frame = pd.DataFrame({"date": dates, "label": "BUY",
                          "target_session_date": dates + pd.Timedelta(days=60)})
    train, val = purged_splits(frame, 2021)
    assert train["target_session_date"].max() < pd.Timestamp("2020-01-01")
    assert val["target_session_date"].max() < pd.Timestamp("2021-01-01")
    assert val["date"].min() >= pd.Timestamp("2020-01-01")
    assert not train.index.intersection(val.index).size
    # rows from late 2019 whose 60-day label reaches 2020 are in neither set
    late_2019 = frame[(frame["date"] > "2019-11-15") & (frame["date"] < "2020-01-01")]
    assert not late_2019.index.isin(train.index).any() and not late_2019.index.isin(val.index).any()


def test_composite_matches_weighted_mean():
    s = {"1M": pd.Series([1.0, -1.0]), "6M": pd.Series([0.0, 1.0])}
    out = composite(s, {"1M": 0.25, "6M": 0.75})
    assert np.allclose(out.values, [0.25, 0.5])


def _prices(n=60, cols=("A", "B", "C")):
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({c: 100 * (1.01 ** np.arange(n)) * (1 + 0.1 * i) for i, c in enumerate(cols)}, index=idx)


def test_top_n_respects_threshold_and_equal_weights():
    idx = pd.bdate_range("2024-01-01", periods=5)
    sc = pd.DataFrame({"A": 0.5, "B": 0.05, "C": 0.3}, index=idx)
    w = target_weights(sc, top_n=10, threshold=0.10, rebalance_days=idx[[0]])
    assert w.loc[idx[0], "A"] == 0.1 and w.loc[idx[0], "C"] == 0.1 and w.loc[idx[0], "B"] == 0.0


def test_one_day_lag_no_lookahead():
    """A signal on day t must not earn the t->t+1 move (traded at t+1's close)."""
    idx = pd.bdate_range("2024-01-01", periods=6)
    px = pd.DataFrame({"A": [100, 100, 200, 200, 200, 200.0]}, index=idx)   # jump on day 2
    w = pd.DataFrame({"A": [1.0]}, index=[idx[1]])                            # signal at day-1 close
    res = simulate(w, px, idx[0], idx[-1], cost_per_side=0.0)
    assert res.equity.iloc[-1] == 100_000                                     # traded at day-2 close: missed the jump
    w_early = pd.DataFrame({"A": [1.0]}, index=[idx[0]])                      # signal one day earlier
    res2 = simulate(w_early, px, idx[0], idx[-1], cost_per_side=0.0)
    assert abs(res2.equity.iloc[-1] - 200_000) < 1e-6


def test_costs_reduce_returns_and_idle_cash_follows_etf():
    px = _prices()
    w = pd.DataFrame({"A": [0.5], "B": [0.0], "C": [0.0]}, index=[px.index[0]])
    free = simulate(w, px, px.index[0], px.index[-1], cost_per_side=0.0)
    paid = simulate(w, px, px.index[0], px.index[-1], cost_per_side=0.001)
    assert paid.equity.iloc[-1] < free.equity.iloc[-1]
    etf = pd.Series(100 * (1.02 ** np.arange(len(px))), index=px.index)
    with_etf = simulate(w, px, px.index[0], px.index[-1], cost_per_side=0.0, idle_prices=etf)
    assert with_etf.equity.iloc[-1] > free.equity.iloc[-1]
    assert abs(free.avg_invested - 0.5) < 0.05


def test_weekly_rebalance_days_are_fridays_for_full_weeks():
    idx = pd.bdate_range("2024-01-01", "2024-02-02")
    reb = weekly_rebalance_days(idx)
    assert all(d.weekday() == 4 for d in reb)


def test_signal_quality_detects_real_skill_and_its_absence():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2023-01-02", periods=150)
    cols = [f"S{i}" for i in range(40)]
    drift = np.linspace(-0.004, 0.004, 40)                       # S39 rises fastest
    px = pd.DataFrame(100 * np.exp(np.cumsum(drift + rng.normal(0, 0.01, (150, 40)), axis=0)), index=idx, columns=cols)
    skilled = pd.DataFrame(np.tile(drift, (150, 1)), index=idx, columns=cols)
    noise = pd.DataFrame(rng.normal(size=(150, 40)), index=idx, columns=cols)
    assert rank_ic(skilled, px).mean() > 0.3 and decile_spread(skilled, px) > 0
    assert abs(rank_ic(noise, px).mean()) < 0.1

"""
Backtesting engine (spec section 44): simulates a long-only strategy driven
by a sequence of BUY/HOLD/SELL signals, net of transaction costs, and
compares the resulting equity curve against Buy & Hold and against SPY/QQQ
over the same window.

Deliberately decoupled from *how* predictions are made — it takes a plain
(date, price, signal) series, so it works identically whether the signals
came from a single model, the full ensemble (signals/signal_engine.py), or
a walk-forward fold's freshly retrained model (backtesting/walk_forward.py).

Position rule: BUY -> enter/stay long. SELL -> exit/stay flat. HOLD -> no
change (don't whipsaw the position on every noisy HOLD day; per the spec's
own conservative-HOLD philosophy, HOLD means "don't act," not "flatten").
No shorting — going short is a materially different risk profile (borrow
costs, unlimited loss potential) that the spec doesn't ask for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtesting.costs import TransactionCostModel


@dataclass
class BacktestResult:
    equity_curve: pd.Series          # indexed by date
    positions: pd.Series             # 0/1, indexed by date
    trades: pd.DataFrame             # one row per position change
    benchmark_equity_curves: dict[str, pd.Series]
    buy_and_hold_equity_curve: pd.Series


def _positions_from_signals(signals: pd.Series) -> pd.Series:
    positions = pd.Series(0, index=signals.index, dtype=int)
    current = 0
    for i, sig in enumerate(signals):
        if sig == "BUY":
            current = 1
        elif sig == "SELL":
            current = 0
        # HOLD: current unchanged
        positions.iloc[i] = current
    return positions


def simulate_long_only(
    dates: pd.Series, prices: pd.Series, signals: pd.Series,
    cost_model: TransactionCostModel = TransactionCostModel(),
    initial_capital: float = 100_000.0,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    df = pd.DataFrame({"date": dates.values, "price": prices.values, "signal": signals.values}).reset_index(drop=True)
    df["position"] = _positions_from_signals(df["signal"])
    df["daily_return"] = df["price"].pct_change().fillna(0.0)

    # position at t determines exposure to the return realized FROM t-1 TO t,
    # i.e. use yesterday's position for today's return (can't trade on a
    # signal before the bar that produced it closes).
    df["exposed_return"] = df["daily_return"] * df["position"].shift(1).fillna(0)

    # Day 0 is a trade too if the very first signal opens a position — track
    # it separately since there's no prior day to diff() against.
    trade_days = set(df.index[df["position"].diff().fillna(0) != 0])
    per_trade_cost_pct = cost_model.entry_or_exit_cost_pct()

    equity = [initial_capital]
    trades = []
    if df.loc[0, "position"] == 1:
        equity[0] = equity[0] * (1 - per_trade_cost_pct)
        trades.append({"date": df.loc[0, "date"], "action": "BUY", "price": df.loc[0, "price"]})

    for i in range(1, len(df)):
        value = equity[-1] * (1 + df.loc[i, "exposed_return"])
        if i in trade_days:
            value *= (1 - per_trade_cost_pct)
            trades.append({
                "date": df.loc[i, "date"], "action": "BUY" if df.loc[i, "position"] == 1 else "SELL",
                "price": df.loc[i, "price"],
            })
        equity.append(value)

    equity_curve = pd.Series(equity, index=pd.to_datetime(df["date"]))
    positions = pd.Series(df["position"].values, index=pd.to_datetime(df["date"]))
    trades_df = pd.DataFrame(trades)
    return equity_curve, positions, trades_df


def buy_and_hold_curve(dates: pd.Series, prices: pd.Series, initial_capital: float = 100_000.0) -> pd.Series:
    returns = prices.pct_change().fillna(0.0)
    equity = initial_capital * (1 + returns).cumprod()
    equity.index = pd.to_datetime(dates.values)
    return equity


def run_backtest(
    dates: pd.Series, prices: pd.Series, signals: pd.Series,
    benchmark_prices: dict[str, pd.Series],
    cost_model: TransactionCostModel = TransactionCostModel(),
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    equity_curve, positions, trades = simulate_long_only(dates, prices, signals, cost_model, initial_capital)
    bh_curve = buy_and_hold_curve(dates, prices, initial_capital)
    benchmark_curves = {
        name: buy_and_hold_curve(dates, bench_prices, initial_capital)
        for name, bench_prices in benchmark_prices.items()
    }
    return BacktestResult(equity_curve, positions, trades, benchmark_curves, bh_curve)

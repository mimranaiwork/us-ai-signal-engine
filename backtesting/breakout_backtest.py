"""
Portfolio backtest of the execution layer's breakout strategy.

Reuses execution/breakout.py's own functions (classify, position_size,
manage, market_filter_ok) so the backtest trades the exact rules the live
bot trades, not a re-implementation of them.

Timing mirrors the live bot (scripts/run_trading_bot.py), which runs once
a day during market hours using the last COMPLETED session:
  close of day t   -> signals computed (scan + manage)
  open of day t+1  -> exits from manage(), then new entries, filled at the open
                      (+ slippage); a GTC stop is placed right after each fill
  during day t+1   -> any position whose low touches its stop is sold at the
                      stop, or at the open if it gapped below the stop

Costs: backtesting/costs.TransactionCostModel spread + slippage on every
fill (default 3 bps per side); commission 0 by default (Alpaca charges none).

Not modelled (each one documented in the report):
  - earnings guard (no point-in-time earnings calendar for the whole history)
  - ensemble AI filter (would need model predictions for every past day)
  - the 3% daily-loss brake (rarely binds with 1% risk per trade)
  - survivorship bias: the universe is TODAY's Shariah list applied to the
    past — names that dropped out (often after falling) are missing, which
    flatters results. Treat absolute returns as optimistic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtesting.costs import TransactionCostModel
from backtesting.metrics import trading_metrics
from execution import breakout


@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    entry_px: float
    qty: int
    risk_per_share: float
    exits: list = field(default_factory=list)   # (date, qty, px, reason)

    @property
    def closed(self) -> bool:
        return sum(q for _, q, _, _ in self.exits) >= self.qty

    def pnl(self) -> float:
        return sum(q * (px - self.entry_px) for _, q, px, _ in self.exits)

    def r_multiple(self) -> float:
        risk = self.risk_per_share * self.qty
        return self.pnl() / risk if risk > 0 else 0.0


def prepare(frames: dict[str, pd.DataFrame], cfg: dict) -> dict[str, pd.DataFrame]:
    """Indicators computed once over full history. Every level used for a
    decision on row t only depends on rows <= t (breakout.add_breakout_indicators
    shifts its levels), so slicing later never leaks future data."""
    return {s: breakout.add_breakout_indicators(df.sort_index(), cfg) for s, df in frames.items() if len(df) > 60}


def run(frames: dict[str, pd.DataFrame], universe: list[str], cfg: dict, start, end,
        initial_capital: float = 100_000.0, benchmark: str = "SPUS", fallback_benchmark: str = "SPY",
        use_market_filter: bool = True, costs: TransactionCostModel | None = None,
        idle_cash_in: str | None = None) -> dict:
    """idle_cash_in: if set (e.g. "SPUS"), cash not used by open trades is held
    in that ETF: it earns the ETF's close-to-close return, and each trade pays
    one extra side of costs for selling/buying the ETF to fund it. Approximates
    a core (ETF) + satellite (breakouts) portfolio."""
    costs = costs or TransactionCostModel(commission_per_share=0.0, commission_minimum=0.0)
    side_cost = costs.entry_or_exit_cost_pct()
    data = prepare(frames, cfg)
    start, end = pd.Timestamp(start), pd.Timestamp(end)

    def norm(idx):
        return idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx
    for s in data:
        data[s].index = norm(data[s].index).normalize()

    bench_key = benchmark if benchmark in data else fallback_benchmark
    calendar = data[bench_key].index if bench_key in data else sorted(set().union(*[d.index for d in data.values()]))
    calendar = [d for d in calendar if start <= d <= end]
    pos_lookup = {s: {d: i for i, d in enumerate(df.index)} for s, df in data.items()}

    cash = initial_capital
    open_trades: dict[str, Trade] = {}
    state: dict[str, dict] = {}          # breakout.manage() position dicts
    closed: list[Trade] = []
    pending_exits: list[tuple[str, int, str]] = []
    pending_entries: list[tuple[str, dict]] = []
    equity_rows = []
    filter_days_on = 0

    def row(sym, d):
        i = pos_lookup[sym].get(d)
        return None if i is None else data[sym].iloc[i]

    def sell(sym, qty, px, d, reason):
        nonlocal cash
        tr = open_trades[sym]
        fill = px * (1 - side_cost)
        cash += qty * fill * ((1 - side_cost) if idle_cash_in else 1)   # proceeds go back into the ETF
        tr.exits.append((d, qty, fill, reason))
        state_pos = state.get(sym)
        if tr.closed:
            closed.append(open_trades.pop(sym))
            state.pop(sym, None)
        return state_pos

    for d in calendar:
        # ── open: queued exits, then queued entries ──
        for sym, qty, reason in pending_exits:
            if sym in open_trades:
                r = row(sym, d)
                if r is not None:
                    sell(sym, min(qty, open_trades[sym].qty - sum(q for _, q, _, _ in open_trades[sym].exits)),
                         float(r["open"]), d, reason)
        pending_exits = []

        equity_open = cash + sum(
            (t.qty - sum(q for _, q, _, _ in t.exits)) * float((row(s, d) if row(s, d) is not None else data[s].loc[:d].iloc[-1])["open"])
            for s, t in open_trades.items())
        for sym, info in pending_entries:
            if len(open_trades) >= cfg["max_positions"] or sym in open_trades:
                continue
            r = row(sym, d)
            if r is None:
                continue
            q = breakout.position_size(info["close"], info["stop"], equity_open, cash, cfg)
            fill = float(r["open"]) * (1 + side_cost)
            q = min(q, math.floor(cash * 0.98 / fill)) if fill > 0 else 0
            if q < 1:
                continue
            stop = min(info["stop"], round(fill * 0.99, 2))
            if stop >= fill:
                continue
            cash -= q * fill * ((1 + side_cost) if idle_cash_in else 1)   # sell ETF to fund the buy
            open_trades[sym] = Trade(sym, d, fill, q, fill - stop)
            state[sym] = {"entry": fill, "stop": stop, "risk": round(fill - stop, 4),
                          "breakout_level": info["breakout_level"], "date": str(d.date()),
                          "high_close": fill, "partial": False, "earnings_cut": False}
        pending_entries = []

        # ── intraday: broker stop orders ──
        for sym in list(open_trades):
            r = row(sym, d)
            if r is None:
                continue
            stop = state[sym]["stop"]
            if float(r["low"]) <= stop:
                px = min(float(r["open"]), stop)
                remaining = open_trades[sym].qty - sum(q for _, q, _, _ in open_trades[sym].exits)
                sell(sym, remaining, px, d, "stop hit" if px == stop else "stop (gap down)")

        # ── close: idle cash earns the ETF's return, mark to market, manage, scan ──
        if idle_cash_in and idle_cash_in in pos_lookup:
            ii = pos_lookup[idle_cash_in].get(d)
            if ii is not None and ii > 0 and cash > 0 and d != calendar[0]:   # no return before day one
                c = data[idle_cash_in]["close"]
                cash *= float(c.iloc[ii]) / float(c.iloc[ii - 1])
        mtm = 0.0
        for sym, t in open_trades.items():
            hist = data[sym].loc[:d]
            mtm += (t.qty - sum(q for _, q, _, _ in t.exits)) * float(hist["close"].iloc[-1])
        equity = cash + mtm
        equity_rows.append((d, equity, len(open_trades)))

        for sym in list(open_trades):
            i = pos_lookup[sym].get(d)
            if i is None:
                continue
            hist = data[sym].iloc[: i + 1]
            remaining = open_trades[sym].qty - sum(q for _, q, _, _ in open_trades[sym].exits)
            actions, newpos = breakout.manage(state[sym], hist, remaining, cfg)
            state[sym] = newpos
            pending_exits += [(sym, q, why) for _, q, why in actions]

        regime_ok = True
        if use_market_filter:
            bi = pos_lookup.get(bench_key, {}).get(d)
            regime_ok = bi is not None and breakout.market_filter_ok(data[bench_key].iloc[: bi + 1])
        filter_days_on += regime_ok
        slots = cfg["max_positions"] - len(open_trades)
        if regime_ok and slots > 0:
            found = []
            for sym in universe:
                i = pos_lookup.get(sym, {}).get(d)
                if i is None or sym in open_trades:
                    continue
                x = data[sym].iloc[i]
                if not (x["close"] > x["breakout_level"]):   # cheap pre-check; classify() requires this
                    continue
                info = breakout.classify(data[sym].iloc[: i + 1], cfg)
                if info and info["status"] == "breakout":
                    found.append((sym, info))
            found.sort(key=lambda t: -t[1]["score"])
            pending_entries = found[:slots]

    # close anything still open at the last close (mark-to-market exit, no cost)
    last = calendar[-1] if calendar else end
    for sym in list(open_trades):
        px = float(data[sym].loc[:last]["close"].iloc[-1])
        t = open_trades.pop(sym)
        rem = t.qty - sum(q for _, q, _, _ in t.exits)
        t.exits.append((last, rem, px, "open at end of test"))
        closed.append(t)

    eq = pd.DataFrame(equity_rows, columns=["date", "equity", "positions"]).set_index("date")
    trade_r = pd.Series([t.r_multiple() for t in closed], dtype=float)
    trade_ret = pd.Series([t.pnl() / (t.entry_px * t.qty) for t in closed], dtype=float)
    metrics = trading_metrics(eq["equity"], trade_ret) if len(eq) > 1 else {}
    bench = None
    if bench_key in data and len(eq) > 1:
        bc = data[bench_key]["close"].loc[eq.index[0]:eq.index[-1]]
        bench = trading_metrics(bc / bc.iloc[0] * initial_capital)
        bench["symbol"] = bench_key

    wins, losses = trade_r[trade_r > 0], trade_r[trade_r <= 0]
    summary = {
        "trades": int(len(closed)),
        "win_rate": float(len(wins) / len(trade_r)) if len(trade_r) else 0.0,
        "avg_win_r": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_r": float(losses.mean()) if len(losses) else 0.0,
        "expectancy_r": float(trade_r.mean()) if len(trade_r) else 0.0,
        "avg_hold_sessions": float(np.mean([
            len(data[t.symbol].loc[t.entry_date:t.exits[-1][0]]) - 1 for t in closed])) if closed else 0.0,
        "time_in_market": float((eq["positions"] > 0).mean()) if len(eq) else 0.0,
        "avg_positions": float(eq["positions"].mean()) if len(eq) else 0.0,
        "market_filter_on_pct": filter_days_on / len(calendar) if calendar else 0.0,
        "exit_reasons": pd.Series([t.exits[-1][3].split(":")[0] for t in closed]).value_counts().to_dict() if closed else {},
    }
    return {"metrics": metrics, "benchmark": bench, "summary": summary, "equity": eq,
            "trades": closed, "start": str(calendar[0].date()) if calendar else None,
            "end": str(calendar[-1].date()) if calendar else None}


def yearly_returns(eq: pd.Series) -> pd.Series:
    y = eq.groupby(eq.index.year).agg(["first", "last"])
    return (y["last"] / y["first"] - 1)


def trades_frame(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([{
        "symbol": t.symbol, "entry_date": t.entry_date.date(), "entry_px": round(t.entry_px, 2), "qty": t.qty,
        "exit_date": t.exits[-1][0].date(), "avg_exit_px": round(sum(q * p for _, q, p, _ in t.exits) / t.qty, 2),
        "pnl": round(t.pnl(), 2), "r": round(t.r_multiple(), 2), "exit_reason": t.exits[-1][3],
    } for t in trades])

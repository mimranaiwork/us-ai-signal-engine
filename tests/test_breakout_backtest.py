"""Breakout portfolio backtest: timing, stops, costs, no look-ahead.
Synthetic prices only — no network."""

import numpy as np
import pandas as pd

from backtesting.breakout_backtest import run, trades_frame
from execution import breakout

CFG = dict(breakout.DEFAULTS)
UP = list(np.linspace(50, 100, 220))
BASE = [100 + 3 * np.sin(i / 2) for i in range(30)]


def frame(closes, vols=None, opens=None, lows=None, start="2024-01-01"):
    c = np.array(closes, float)
    idx = pd.bdate_range(start, periods=len(c))
    v = np.full(len(c), 1e6) if vols is None else np.array(vols, float)
    o = c.copy() if opens is None else np.array(opens, float)
    lo = c * 0.995 if lows is None else np.array(lows, float)
    return pd.DataFrame({"open": o, "high": np.maximum(c, o) * 1.005, "low": np.minimum(lo, o),
                         "close": c, "volume": v}, idx)


def breakout_then(after, gap_open=None):
    """Uptrend, tight base, a volume breakout to 105.5, then `after` closes."""
    closes = UP + BASE + [105.5] + after
    vols = [1e6] * (len(UP) + len(BASE)) + [2.5e6] + [1e6] * len(after)
    df = frame(closes, vols)
    df.iloc[len(UP) + len(BASE), df.columns.get_loc("low")] = 104.6   # close in the upper half of the day
    if gap_open is not None:
        df.iloc[len(UP) + len(BASE) + 1, df.columns.get_loc("open")] = gap_open
    return df


def bench(n, start="2024-01-01"):
    return frame(list(np.linspace(40, 60, n)), start=start)


def _run(df, **kw):
    frames = {"AAA": df, "SPUS": bench(len(df))}
    sig_day = df.index[len(UP) + len(BASE)]
    return run(frames, ["AAA"], CFG, df.index[230], df.index[-1], **kw), sig_day


def test_entry_fills_next_open_not_signal_close():
    df = breakout_then([106, 107, 108], gap_open=106.8)
    res, sig_day = _run(df)
    t = res["trades"][0]
    assert t.entry_date == df.index[df.index.get_loc(sig_day) + 1]
    assert abs(t.entry_px - 106.8 * 1.0003) < 1e-6          # next open + 3 bps cost


def test_stop_hit_intraday_sells_at_stop():
    df = breakout_then([106, 103, 101, 100])
    df.iloc[-2, df.columns.get_loc("low")] = 95.0                # day's low pierces the stop
    res, _ = _run(df)
    t = res["trades"][0]
    reason = t.exits[-1][3]
    assert "stop" in reason or "failed" in reason
    assert t.r_multiple() < 0


def test_gap_below_stop_fills_at_open():
    df = breakout_then([106, 106, 106])
    i = len(df) - 2
    df.iloc[i, df.columns.get_loc("open")] = 90.0
    df.iloc[i, df.columns.get_loc("low")] = 89.0
    res, _ = _run(df)
    t = res["trades"][0]
    assert t.exits[-1][3] == "stop (gap down)"
    assert abs(t.exits[-1][2] - 90.0 * (1 - 0.0003)) < 1e-6
    assert t.r_multiple() < -1                                   # gaps can lose more than 1R


def test_winner_takes_partial_and_trails():
    df = breakout_then(list(np.linspace(107, 140, 25)))
    res, _ = _run(df)
    t = res["trades"][0]
    assert any("partial" in e[3] for e in t.exits)
    assert t.r_multiple() > 2


def test_no_trades_when_market_filter_off():
    df = breakout_then([106, 107])
    falling = frame(list(np.linspace(60, 40, len(df))))
    res = run({"AAA": df, "SPUS": falling}, ["AAA"], CFG, df.index[230], df.index[-1])
    assert res["summary"]["trades"] == 0
    res2 = run({"AAA": df, "SPUS": falling}, ["AAA"], CFG, df.index[230], df.index[-1], use_market_filter=False)
    assert res2["summary"]["trades"] == 1


def test_future_prices_do_not_change_past_decisions():
    """Look-ahead check: appending wildly different future data must not change the first trade."""
    a = breakout_then([106, 107, 108, 109, 110])
    b = pd.concat([a, frame([300] * 10, start=str((a.index[-1] + pd.offsets.BDay()).date()))])
    ra, _ = _run(a)
    rb = run({"AAA": b, "SPUS": bench(len(b))}, ["AAA"], CFG, a.index[230], a.index[-1])
    assert ra["trades"][0].entry_date == rb["trades"][0].entry_date
    assert abs(ra["trades"][0].entry_px - rb["trades"][0].entry_px) < 1e-9


def test_max_positions_respected():
    frames = {f"S{i}": breakout_then([106, 107, 108]) for i in range(8)}
    n = len(next(iter(frames.values())))
    frames["SPUS"] = bench(n)
    idx = frames["S0"].index
    res = run(frames, [f"S{i}" for i in range(8)], CFG, idx[230], idx[-1])
    assert res["equity"]["positions"].max() <= CFG["max_positions"]
    assert res["summary"]["trades"] == CFG["max_positions"]


def test_equity_accounts_for_every_dollar():
    df = breakout_then(list(np.linspace(107, 120, 10)))
    res, _ = _run(df, initial_capital=50_000)
    pnl = sum(t.pnl() for t in res["trades"])
    entry_costs = sum(t.qty * t.entry_px * (1 - 1 / 1.0003) for t in res["trades"])
    assert abs(res["equity"]["equity"].iloc[-1] - (50_000 + pnl)) < entry_costs + 1.0
    assert not trades_frame(res["trades"]).empty

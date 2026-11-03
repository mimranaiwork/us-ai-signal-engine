"""
Breakout scan + trade management — pure functions, no I/O, no broker.

Same honesty class as features/technical/pivot_points.py and
signals/stock_regime.py: these are deterministic, documented rules with
round, commonly-used thresholds (50-day high breakout, 1.5x volume, 2R
partial, 3xATR trail). None of them are walk-forward-validated by this
project yet — backtesting/engine.py could be pointed at them, but hasn't
been. Every threshold lives in DEFAULTS and can be overridden from
config/settings.yaml's `trading:` section.

Why a separate breakout layer instead of trading the ensemble's BUY signal
directly: the ensemble answers "which direction over 1D..6M", it doesn't
give an entry trigger, a stop, or a position size. This module supplies
those; the ensemble is used as an optional confirmation/veto in
execution/ai_filter.py.
"""

from __future__ import annotations

import math
from datetime import date

import pandas as pd

DEFAULTS = {
    # scan
    "min_price": 10.0,
    "pivot_lookback": 50,          # breakout level = highest high of the prior 50 sessions
    "base_days": 20,
    "max_base_depth": 0.15,        # prior 20 sessions' high-to-low range must be <= 15%
    "near_pct": 0.03,              # watchlist: within 3% below the breakout level
    "breakout_vol_mult": 1.5,
    "max_extension": 0.05,         # skip if already > 5% above the breakout level
    "max_stop_pct": 0.10,          # skip if the stop would be > 10% below entry
    "atr_mult_stop": 2.0,
    # sizing (long only, cash only)
    "risk_per_trade": 0.01,
    "max_positions": 5,
    "max_position_pct": 0.20,
    "max_daily_loss_pct": 0.03,
    # exits
    "partial_r": 2.0,
    "partial_fraction": 0.5,
    "trail_atr": 3.0,
    "fail_days": 5,
    "fail_below_pivot": 0.03,
    "time_stop_days": 15,
    "time_stop_min_r": 0.5,
    "earnings_cut_fraction": 0.5,  # sell this share of an open position before earnings (0 disables)
}

MIN_HISTORY = 210  # SMA200 + a little slack


def add_breakout_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """df: daily OHLCV indexed by date, oldest first. Every derived level is
    shifted so row T only uses data through T-1 for the breakout level and
    base — the same no-leakage convention as pivot_points.add_pivot_points."""
    df = df.copy()
    h, l, c = df["high"], df["low"], df["close"]
    df["sma50"] = c.rolling(50).mean()
    df["sma200"] = c.rolling(200).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["vol50"] = df["volume"].rolling(50).mean().shift(1)
    df["breakout_level"] = h.rolling(cfg["pivot_lookback"]).max().shift(1)
    bd = cfg["base_days"]
    df["base_high"] = h.rolling(bd).max().shift(1)
    df["base_low"] = l.rolling(bd).min().shift(1)
    df["base_depth"] = (df["base_high"] - df["base_low"]) / df["base_high"]
    return df


def classify(df: pd.DataFrame, cfg: dict) -> dict | None:
    """Latest completed session -> {'status': 'breakout' | 'near', ...} or None."""
    if len(df) < MIN_HISTORY:
        return None
    x = df.iloc[-1]
    if any(pd.isna(x[k]) for k in ("sma200", "breakout_level", "vol50", "atr", "base_depth")):
        return None
    close, level = float(x["close"]), float(x["breakout_level"])
    if close < cfg["min_price"] or not (close > x["sma50"] > x["sma200"]):
        return None
    if x["base_depth"] > cfg["max_base_depth"]:
        return None

    vol_ratio = float(x["volume"] / x["vol50"]) if x["vol50"] else 0.0
    rng = float(x["high"] - x["low"])
    close_pos = (close - float(x["low"])) / rng if rng > 0 else 1.0
    info = {
        "close": close, "breakout_level": level, "vol_ratio": vol_ratio,
        "base_depth": float(x["base_depth"]), "atr": float(x["atr"]),
        "base_low": float(x["base_low"]), "dist_pct": (close / level - 1) * 100,
    }

    if close > level:
        if (close <= level * (1 + cfg["max_extension"]) and vol_ratio >= cfg["breakout_vol_mult"]
                and close_pos >= 0.5):
            stop = max(info["base_low"] * 0.99, close - cfg["atr_mult_stop"] * info["atr"])
            if stop >= close or (close - stop) / close > cfg["max_stop_pct"]:
                return None
            info.update(status="breakout", stop=round(stop, 2),
                        score=vol_ratio / max(info["base_depth"], 0.02))
            return info
        return None
    if close >= level * (1 - cfg["near_pct"]):
        info.update(status="near")
        return info
    return None


def position_size(entry: float, stop: float, equity: float, cash: float, cfg: dict) -> int:
    """Whole shares risking cfg['risk_per_trade'] of equity, capped by
    max_position_pct and by available CASH (never margin — a deliberate
    Shariah constraint, not just a risk one)."""
    risk_share = entry - stop
    if risk_share <= 0 or entry <= 0:
        return 0
    q = math.floor(equity * cfg["risk_per_trade"] / risk_share)
    q = min(q, math.floor(equity * cfg["max_position_pct"] / entry))
    q = min(q, math.floor(max(cash, 0) * 0.98 / entry))
    return max(q, 0)


def sessions_held(df: pd.DataFrame, entry_date: str) -> int:
    d = pd.Timestamp(entry_date).date()
    idx = df.index.tz_convert("America/New_York") if df.index.tz is not None else df.index
    return int(sum(1 for t in idx if t.date() > d))


def manage(pos: dict, df: pd.DataFrame, qty: int, cfg: dict,
           days_to_earnings: int | None = None, earnings_window: int = 5) -> tuple[list, dict]:
    """Exit rules for one open trade. Returns ([(side, qty, reason)], updated_pos).
    `pos` keys: entry, stop, risk, breakout_level, date, high_close, partial, earnings_cut."""
    pos = dict(pos)
    x = df.iloc[-1]
    close, atr = float(x["close"]), float(x["atr"])
    entry, risk = pos["entry"], pos["risk"]
    r = (close - entry) / risk if risk > 0 else 0.0
    held = sessions_held(df, pos["date"])
    pos["high_close"] = max(pos.get("high_close", entry), close)

    if held <= cfg["fail_days"] and close < pos["breakout_level"] * (1 - cfg["fail_below_pivot"]):
        return [("sell", qty, f"failed breakout: close {close:.2f} back under {pos['breakout_level']:.2f}")], pos
    if held >= cfg["time_stop_days"] and r < cfg["time_stop_min_r"]:
        return [("sell", qty, f"time stop: {held} sessions, only {r:.1f}R")], pos

    actions, remaining = [], qty
    if (cfg["earnings_cut_fraction"] > 0 and not pos.get("earnings_cut") and days_to_earnings is not None
            and 0 <= days_to_earnings <= earnings_window and remaining >= 2):
        q = max(1, math.floor(remaining * cfg["earnings_cut_fraction"]))
        actions.append(("sell", q, f"earnings in ~{days_to_earnings} days: reducing risk"))
        pos["earnings_cut"] = True
        remaining -= q
    # At +partial_r: optionally sell a slice, and in every case move the stop to
    # break-even and start trailing. partial_fraction 0 = keep the full position.
    if not pos.get("partial") and r >= cfg["partial_r"]:
        if cfg["partial_fraction"] > 0 and remaining >= 2:
            q = max(1, math.floor(remaining * cfg["partial_fraction"]))
            actions.append(("sell", q, f"+{r:.1f}R: partial profit, stop to break-even"))
        pos["partial"] = True
        pos["stop"] = max(pos["stop"], round(entry, 2))
    if pos.get("partial"):
        pos["stop"] = max(pos["stop"], round(pos["high_close"] - cfg["trail_atr"] * atr, 2))
    return actions, pos


def scan(frames: dict[str, pd.DataFrame], symbols: list[str], cfg: dict) -> tuple[list, list]:
    """frames: symbol -> indicator-enriched df. Returns (breakouts, near), best first."""
    breakouts, near = [], []
    for sym in symbols:
        if sym not in frames:
            continue
        info = classify(frames[sym], cfg)
        if info is None:
            continue
        (breakouts if info["status"] == "breakout" else near).append((sym, info))
    breakouts.sort(key=lambda t: -t[1]["score"])
    near.sort(key=lambda t: -t[1]["dist_pct"])
    return breakouts, near


def market_filter_ok(df: pd.DataFrame | None) -> bool:
    """New entries only while the benchmark closes above its 50-day average."""
    if df is None or len(df) < 50 or pd.isna(df["sma50"].iloc[-1]):
        return False
    return float(df["close"].iloc[-1]) > float(df["sma50"].iloc[-1])


def today_str(d: date) -> str:
    return d.strftime("%Y-%m-%d")

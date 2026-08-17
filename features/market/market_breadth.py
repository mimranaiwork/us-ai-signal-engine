"""
US market breadth (spec section 14) — Advance/Decline, new highs/lows,
%-above-SMA50/200, and a composite market_breadth_score. Unlike every other
feature module, this one is inherently cross-sectional: it aggregates ACROSS
the whole universe's already-built per-symbol features
(data/processed/{symbol}_features.csv) rather than computing something from
one symbol's own history, so it lives under features/market/ (US market
context) rather than features/technical/.

52-week new-highs/lows uses a 252-session rolling window — the standard
"52-week" convention in trading-session terms, consistent with the rest of
this project's trading-session-not-calendar-day philosophy (see
calendar_us/trading_calendar.py).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from models.common import PROCESSED_DIR

NEW_HIGH_LOW_WINDOW = 252  # ~52 weeks of trading sessions


def _load_symbol_breadth_inputs(symbol: str, processed_dir: str) -> pd.DataFrame | None:
    path = os.path.join(processed_dir, f"{symbol}_features.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, usecols=["date", "adj_close", "daily_return", "price_vs_sma50", "price_vs_sma200"],
                      parse_dates=["date"])
    df["symbol"] = symbol
    df["rolling_high"] = df["adj_close"].rolling(NEW_HIGH_LOW_WINDOW, min_periods=20).max()
    df["rolling_low"] = df["adj_close"].rolling(NEW_HIGH_LOW_WINDOW, min_periods=20).min()
    df["is_new_high"] = df["adj_close"] >= df["rolling_high"]
    df["is_new_low"] = df["adj_close"] <= df["rolling_low"]
    return df


def compute_market_breadth(symbols: list[str], processed_dir: str = PROCESSED_DIR) -> pd.DataFrame:
    """Returns a DataFrame indexed by date with one row per trading session:
    advancers, decliners, advance_decline_ratio, pct_above_sma50,
    pct_above_sma200, new_highs, new_lows, net_new_highs_lows,
    market_breadth_score."""
    frames = [f for f in (_load_symbol_breadth_inputs(s, processed_dir) for s in symbols) if f is not None]
    if not frames:
        raise RuntimeError("No processed features found for any symbol — run scripts/build_phase1_dataset.py first.")

    long_df = pd.concat(frames, ignore_index=True)

    def _daily_breadth(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "advancers": (g["daily_return"] > 0).sum(),
            "decliners": (g["daily_return"] < 0).sum(),
            "n_symbols": len(g),
            "pct_above_sma50": (g["price_vs_sma50"] > 0).mean(),
            "pct_above_sma200": (g["price_vs_sma200"] > 0).mean(),
            "new_highs": g["is_new_high"].sum(),
            "new_lows": g["is_new_low"].sum(),
        })

    breadth = long_df.groupby("date").apply(_daily_breadth, include_groups=False).sort_index()

    breadth["advance_decline_ratio"] = breadth["advancers"] / breadth["decliners"].replace(0, np.nan)
    breadth["net_advances"] = breadth["advancers"] - breadth["decliners"]
    breadth["net_new_highs_lows"] = breadth["new_highs"] - breadth["new_lows"]

    # Composite score in roughly [-1, 1]: average of three normalized signals
    # — net advance/decline share, %-above-SMA50, %-above-SMA200 (each
    # rescaled from [0,1]-or-[-1,1] onto a common [-1,1] scale) — plus net
    # new-highs-vs-lows share. Simple average, in the same spirit as
    # signals/scoring.py's composite_score — not claimed to be optimal,
    # walk-forward validation (Phase 4 harness) could tune the weights.
    net_advance_share = breadth["net_advances"] / breadth["n_symbols"].replace(0, np.nan)
    sma50_signal = 2 * breadth["pct_above_sma50"] - 1
    sma200_signal = 2 * breadth["pct_above_sma200"] - 1
    new_hl_share = breadth["net_new_highs_lows"] / breadth["n_symbols"].replace(0, np.nan)

    breadth["market_breadth_score"] = pd.concat(
        [net_advance_share, sma50_signal, sma200_signal, new_hl_share], axis=1
    ).mean(axis=1)

    return breadth


def latest_breadth_snapshot(breadth: pd.DataFrame) -> dict:
    if breadth.empty:
        return {}
    row = breadth.iloc[-1]
    return {k: (None if pd.isna(v) else float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in row.to_dict().items()}

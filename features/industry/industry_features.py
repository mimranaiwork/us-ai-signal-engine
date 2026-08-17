"""
Industry features (spec section 16) — Semiconductors, Banks, Biotechnology,
etc. Unlike sectors (real, liquid SPDR ETFs exist — XLK, XLF, ...), there is
no free, individually-investable index for most of these sub-industries.
This computes an equal-weighted PEER-GROUP proxy instead: the average daily
return of every OTHER universe stock sharing the same industry_code, as an
"industry index" stand-in. Documented limitation: with a 40-symbol universe,
many industries have only 1-3 members (see data/universe/stocks.csv), so the
peer group can be thin or, for a single-member industry, nonexistent — those
rows correctly come back as NaN (a small/no peer group isn't a meaningful
"industry return", so there's nothing better to substitute than admitting
it's unavailable) rather than silently reducing to "compare a stock to
itself." A real deployment would replace this with a proper industry index
(e.g. a licensed industry-classification data provider) behind the same
`stock_vs_industry` interface.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from models.common import PROCESSED_DIR
from universe.universe_loader import Universe

INDUSTRY_MOMENTUM_WINDOW = 21  # ~1M, matching sector_momentum's window


def _load_symbol_returns(symbol: str, processed_dir: str) -> pd.DataFrame | None:
    path = os.path.join(processed_dir, f"{symbol}_features.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, usecols=["date", "daily_return"], parse_dates=["date"]).assign(symbol=symbol)


def compute_industry_peer_returns(universe: Universe | None = None, processed_dir: str = PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    """
    Returns {industry_code: DataFrame(date, industry_return, industry_momentum,
    industry_volatility, n_peers)} — one equal-weighted peer-group "index"
    per industry that has at least 2 universe members with processed data.
    """
    universe = universe or Universe()
    by_industry: dict[str, list[str]] = {}
    for symbol in universe.all_symbols(active_only=True):
        stock = universe.get(symbol)
        by_industry.setdefault(stock.industry_code, []).append(symbol)

    result = {}
    for industry_code, symbols in by_industry.items():
        frames = [f for f in (_load_symbol_returns(s, processed_dir) for s in symbols) if f is not None]
        if len(frames) < 2:
            continue  # not enough peers for a meaningful "industry return" — see module docstring
        long_df = pd.concat(frames, ignore_index=True)
        grouped = long_df.groupby("date")["daily_return"].agg(["mean", "std", "count"])
        grouped.columns = ["industry_return", "industry_volatility_raw", "n_peers"]
        grouped["industry_volatility"] = grouped["industry_volatility_raw"] * np.sqrt(252)
        grouped["industry_momentum"] = (1 + grouped["industry_return"]).cumprod().pct_change(INDUSTRY_MOMENTUM_WINDOW)
        result[industry_code] = grouped.drop(columns=["industry_volatility_raw"]).reset_index()

    return result


def add_stock_vs_industry(stock_df: pd.DataFrame, industry_returns: pd.DataFrame | None) -> pd.DataFrame:
    """industry_returns: one entry from compute_industry_peer_returns()'s
    result dict for this stock's OWN industry_code, or None if that industry
    had too few peers (all industry_* columns come back NaN, not fabricated)."""
    df = stock_df.copy()
    if industry_returns is None:
        for col in ("industry_return", "industry_volatility", "industry_momentum", "n_peers", "stock_vs_industry_1d"):
            df[col] = np.nan
        return df

    df = df.merge(industry_returns, on="date", how="left")
    df["stock_vs_industry_1d"] = df["daily_return"] - df["industry_return"]
    return df

"""
Macro features (spec section 17) — the market-based half only: VIX, DXY,
10Y Treasury yield, Gold, Oil, Copper, Bitcoin, via scripts/fetch_macro_data.py.
CPI/PPI/unemployment/GDP/ISM/Consumer Confidence are NOT here — see that
script's module docstring for why (point-in-time correctness for real
economic releases needs FRED, not implemented).

No leakage concern for these specific series: they're daily market prices/
yields, same-day available with no revision lag (unlike CPI/GDP), so simple
date-alignment is sufficient — no as_of/point-in-time machinery needed here,
unlike features/fundamentals/ or features/earnings/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MACRO_SERIES = ["VIX", "US10Y", "DXY", "GOLD", "OIL", "COPPER", "BITCOIN"]


def compute_macro_series_features(bars: pd.DataFrame, name: str) -> pd.DataFrame:
    """bars: OHLCV_COLUMNS for one macro series (e.g. VIX). Returns date +
    level/return/change features, prefixed with the series name lowercased."""
    prefix = name.lower()
    px = bars["adj_close"]
    df = pd.DataFrame({"date": bars["date"]})
    df[f"{prefix}_level"] = px
    df[f"{prefix}_change_1d"] = px.pct_change()
    df[f"{prefix}_change_1w"] = px.pct_change(5)
    df[f"{prefix}_change_1m"] = px.pct_change(21)
    df[f"{prefix}_volatility_20d"] = px.pct_change().rolling(20).std() * np.sqrt(252)
    return df


def merge_macro_features(stock_df: pd.DataFrame, macro_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """stock_df: a symbol's feature dataframe (needs `date`).
    macro_frames: {"VIX": compute_macro_series_features(vix_bars, "VIX"), ...}
    — same macro data merged onto every symbol (macro isn't stock-specific)."""
    df = stock_df
    for name, macro_df in macro_frames.items():
        df = df.merge(macro_df, on="date", how="left")
    return df


def vix_regime(vix_level: float | None) -> str | None:
    """Rough, commonly-used VIX bands — not backtested/validated thresholds,
    same honesty caveat as signals/market_regime.py's HIGH_VOLATILITY_THRESHOLD."""
    if vix_level is None or pd.isna(vix_level):
        return None
    if vix_level < 15:
        return "LOW"
    if vix_level < 25:
        return "NORMAL"
    if vix_level < 35:
        return "ELEVATED"
    return "EXTREME"

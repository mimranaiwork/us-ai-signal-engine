"""
Stock-specific regime detection (spec section 24) — distinct from
signals/market_regime.py, which classifies the whole MARKET (SPY/QQQ/IWM).
This classifies one STOCK using ADX (trend strength), price vs. SMA200
(trend direction context), momentum, and its own volatility — the exact
inputs section 24 lists, minus "sector trend" which isn't wired in until
sector features exist (see features/sector/ — Phase 2+ TODO); pass it in via
sector_trend once available and it'll be used the same way market_trend is.

Priority order, matching signals/market_regime.py's own HIGH_VOLATILITY-first
convention: elevated volatility overrides any directional read (a choppy,
violently-moving stock isn't reliably "bullish" or "bearish" even if it's
nominally up over the window), then a weak/no trend (ADX < 25, the standard
Wilder threshold for "trending") is LOW_VOLATILITY or SIDEWAYS depending on
how calm the stock actually is, then a strong trend (ADX >= 25) is BULLISH
or BEARISH depending on direction (price vs SMA200 + momentum + market/sector
trend agreement).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ADX_TRENDING_THRESHOLD = 25.0
HIGH_VOLATILITY_THRESHOLD = 0.45
LOW_VOLATILITY_THRESHOLD = 0.15


@dataclass
class StockRegimeResult:
    regime: str  # BULLISH | BEARISH | SIDEWAYS | HIGH_VOLATILITY | LOW_VOLATILITY
    confidence: float  # how many of the available directional signals agree, in [0, 1]
    detail: dict


def detect_stock_regime(
    latest_row: pd.Series, market_trend: str | None = None, sector_trend: str | None = None,
) -> StockRegimeResult:
    adx = latest_row.get("adx_14")
    price_vs_sma200 = latest_row.get("price_vs_sma200")
    momentum = latest_row.get("roc_10")
    volatility = latest_row.get("volatility_20d")

    detail = {
        "adx_14": None if pd.isna(adx) else float(adx),
        "price_vs_sma200": None if pd.isna(price_vs_sma200) else float(price_vs_sma200),
        "momentum_roc_10": None if pd.isna(momentum) else float(momentum),
        "volatility_20d": None if pd.isna(volatility) else float(volatility),
        "market_trend": market_trend,
        "sector_trend": sector_trend,
    }

    if volatility is not None and not pd.isna(volatility) and volatility >= HIGH_VOLATILITY_THRESHOLD:
        return StockRegimeResult("HIGH_VOLATILITY", 1.0, detail)

    trending = adx is not None and not pd.isna(adx) and adx >= ADX_TRENDING_THRESHOLD
    if not trending:
        if volatility is not None and not pd.isna(volatility) and volatility <= LOW_VOLATILITY_THRESHOLD:
            return StockRegimeResult("LOW_VOLATILITY", 1.0, detail)
        return StockRegimeResult("SIDEWAYS", 0.5, detail)

    # Trending (ADX >= 25) — direction comes from a vote across whichever of
    # price-vs-SMA200 / momentum / market trend / sector trend are available.
    votes = []
    if price_vs_sma200 is not None and not pd.isna(price_vs_sma200):
        votes.append(1 if price_vs_sma200 > 0 else -1)
    if momentum is not None and not pd.isna(momentum):
        votes.append(1 if momentum > 0 else -1)
    if market_trend in ("UP", "DOWN"):
        votes.append(1 if market_trend == "UP" else -1)
    if sector_trend in ("UP", "DOWN"):
        votes.append(1 if sector_trend == "UP" else -1)

    if not votes:
        return StockRegimeResult("SIDEWAYS", 0.0, detail)

    net = sum(votes)
    agreement = abs(net) / len(votes)
    if net > 0:
        return StockRegimeResult("BULLISH", agreement, detail)
    if net < 0:
        return StockRegimeResult("BEARISH", agreement, detail)
    return StockRegimeResult("SIDEWAYS", 0.0, detail)  # trending but votes split evenly

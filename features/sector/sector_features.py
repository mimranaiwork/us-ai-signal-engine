"""
Sector features (spec section 15) — real data, using the SPDR sector ETFs
already fetched by scripts/fetch_real_data.py (XLK, XLF, XLE, XLV, XLI, XLY,
XLP, XLU, XLB, XLRE, XLC — see data/universe/sectors.csv for the
sector_code -> ETF mapping). Reuses features/market/market_features.py's
compute_benchmark_features() for the base return/RSI/volatility/trend
computation — a sector ETF and a benchmark ETF (SPY/QQQ/...) are the same
kind of thing (a liquid, tradable proxy for a basket), just at a different
level of the market.
"""

from __future__ import annotations

import pandas as pd

from features.market.market_features import compute_benchmark_features


def compute_sector_features(sector_bars: pd.DataFrame) -> pd.DataFrame:
    """sector_bars: OHLCV_COLUMNS for one sector ETF (e.g. XLK)."""
    df = compute_benchmark_features(sector_bars)  # date, return_1d, rsi_14, volatility_20d, trend
    px = sector_bars["adj_close"].reset_index(drop=True)

    df["sector_return_1d"] = df["return_1d"]
    df["sector_return_1w"] = px.pct_change(5)
    df["sector_return_1m"] = px.pct_change(21)
    df["sector_momentum"] = px.pct_change(21)  # ROC-21, the same window as "1M" — sector-level trend strength
    return df


def add_stock_vs_sector(stock_df: pd.DataFrame, sector_features: pd.DataFrame) -> pd.DataFrame:
    """
    stock_df: a symbol's feature dataframe (needs `date`, `daily_return`).
    sector_features: compute_sector_features() output for that symbol's OWN
    sector ETF (looked up via universe.sector_etf(stock.sector_code)).
    Adds sector_return_*, sector_momentum, sector_trend, and stock_vs_sector_*
    (the spec's core "is this stock outperforming its sector" signal).
    """
    df = stock_df.merge(
        sector_features[["date", "sector_return_1d", "sector_return_1w", "sector_return_1m",
                          "sector_momentum", "trend"]].rename(columns={"trend": "sector_trend"}),
        on="date", how="left",
    )
    df["stock_vs_sector_1d"] = df["daily_return"] - df["sector_return_1d"]

    stock_cum = (1 + df["daily_return"].fillna(0)).cumprod()
    sector_cum = (1 + df["sector_return_1d"].fillna(0)).cumprod()
    for w in (5, 21):
        df[f"stock_vs_sector_{w}d"] = stock_cum / stock_cum.shift(w) - sector_cum / sector_cum.shift(w)

    return df

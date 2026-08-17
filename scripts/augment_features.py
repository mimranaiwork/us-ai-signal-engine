"""
Adds sector/industry/macro columns (spec sections 15-17) to the
already-built data/processed/{symbol}_features.csv files, in place. Run
AFTER scripts/build_phase1_dataset.py — sector-ETF and industry-peer
features are computed from other symbols' OWN daily_return columns, which
only exist once the base build has already run once.

Labels (data/processed/{symbol}_labels.csv) are untouched — they only
depend on price/technical columns (see labels/horizons.py), not on these.
"""

from __future__ import annotations

import os
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features.industry.industry_features import add_stock_vs_industry, compute_industry_peer_returns  # noqa: E402
from features.macro.macro_features import MACRO_SERIES, compute_macro_series_features, merge_macro_features  # noqa: E402
from features.sector.sector_features import add_stock_vs_sector, compute_sector_features  # noqa: E402
from ingestion.ohlcv_source import SampleCSVSource  # noqa: E402
from models.common import PROCESSED_DIR  # noqa: E402
from universe.universe_loader import Universe  # noqa: E402

REAL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "market", "real")


def main():
    universe = Universe()
    symbols = universe.all_symbols(active_only=True)
    source = SampleCSVSource(data_dir=REAL_DIR)
    start, end = date(2018, 1, 1), date.today()

    print("Computing sector ETF features (one per sector, reused across its stocks)...")
    sector_features_cache: dict[str, pd.DataFrame] = {}
    for sector_code in universe.sectors["sector_code"]:
        etf = universe.sector_etf(sector_code)
        try:
            bars = source.get_bars(etf, start, end)
        except FileNotFoundError:
            print(f"  {sector_code} ({etf}): no data, skipping")
            continue
        sector_features_cache[sector_code] = compute_sector_features(bars)
    print(f"  {len(sector_features_cache)} sector ETFs computed")

    print("Computing industry peer-group returns...")
    industry_returns = compute_industry_peer_returns(universe)
    print(f"  {len(industry_returns)} industries have >=2 peers: {sorted(industry_returns.keys())}")

    print("Computing macro series features...")
    macro_frames = {}
    for name in MACRO_SERIES:
        try:
            bars = source.get_bars(name, start, end)
        except FileNotFoundError:
            print(f"  {name}: no data, skipping (run scripts/fetch_macro_data.py)")
            continue
        macro_frames[name] = compute_macro_series_features(bars, name)
    print(f"  {len(macro_frames)}/{len(MACRO_SERIES)} macro series computed")

    ok = 0
    for symbol in symbols:
        path = os.path.join(PROCESSED_DIR, f"{symbol}_features.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        stock = universe.get(symbol)

        sector_feat = sector_features_cache.get(stock.sector_code)
        if sector_feat is not None:
            df = add_stock_vs_sector(df, sector_feat)

        df = add_stock_vs_industry(df, industry_returns.get(stock.industry_code))

        if macro_frames:
            df = merge_macro_features(df, macro_frames)

        df.to_csv(path, index=False)
        ok += 1

    print(f"\nAugmented {ok}/{len(symbols)} symbols' feature files with sector/industry/macro columns")


if __name__ == "__main__":
    main()

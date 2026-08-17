"""
Fetches REAL historical OHLCV bars (via yfinance/Yahoo Finance) for the full
universe + benchmark/sector ETFs, and writes them in the exact same CSV
schema `scripts/generate_sample_data.py` produces — so every downstream
piece (features, labels, training, backtesting, the API) works unmodified
against real data by pointing SampleCSVSource at data/market/real/ instead
of data/market/sample/.

Adjusted close (`adj_close`) comes straight from Yahoo, which already
handles split/dividend adjustment — no separate corporate-action processing
needed for this path (ingestion/corporate_actions.py's validators still work
against this data if you want to sanity-check it).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features.market.market_features import BENCHMARKS  # noqa: E402
from ingestion.ohlcv_source import YFinanceSource  # noqa: E402
from universe.universe_loader import Universe  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "market", "real")
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]


def fetch_all(start: date, end: date, out_dir: str = OUT_DIR, pause_seconds: float = 0.5) -> None:
    os.makedirs(out_dir, exist_ok=True)
    universe = Universe()
    symbols = universe.all_symbols(active_only=True)
    all_symbols = sorted(set(symbols) | set(BENCHMARKS) | set(SECTOR_ETFS))

    source = YFinanceSource()
    ok, failed = 0, []
    for i, symbol in enumerate(all_symbols):
        try:
            df = source.get_bars(symbol, start, end)
            if df.empty or len(df) < 100:
                failed.append((symbol, f"only {len(df)} rows"))
                continue
            df.to_csv(os.path.join(out_dir, f"{symbol}.csv"), index=False)
            ok += 1
            print(f"  [{i+1}/{len(all_symbols)}] {symbol}: {len(df)} rows ({df['date'].min().date()} to {df['date'].max().date()})")
        except Exception as exc:
            failed.append((symbol, str(exc)))
            print(f"  [{i+1}/{len(all_symbols)}] {symbol}: FAILED ({exc})")
        time.sleep(pause_seconds)  # be polite to Yahoo's free endpoint

    print(f"\nFetched {ok}/{len(all_symbols)} symbols to {out_dir}")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--pause", type=float, default=0.5)
    args = parser.parse_args()

    fetch_all(date.fromisoformat(args.start), date.fromisoformat(args.end), pause_seconds=args.pause)

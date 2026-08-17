"""
Fetches REAL macro/market-indicator data (spec section 17) via yfinance —
the market-based half of that section: VIX, DXY, 10Y Treasury yield, Gold,
Oil, Copper, Bitcoin. All free, no registration, no API key.

NOT included: CPI, PPI, unemployment, GDP, ISM, Consumer Confidence, Fed
Funds Rate. Those are real economic releases with a genuine point-in-time
leakage risk (reported with a lag, then revised later) — the exact thing
the spec explicitly warns about ("historical macro data must only be
available after its actual publication timestamp, do not leak revised
future values"). yfinance doesn't carry them at all; a correct
implementation needs FRED's API (free, but requires registration for an API
key this project doesn't have) with the ingestion built around FRED's own
`realtime_start`/`realtime_end` vintage fields to get point-in-time
correctness right — a real, separate piece of work, not something to fake
by pulling today's CPI value into 2019's training rows.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion.ohlcv_source import YFinanceSource  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "market", "real")

# yfinance ticker -> our internal macro series name
MACRO_TICKERS = {
    "^VIX": "VIX",
    "^TNX": "US10Y",       # CBOE 10-Year Treasury Note Yield Index (already in % terms, e.g. 3.97 = 3.97%)
    "DX-Y.NYB": "DXY",
    "GLD": "GOLD",
    "USO": "OIL",
    "CPER": "COPPER",
    "BTC-USD": "BITCOIN",
}


def fetch_all(start: date, end: date, out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    source = YFinanceSource()
    ok, failed = 0, []

    for yf_ticker, name in MACRO_TICKERS.items():
        try:
            df = source.get_bars(yf_ticker, start, end)
            if df.empty:
                failed.append((name, "empty"))
                continue
            df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
            ok += 1
            print(f"  {name} ({yf_ticker}): {len(df)} rows ({df['date'].min().date()} to {df['date'].max().date()})")
        except Exception as exc:
            failed.append((name, str(exc)))
            print(f"  {name} ({yf_ticker}): FAILED ({exc})")

    print(f"\nFetched {ok}/{len(MACRO_TICKERS)} macro series to {out_dir}")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=date.today().isoformat())
    args = parser.parse_args()
    fetch_all(date.fromisoformat(args.start), date.fromisoformat(args.end))

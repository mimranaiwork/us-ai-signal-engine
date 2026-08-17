"""
End-to-end Phase 1 pipeline: universe -> OHLCV -> benchmark features ->
per-symbol price/technical features + relative strength -> seven-horizon
labels -> data/processed/*.csv.

Run after scripts/generate_sample_data.py (synthetic) or
scripts/fetch_real_data.py (real, via yfinance) depending on --data-dir.
This is what "complete runnable code" means for Phase 1 — everything before
Phase 2's model training.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features.market.market_features import BENCHMARKS, add_relative_strength, compute_all_benchmark_features  # noqa: E402
from features.price.price_features import add_price_features  # noqa: E402
from features.technical.indicators import add_all_technical_indicators  # noqa: E402
from ingestion.ohlcv_source import SampleCSVSource  # noqa: E402
from labels.horizons import LabelConfig, generate_all_horizons  # noqa: E402
from universe.universe_loader import Universe  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "market", "sample")
REAL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "market", "real")


def build(start: date, end: date, out_dir: str = OUT_DIR, label_method: str = "volatility",
          data_dir: str = SAMPLE_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    source = SampleCSVSource(data_dir=data_dir)
    universe = Universe()
    symbols = universe.all_symbols(active_only=True)

    print(f"Universe: {len(symbols)} active symbols")
    print("Loading benchmark ETFs and computing market-context features...")
    benchmark_bars = {b: source.get_bars(b, start, end) for b in BENCHMARKS}
    benchmark_features = compute_all_benchmark_features(benchmark_bars)

    label_config = LabelConfig(method=label_method)
    ok, failed = 0, []

    for symbol in symbols:
        try:
            bars = source.get_bars(symbol, start, end)
            if len(bars) < 260:  # need at least ~1yr for the 200D/252D lookbacks to mean anything
                failed.append((symbol, "insufficient history"))
                continue

            df = add_price_features(bars)
            df = add_all_technical_indicators(df)
            for bench in BENCHMARKS:
                df = add_relative_strength(df, benchmark_features[bench], prefix=bench)

            labels = generate_all_horizons(df, label_config)

            df.to_csv(os.path.join(out_dir, f"{symbol}_features.csv"), index=False)
            labels.to_csv(os.path.join(out_dir, f"{symbol}_labels.csv"), index=False)
            ok += 1
        except FileNotFoundError as exc:
            failed.append((symbol, str(exc)))

    print(f"Built features + labels for {ok}/{len(symbols)} symbols")
    if failed:
        print(f"Skipped {len(failed)}: {failed[:5]}{' ...' if len(failed) > 5 else ''}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-08-08")
    parser.add_argument("--label-method", default="volatility",
                         choices=["fixed", "atr", "volatility", "market_relative", "risk_adjusted"])
    parser.add_argument("--source", default="sample", choices=["sample", "real"],
                         help="sample = synthetic GBM data; real = yfinance data (run scripts/fetch_real_data.py first)")
    args = parser.parse_args()

    build(
        date.fromisoformat(args.start),
        date.fromisoformat(args.end),
        label_method=args.label_method,
        data_dir=REAL_DIR if args.source == "real" else SAMPLE_DIR,
    )

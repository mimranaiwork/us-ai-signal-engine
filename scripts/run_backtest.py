"""
Phase 4 orchestration: runs the walk-forward classification harness across
all seven horizons (retraining XGBoost per fold — see backtesting/
walk_forward.py's docstring for why sequence models aren't auto-walked), and
a trading-simulation backtest for a few representative symbols, writing
everything to backtesting_report.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtesting.walk_forward import evaluate_fold, evaluate_fold_trading, generate_folds  # noqa: E402
from calendar_us.trading_calendar import HORIZON_SESSIONS  # noqa: E402
from models.common import default_universe_symbols  # noqa: E402

REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "backtesting", "backtesting_report.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--n-symbols", type=int, default=10, help="Used when --symbols isn't given")
    parser.add_argument("--trading-symbols", nargs="*", default=["AAPL", "NVDA", "JPM"])
    parser.add_argument("--data-start", default="2018-01-01")
    parser.add_argument("--data-end", default="2026-08-08")
    args = parser.parse_args()

    symbols = args.symbols or default_universe_symbols()[: args.n_symbols]
    folds = generate_folds(date.fromisoformat(args.data_start), date.fromisoformat(args.data_end))
    print(f"{len(folds)} walk-forward folds, {len(symbols)} symbols for classification evaluation")

    classification_results = []
    for horizon in HORIZON_SESSIONS:
        for fold in folds:
            print(f"  horizon={horizon}  test=[{fold.val_end}, {fold.test_end})")
            result = evaluate_fold(symbols, horizon, fold)
            classification_results.append(result)
            if "error" not in result:
                acc = result["test_classification"]["accuracy"]
                print(f"    test accuracy: {acc:.4f}")

    trading_results = []
    for symbol in args.trading_symbols:
        for horizon in ["1W", "1M", "3M"]:  # representative subset — full 7x is slow to run repeatedly
            for fold in folds[-2:]:  # most recent folds only, for a runtime-manageable report
                print(f"  trading sim: {symbol} {horizon} test=[{fold.val_end}, {fold.test_end})")
                result = evaluate_fold_trading(symbol, horizon, fold)
                trading_results.append(result)

    report = {"classification": classification_results, "trading": trading_results}
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

"""
LSTM/Transformer walk-forward evaluation (backtesting/walk_forward.py's
evaluate_fold_sequence) — genuinely retrains both architectures per fold,
unlike the Phase 2 single train/val/test split in training/train.py.

Deliberately reduced scope vs. scripts/run_backtest.py's XGBoost walk-forward
(all 7 horizons x every fold): sequence retraining is minutes per
fold-horizon-architecture combination, not seconds, so this runs a
REPRESENTATIVE subset — 3 horizons (short/medium/long) x the 2 most recent
folds x both architectures — and says so in the report rather than silently
presenting a partial run as the full picture. Sequence windows are built
ONCE per (symbols, seq_len) and reused across every fold/horizon/architecture
combination (the expensive, fold-independent part), not rebuilt per call.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtesting.walk_forward import evaluate_fold_sequence, generate_folds  # noqa: E402
from models.common import default_universe_symbols  # noqa: E402
from models.lstm.model import LSTMMultiHorizonModel  # noqa: E402
from models.sequence_common import build_sequences  # noqa: E402
from models.transformer.model import TransformerMultiHorizonModel  # noqa: E402

REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "backtesting", "sequence_walk_forward_report.json")
ARCHITECTURES = {"lstm": LSTMMultiHorizonModel, "transformer": TransformerMultiHorizonModel}
REPRESENTATIVE_HORIZONS = ["1D", "1M", "6M"]  # short/medium/long, not all 7 — see module docstring


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--n-symbols", type=int, default=15, help="Used when --symbols isn't given")
    parser.add_argument("--data-start", default="2018-01-01")
    parser.add_argument("--data-end", default="2026-08-08")
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--n-folds", type=int, default=2, help="Most recent N folds only")
    args = parser.parse_args()

    symbols = args.symbols or default_universe_symbols()[: args.n_symbols]
    folds = generate_folds(date.fromisoformat(args.data_start), date.fromisoformat(args.data_end))
    folds = folds[-args.n_folds:]
    print(f"{len(folds)} folds (most recent), {len(symbols)} symbols, seq_len={args.seq_len}, epochs={args.epochs}")

    print("Building sequence windows once (reused across every fold/horizon/architecture)...")
    t0 = time.time()
    sequences = build_sequences(symbols, seq_len=args.seq_len)
    print(f"  {sequences[0].shape} in {time.time() - t0:.1f}s")

    results = []
    for arch_name, model_cls in ARCHITECTURES.items():
        for horizon in REPRESENTATIVE_HORIZONS:
            for fold in folds:
                print(f"  {arch_name} horizon={horizon} test=[{fold.val_end}, {fold.test_end})")
                t0 = time.time()
                result = evaluate_fold_sequence(
                    model_cls, symbols, horizon, fold,
                    seq_len=args.seq_len, epochs=args.epochs, hidden_size=args.hidden_size,
                    sequences=sequences,
                )
                elapsed = time.time() - t0
                result["elapsed_seconds"] = round(elapsed, 1)
                results.append(result)
                if "error" not in result:
                    acc = result["test_classification"]["accuracy"]
                    print(f"    test accuracy: {acc:.4f}  ({elapsed:.1f}s)")
                else:
                    print(f"    {result['error']}")

    report = {
        "note": (
            f"Representative subset: horizons={REPRESENTATIVE_HORIZONS}, "
            f"{len(folds)} most recent fold(s), {len(symbols)} symbols, epochs={args.epochs} — "
            "NOT the full 7-horizon x all-folds sweep run_backtest.py does for XGBoost, "
            "see backtesting/walk_forward.py's module docstring for why."
        ),
        "config": vars(args),
        "results": results,
    }
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

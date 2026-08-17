"""
Runs backtesting/optimize.py's grid search on real market data and prints
the winning horizon-weight/model-weight/threshold config — spec sections
32/33/35's "don't assume the initial weights/thresholds are optimal, use
walk-forward validation" instruction, actually carried out rather than left
as a TODO comment.

Validates on 2024 (a full year not used in training/train.py's Phase 2 train
window, which ends 2023-01-01), confirms on 2025+ as a genuinely held-out
check the optimizer never saw.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtesting.optimize import optimize, evaluate_config, HORIZON_WEIGHT_PRESETS, MODEL_WEIGHT_PRESETS  # noqa: E402
from models.common import default_universe_symbols  # noqa: E402
from models.xgboost.model import XGBoostMultiHorizonModel  # noqa: E402
from models.lightgbm.model import LightGBMMultiHorizonModel  # noqa: E402

RESULT_PATH = os.path.join(os.path.dirname(__file__), "..", "backtesting", "optimize_report.json")


def main():
    symbols = default_universe_symbols()[:12]  # representative subset — full 40x80-combo grid is a lot of predict_proba calls
    val_start, val_end = date(2024, 1, 1), date(2025, 1, 1)
    test_start, test_end = date(2025, 1, 1), date(2026, 8, 11)

    result = optimize(symbols, val_start, val_end)
    best = result["best"]
    print(f"\nBest on validation [{val_start}, {val_end}): {best}")

    # Confirm on the held-out test window — the optimizer never saw this.
    xgb = XGBoostMultiHorizonModel.load()
    lgb = LightGBMMultiHorizonModel.load()
    tree_models = {"xgboost": xgb, "lightgbm": lgb}
    test_sharpe = evaluate_config(
        symbols, tree_models, result["horizon_weights"], result["model_weights"],
        best["buy_threshold"], best["sell_threshold"], test_start, test_end,
    )
    spec_default_sharpe = evaluate_config(
        symbols, tree_models, HORIZON_WEIGHT_PRESETS["spec_default"], MODEL_WEIGHT_PRESETS["spec_default"],
        0.25, -0.25, test_start, test_end,
    )
    print(f"Test-window [{test_start}, {test_end}) avg Sharpe — optimized: {test_sharpe:+.3f}  "
          f"vs. spec-default weights/thresholds: {spec_default_sharpe:+.3f}")

    report = {
        "symbols": symbols, "validation_window": [str(val_start), str(val_end)],
        "test_window": [str(test_start), str(test_end)],
        "best_on_validation": best, "test_sharpe_optimized": test_sharpe,
        "test_sharpe_spec_default": spec_default_sharpe,
        "leaderboard": result["leaderboard"],
    }
    with open(RESULT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote {RESULT_PATH}")


if __name__ == "__main__":
    main()

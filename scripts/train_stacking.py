"""
Trains the stacking meta-model (signals/stacking.py) per horizon and
compares it against the existing fixed-weight ensemble (signals/scoring.py's
model_weighted_average) on genuinely held-out data.

Split (matches training/train.py's TRAIN_END/VAL_END, so base models never
see anything used here for gradient/tree updates):
  - meta-train: VAL split [2023-01-01, 2024-01-01) — base models only used
    this for early-stopping/epoch monitoring, not fitting.
  - eval:       TEST split [2024-01-01, ...) — untouched by base models
    AND untouched by the meta-model's own fitting.

Reports classification metrics (argmax accuracy/F1/ROC-AUC) for both
ensembling methods per horizon, on the same eval rows, so the winner is
decided on equal footing. Whichever wins is intended to become the
signal_engine's ensemble method (see README for how this gets wired in).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402

from backtesting.metrics import classification_metrics  # noqa: E402
from calendar_us.trading_calendar import HORIZON_SESSIONS  # noqa: E402
from models.common import LABEL_ORDER, LABEL_TO_IDX, default_universe_symbols  # noqa: E402
from models.predict import ModelRegistry  # noqa: E402
from signals.stacking import StackingEnsemble, collect_base_predictions, prob_columns  # noqa: E402
from training.train import TRAIN_END, VAL_END  # noqa: E402

REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "saved", "stacking_report.json")

# Un-optimized spec placeholders for lstm/transformer (matches config/settings.yaml
# model_weights — see that file's comment on why lstm/transformer/ppo weren't
# part of the walk-forward weight search).
MODEL_WEIGHTS = {"xgboost": 0.30, "lightgbm": 0.25, "lstm": 0.20, "transformer": 0.15}


def weighted_average_proba(df, model_weights: dict[str, float], models_present: list[str]) -> np.ndarray:
    """Same math as signals/scoring.py's model_weighted_average(), vectorized
    over rows instead of looping ModelHorizonPrediction objects — every row
    here already has all `models_present`, so total_weight is constant."""
    total_w = sum(model_weights.get(m, 0.0) for m in models_present)
    if total_w <= 0:
        total_w, weights = len(models_present), {m: 1.0 for m in models_present}
    else:
        weights = {m: model_weights.get(m, 0.0) for m in models_present}

    proba = np.zeros((len(df), 3))
    for i, label in enumerate(LABEL_ORDER):
        for m in models_present:
            proba[:, i] += (weights[m] / total_w) * df[f"{m}_{label}"].to_numpy()
    return proba


def main():
    symbols = default_universe_symbols()
    registry = ModelRegistry()
    models_present = list(registry.tree_models) + list(registry.sequence_models)
    print(f"Loaded base models: {models_present}")

    stacker = StackingEnsemble()
    report = {"models_present": models_present, "horizons": {}}

    for horizon in HORIZON_SESSIONS:
        meta_train_df = collect_base_predictions(registry, symbols, horizon, date_start=TRAIN_END, date_end=VAL_END)
        eval_df = collect_base_predictions(registry, symbols, horizon, date_start=VAL_END)

        if len(meta_train_df) < 50 or len(eval_df) < 50:
            print(f"  {horizon}: insufficient rows (train={len(meta_train_df)}, eval={len(eval_df)}), skipping")
            continue

        fit_info = stacker.fit_horizon(horizon, meta_train_df)

        y_eval = eval_df["label"].map(LABEL_TO_IDX).to_numpy()

        stack_proba = stacker.predict_proba(horizon, eval_df)
        stack_pred = np.argmax(stack_proba, axis=1)
        stack_metrics = classification_metrics(y_eval, stack_pred, stack_proba)

        wavg_proba = weighted_average_proba(eval_df, MODEL_WEIGHTS, models_present)
        wavg_pred = np.argmax(wavg_proba, axis=1)
        wavg_metrics = classification_metrics(y_eval, wavg_pred, wavg_proba)

        winner = "stacking" if stack_metrics["f1_macro"] > wavg_metrics["f1_macro"] else "weighted_average"
        print(
            f"  {horizon}: n_meta_train={fit_info['n_train']} n_eval={len(eval_df)}  "
            f"stacking f1={stack_metrics['f1_macro']:.4f} acc={stack_metrics['accuracy']:.4f}  |  "
            f"weighted_avg f1={wavg_metrics['f1_macro']:.4f} acc={wavg_metrics['accuracy']:.4f}  "
            f"-> {winner}"
        )

        report["horizons"][horizon] = {
            "n_meta_train": fit_info["n_train"],
            "n_eval": len(eval_df),
            "stacking": stack_metrics,
            "weighted_average": wavg_metrics,
            "winner": winner,
        }

    n_stacking_wins = sum(1 for h in report["horizons"].values() if h["winner"] == "stacking")
    n_total = len(report["horizons"])
    report["summary"] = {"stacking_wins": n_stacking_wins, "total_horizons": n_total}
    print(f"\nStacking wins {n_stacking_wins}/{n_total} horizons (by macro-F1 on held-out test).")

    stacker.save()
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Stacking models saved to {os.path.join(os.path.dirname(__file__), '..', 'models', 'saved', 'stacking')}")
    print(f"Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()

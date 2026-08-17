"""
Phase 2 training orchestration: trains all four model types (XGBoost,
LightGBM, LSTM, Transformer) across all seven horizons on Phase 1's
processed universe data, and saves each to models/saved/{arch}/.

Uses a simple chronological train/val/test split (train < 2023, val 2023,
test >= 2024) — Phase 4's backtesting engine replaces this single split with
a full walk-forward harness (multiple rolling windows); Phase 2's job is
just "does each architecture train and produce calibrated-shape probability
outputs per horizon", not picking a final validated model yet.

Expect near-baseline (~33-40%) validation accuracy on the shipped synthetic
sample data — it's geometric Brownian motion with no engineered predictive
signal, so a model "learning" much more than the majority-class baseline
would actually indicate a leakage bug, not a good model. This is the correct,
honest outcome for Phase 2 on synthetic data; real predictive power is a
question for real market data, which is a data-sourcing decision, not a
modeling one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from calendar_us.trading_calendar import HORIZON_SESSIONS  # noqa: E402
from models.common import build_pooled_dataset, chronological_split, default_universe_symbols
from models.lightgbm.model import LightGBMMultiHorizonModel
from models.lstm.model import LSTMMultiHorizonModel
from models.sequence_common import build_sequences, fit_scaler, apply_scaler
from models.transformer.model import TransformerMultiHorizonModel
from models.xgboost.model import XGBoostMultiHorizonModel

TRAIN_END = date(2023, 1, 1)
VAL_END = date(2024, 1, 1)

REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "saved", "training_report.json")


def train_tree_models(symbols: list[str]) -> tuple[dict, dict]:
    report = {"xgboost": {}, "lightgbm": {}}
    xgb_model = XGBoostMultiHorizonModel()
    lgb_model = LightGBMMultiHorizonModel()

    for horizon in HORIZON_SESSIONS:
        print(f"\n=== Tree models — horizon {horizon} ===")
        pooled = build_pooled_dataset(symbols, horizon)
        train, val, test = chronological_split(pooled, TRAIN_END, VAL_END)
        print(f"  rows: train={len(train)} val={len(val)} test={len(test)}")

        report["xgboost"][horizon] = xgb_model.train_horizon(train, val, horizon)
        print(f"  [XGBoost] val_accuracy={report['xgboost'][horizon]['val_accuracy']:.4f}")

        report["lightgbm"][horizon] = lgb_model.train_horizon(train, val, horizon)
        print(f"  [LightGBM] val_accuracy={report['lightgbm'][horizon]['val_accuracy']:.4f}")

    xgb_model.save()
    lgb_model.save()
    return report, {"xgboost": xgb_model, "lightgbm": lgb_model}


def train_sequence_models(symbols: list[str], seq_len: int = 60, epochs: int = 3, hidden_size: int = 32) -> tuple[dict, dict]:
    print(f"\n=== Building sequences (seq_len={seq_len}) ===")
    X, Y, meta = build_sequences(symbols, seq_len=seq_len)
    print(f"  sequences: {X.shape}")

    dates = meta["date"]
    train_mask = dates.dt.date < TRAIN_END
    val_mask = (dates.dt.date >= TRAIN_END) & (dates.dt.date < VAL_END)

    mean, std = fit_scaler(X[train_mask.to_numpy()])
    Xs = apply_scaler(X, mean, std)

    X_train, X_val = Xs[train_mask.to_numpy()], Xs[val_mask.to_numpy()]
    Y_train = {h: y[train_mask.to_numpy()] for h, y in Y.items()}
    Y_val = {h: y[val_mask.to_numpy()] for h, y in Y.items()}
    print(f"  train={len(X_train)} val={len(X_val)}")

    report = {}

    print("\n=== LSTM ===")
    lstm = LSTMMultiHorizonModel(seq_len=seq_len, hidden_size=hidden_size)
    lstm_history = lstm.fit(X_train, Y_train, X_val, Y_val, mean, std, epochs=epochs)
    lstm.save()
    report["lstm"] = lstm_history[-1]

    print("\n=== Transformer ===")
    transformer = TransformerMultiHorizonModel(seq_len=seq_len, hidden_size=hidden_size, n_heads=4, n_layers=2)
    transformer_history = transformer.fit(X_train, Y_train, X_val, Y_val, mean, std, epochs=epochs)
    transformer.save()
    report["transformer"] = transformer_history[-1]

    return report, {"lstm": lstm, "transformer": transformer}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*", default=None, help="Subset of symbols; default = full active universe")
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--skip-sequence-models", action="store_true", help="Only train XGBoost/LightGBM (faster)")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow experiment tracking/model logging")
    args = parser.parse_args()

    symbols = args.symbols or default_universe_symbols()
    print(f"Training on {len(symbols)} symbols")

    report = {}
    trained_models = {}

    report["tree_models"], tree_models = train_tree_models(symbols)
    trained_models.update(tree_models)

    if not args.skip_sequence_models:
        report["sequence_models"], sequence_models = train_sequence_models(
            symbols, seq_len=args.seq_len, epochs=args.epochs, hidden_size=args.hidden_size
        )
        trained_models.update(sequence_models)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nTraining report written to {REPORT_PATH}")

    if not args.no_mlflow:
        from training.mlflow_tracking import log_training_run
        run_params = {
            "n_symbols": len(symbols), "seq_len": args.seq_len,
            "epochs": args.epochs, "hidden_size": args.hidden_size,
            "skip_sequence_models": args.skip_sequence_models,
        }
        run_id = log_training_run(run_params, report, trained_models)
        print(f"MLflow run: {run_id}  (mlflow ui --backend-store-uri mlruns to view)")


if __name__ == "__main__":
    main()

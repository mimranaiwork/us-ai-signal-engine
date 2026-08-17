"""
Walk-forward validation harness (spec section 45): expanding-window
train/validation/test folds, each retraining fresh rather than reusing a
single static train/val/test split.

Retrains the XGBoost/LightGBM tree models per fold via `evaluate_fold()`
(seconds per horizon — cheap enough to actually walk forward).
`evaluate_fold_sequence()` does the same for LSTM/Transformer — genuinely
retraining per fold, not just re-evaluating the single Phase 2 split — but
it's materially more expensive (minutes per fold per architecture, not
seconds) since sequence training already hit real memory/threading issues
on this machine during Phase 2 (see README); `scripts/run_sequence_walk_forward.py`
runs it with a reduced epoch count and reports honestly what that buys vs.
the tree models' full per-horizon walk-forward. `WalkForwardSplitter` and
`evaluate_fold`/`evaluate_fold_sequence` are architecture-agnostic though —
point them at any model with a compatible train/predict interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from backtesting.costs import TransactionCostModel
from backtesting.engine import run_backtest
from backtesting.metrics import classification_metrics, trading_metrics
from calendar_us.trading_calendar import HORIZON_SESSIONS
from features.market.market_features import BENCHMARKS
from ingestion.ohlcv_source import SampleCSVSource
from models.common import LABEL_TO_IDX, build_pooled_dataset, encode_labels, feature_columns
from models.sequence_common import MISSING_LABEL, apply_scaler, build_sequences, fit_scaler
from models.xgboost.model import XGBoostMultiHorizonModel


@dataclass
class WalkForwardFold:
    train_start: date
    train_end: date   # exclusive
    val_end: date      # exclusive
    test_end: date     # exclusive


def generate_folds(
    data_start: date, data_end: date,
    min_train_years: int = 3, val_years: int = 1, test_years: int = 1, step_years: int = 1,
) -> list[WalkForwardFold]:
    """
    Expanding-window folds, e.g. (matching the spec's own example):
        2018-2021 train | 2021-2022 val | 2022-2023 test
        2018-2022 train | 2022-2023 val | 2023-2024 test
        ...
    Train window always starts at data_start and grows; val/test windows are
    each `val_years`/`test_years` long and roll forward by `step_years`.
    """
    folds = []
    train_end = date(data_start.year + min_train_years, data_start.month, data_start.day)
    while True:
        val_end = date(train_end.year + val_years, train_end.month, train_end.day)
        test_end = date(val_end.year + test_years, val_end.month, val_end.day)
        if test_end > data_end:
            break
        folds.append(WalkForwardFold(data_start, train_end, val_end, test_end))
        train_end = date(train_end.year + step_years, train_end.month, train_end.day)
    return folds


def evaluate_fold(
    symbols: list[str], horizon: str, fold: WalkForwardFold,
) -> dict:
    """Retrains an XGBoost model on [train_start, train_end), validates on
    [train_end, val_end), and reports classification metrics on the held-out
    [val_end, test_end) test window — the fold's genuinely out-of-sample slice."""
    pooled = build_pooled_dataset(symbols, horizon)
    d = pooled["date"].dt.date

    train_df = pooled[(d >= fold.train_start) & (d < fold.train_end)]
    val_df = pooled[(d >= fold.train_end) & (d < fold.val_end)]
    test_df = pooled[(d >= fold.val_end) & (d < fold.test_end)]

    if len(train_df) < 100 or len(val_df) < 20 or len(test_df) < 20:
        return {"horizon": horizon, "fold": fold, "error": "insufficient data in one or more splits"}

    model = XGBoostMultiHorizonModel()
    train_report = model.train_horizon(train_df, val_df, horizon)

    feat_cols = model.feature_cols
    X_test = test_df[feat_cols]
    y_test = encode_labels(test_df["label"])
    y_proba = model.predict_proba(X_test, horizon)
    y_pred = np.argmax(y_proba, axis=1)

    cls_metrics = classification_metrics(y_test, y_pred, y_proba)

    return {
        "horizon": horizon,
        "fold": {"train_start": str(fold.train_start), "train_end": str(fold.train_end),
                  "val_end": str(fold.val_end), "test_end": str(fold.test_end)},
        "train_val_accuracy": train_report["val_accuracy"],
        "test_classification": cls_metrics,
        "n_test": len(test_df),
    }


def evaluate_fold_sequence(
    model_cls: type, symbols: list[str], horizon: str, fold: WalkForwardFold,
    seq_len: int = 60, epochs: int = 3, hidden_size: int = 32,
    sequences: tuple[np.ndarray, dict[str, np.ndarray], pd.DataFrame] | None = None,
) -> dict:
    """LSTM/Transformer counterpart to evaluate_fold(): genuinely retrains
    `model_cls` (LSTMMultiHorizonModel or TransformerMultiHorizonModel) on
    [train_start, train_end), using [train_end, val_end) the same way
    training/train.py does — epoch-level monitoring only, no gradient
    updates from it — then reports classification metrics for `horizon` on
    the held-out [val_end, test_end) test window.

    Trains jointly across all seven horizon heads per fold (that's the
    model's actual architecture — one shared encoder, see
    models/sequence_common.py), but only this fold's requested `horizon` is
    scored, matching evaluate_fold()'s one-horizon-per-call shape so the two
    are directly comparable.

    `sequences`: optional pre-built `build_sequences(symbols, seq_len=seq_len)`
    result — building sequence windows is the expensive, fold-independent
    part; a driver walking multiple folds/horizons/architectures should
    build it once and pass it in rather than re-windowing per call."""
    X, Y, meta = sequences if sequences is not None else build_sequences(symbols, seq_len=seq_len)
    d = meta["date"].dt.date

    train_mask = ((d >= fold.train_start) & (d < fold.train_end)).to_numpy()
    val_mask = ((d >= fold.train_end) & (d < fold.val_end)).to_numpy()
    test_mask = ((d >= fold.val_end) & (d < fold.test_end)).to_numpy()

    y_horizon = Y[horizon]
    test_valid = test_mask & (y_horizon != MISSING_LABEL)

    if train_mask.sum() < 100 or val_mask.sum() < 20 or test_valid.sum() < 20:
        return {"horizon": horizon, "fold": fold, "error": "insufficient data in one or more splits"}

    mean, std = fit_scaler(X[train_mask])
    Xs = apply_scaler(X, mean, std)
    X_train, X_val = Xs[train_mask], Xs[val_mask]
    Y_train = {h: y[train_mask] for h, y in Y.items()}
    Y_val = {h: y[val_mask] for h, y in Y.items()}

    model = model_cls(seq_len=seq_len, hidden_size=hidden_size)
    history = model.fit(X_train, Y_train, X_val, Y_val, mean, std, epochs=epochs)

    y_test = y_horizon[test_valid]
    y_proba = model.predict_proba_batch(X[test_valid])[horizon]
    y_pred = np.argmax(y_proba, axis=1)

    cls_metrics = classification_metrics(y_test, y_pred, y_proba)

    return {
        "horizon": horizon,
        "architecture": model.ARCH_NAME,
        "fold": {"train_start": str(fold.train_start), "train_end": str(fold.train_end),
                  "val_end": str(fold.val_end), "test_end": str(fold.test_end)},
        "final_val_acc": history[-1]["val_acc"].get(horizon),
        "test_classification": cls_metrics,
        "n_test": int(test_valid.sum()),
    }


def evaluate_fold_trading(
    symbol: str, horizon: str, fold: WalkForwardFold,
    cost_model: TransactionCostModel = TransactionCostModel(),
) -> dict:
    """Same fold, but simulates the actual long-only strategy on ONE symbol
    over the test window and reports trading metrics vs Buy&Hold/SPY/QQQ."""
    pooled = build_pooled_dataset([symbol], horizon)
    d = pooled["date"].dt.date
    train_df = pooled[(d >= fold.train_start) & (d < fold.train_end)]
    val_df = pooled[(d >= fold.train_end) & (d < fold.val_end)]
    test_df = pooled[(d >= fold.val_end) & (d < fold.test_end)].sort_values("date").reset_index(drop=True)

    if len(train_df) < 100 or len(val_df) < 20 or len(test_df) < 20:
        return {"error": "insufficient data in one or more splits"}

    model = XGBoostMultiHorizonModel()
    model.train_horizon(train_df, val_df, horizon)

    y_proba = model.predict_proba(test_df[model.feature_cols], horizon)
    idx_to_label = {v: k for k, v in LABEL_TO_IDX.items()}
    predicted_labels = pd.Series([idx_to_label[i] for i in np.argmax(y_proba, axis=1)])

    source = SampleCSVSource()
    start, end = test_df["date"].min().date(), test_df["date"].max().date()
    benchmark_prices = {
        b: source.get_bars(b, start, end).set_index("date")["adj_close"].reindex(test_df["date"]).reset_index(drop=True)
        for b in BENCHMARKS
    }

    result = run_backtest(test_df["date"], test_df["adj_close"], predicted_labels, benchmark_prices, cost_model)

    return {
        "symbol": symbol, "horizon": horizon,
        "strategy": trading_metrics(result.equity_curve),
        "buy_and_hold": trading_metrics(result.buy_and_hold_equity_curve),
        "benchmarks": {name: trading_metrics(curve) for name, curve in result.benchmark_equity_curves.items()},
        "n_trades": len(result.trades),
    }

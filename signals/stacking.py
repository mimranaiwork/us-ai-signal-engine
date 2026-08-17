"""
Stacking / meta-model ensemble (spec section 34's other named option,
alongside the weighted-average already in signals/scoring.py's
model_weighted_average()): a per-horizon logistic regression trained on the
base models' own P(BUY)/P(HOLD)/P(SELL) outputs, learning a blend instead of
using the fixed model_weights.

Discipline, matching backtesting/walk_forward.py's train/val/test split:
  - base models (XGBoost/LightGBM/LSTM/Transformer) are already trained on
    TRAIN, using VAL only for early-stopping/epoch monitoring (no gradient
    updates from it) — see training/train.py.
  - the meta-model is fit on the base models' predictions on VAL — genuinely
    unseen by each base model's weight updates, so this is a legitimate
    (if slightly looser than k-fold-OOF) stacking setup, not double-dipping.
  - both the meta-model and the fixed-weight baseline are then scored on
    TEST, fully out-of-sample for everything, for an apples-to-apples
    comparison (see scripts/train_stacking.py).

LSTM/Transformer predictions are generated over every date via
models/sequence_common.build_sequences rather than per-row predict_latest
(there's no cheap "latest row" sequence version at scale) — inner-joined
with the tree models' per-row predictions on (symbol, date), so only rows
where all four base models have a real prediction enter the meta-model's
training/eval set.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from models.common import LABEL_ORDER, LABEL_TO_IDX, build_pooled_dataset
from models.sequence_common import MISSING_LABEL, build_sequences

if TYPE_CHECKING:
    from models.predict import ModelRegistry

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved", "stacking")

IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


def _tree_predictions(registry: ModelRegistry, horizon: str, pooled: pd.DataFrame) -> pd.DataFrame:
    out = pooled[["symbol", "date", "label"]].reset_index(drop=True)
    pooled = pooled.reset_index(drop=True)
    for name, model in registry.tree_models.items():
        if horizon not in model.models:
            continue
        proba = model.predict_proba(pooled[model.feature_cols], horizon)
        for i, label in enumerate(LABEL_ORDER):
            out[f"{name}_{label}"] = proba[:, i]
    return out


def _sequence_predictions(registry: ModelRegistry, symbols: list[str], horizon: str) -> pd.DataFrame | None:
    if not registry.sequence_models:
        return None
    seq_len = next(iter(registry.sequence_models.values())).seq_len
    X, Y, meta = build_sequences(symbols, seq_len=seq_len)

    y_enc = Y[horizon]
    valid = y_enc != MISSING_LABEL
    out = meta.loc[valid].reset_index(drop=True).copy()
    out["label"] = [IDX_TO_LABEL[v] for v in y_enc[valid]]

    for name, model in registry.sequence_models.items():
        proba = model.predict_proba_batch(X)[horizon][valid]
        for i, label in enumerate(LABEL_ORDER):
            out[f"{name}_{label}"] = proba[:, i]
    return out


def collect_base_predictions(
    registry: ModelRegistry, symbols: list[str], horizon: str,
    date_start=None, date_end=None,
) -> pd.DataFrame:
    """One row per (symbol, date) that ALL loaded base models produced a
    prediction for for `horizon`, restricted to [date_start, date_end) if
    given. Columns: symbol, date, label, {model}_{BUY,HOLD,SELL}..."""
    pooled = build_pooled_dataset(symbols, horizon)
    if date_start is not None:
        pooled = pooled[pooled["date"].dt.date >= date_start]
    if date_end is not None:
        pooled = pooled[pooled["date"].dt.date < date_end]

    tree_preds = _tree_predictions(registry, horizon, pooled)
    seq_preds = _sequence_predictions(registry, symbols, horizon)
    if seq_preds is None:
        return tree_preds

    merged = tree_preds.merge(
        seq_preds.drop(columns=["label"]), on=["symbol", "date"], how="inner",
    )
    return merged.reset_index(drop=True)


def prob_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("symbol", "date", "label")]


class StackingEnsemble:
    """One multinomial logistic regression meta-model per horizon, mapping
    the base models' concatenated class probabilities -> final BUY/HOLD/SELL
    probabilities. `feature_cols` (the exact {model}_{label} column order)
    is captured at fit time and enforced at predict time."""

    def __init__(self):
        self.models: dict[str, LogisticRegression] = {}
        self.feature_cols: dict[str, list[str]] = {}

    def fit_horizon(self, horizon: str, train_df: pd.DataFrame) -> dict:
        feat_cols = prob_columns(train_df)
        X = train_df[feat_cols].to_numpy()
        y = train_df["label"].map(LABEL_TO_IDX).to_numpy()

        clf = LogisticRegression(max_iter=1000)  # lbfgs solver is multinomial by default for >2 classes
        clf.fit(X, y)

        self.models[horizon] = clf
        self.feature_cols[horizon] = feat_cols
        train_acc = float((clf.predict(X) == y).mean())
        return {"horizon": horizon, "n_train": len(y), "train_accuracy": train_acc, "feature_cols": feat_cols}

    def predict_proba(self, horizon: str, df: pd.DataFrame) -> np.ndarray:
        clf = self.models[horizon]
        feat_cols = self.feature_cols[horizon]
        raw_proba = clf.predict_proba(df[feat_cols].to_numpy())
        # clf.classes_ only contains labels actually seen in this horizon's
        # meta-train split — if one class was entirely absent (label
        # imbalance is real here: SELL is rare on this mostly-uptrending
        # data, see config/settings.yaml's signal_thresholds comment),
        # sklearn drops that column rather than zero-filling it, which would
        # silently misalign column i with LABEL_TO_IDX[i] downstream. Map
        # explicitly through clf.classes_ instead of assuming a fixed width.
        proba = np.zeros((len(df), 3))
        for col_idx, class_idx in enumerate(clf.classes_):
            proba[:, class_idx] = raw_proba[:, col_idx]
        return proba

    def save(self, save_dir: str = SAVE_DIR) -> None:
        import joblib

        os.makedirs(save_dir, exist_ok=True)
        for horizon, clf in self.models.items():
            joblib.dump(clf, os.path.join(save_dir, f"{horizon}.joblib"))
        with open(os.path.join(save_dir, "meta.json"), "w") as f:
            json.dump({"feature_cols": self.feature_cols}, f)

    @classmethod
    def load(cls, save_dir: str = SAVE_DIR) -> "StackingEnsemble":
        import joblib

        with open(os.path.join(save_dir, "meta.json")) as f:
            meta = json.load(f)
        obj = cls()
        obj.feature_cols = meta["feature_cols"]
        for horizon in obj.feature_cols:
            path = os.path.join(save_dir, f"{horizon}.joblib")
            if os.path.exists(path):
                obj.models[horizon] = joblib.load(path)
        return obj

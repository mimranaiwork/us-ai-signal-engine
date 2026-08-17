"""LightGBM multi-horizon classifier — same interface as the XGBoost model
(models/xgboost/model.py) so the ensemble layer can treat both identically.
Trained independently rather than assumed better/worse than XGBoost, per
the spec — Phase 3's ensemble/backtesting decides which (or what blend)
actually performs better out-of-sample."""

from __future__ import annotations

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from calendar_us.trading_calendar import HORIZON_SESSIONS
from models.common import LABEL_ORDER, encode_labels, feature_columns, probs_to_dict

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "saved", "lightgbm")


class LightGBMMultiHorizonModel:
    def __init__(self):
        self.models: dict[str, lgb.Booster] = {}
        self.feature_cols: list[str] | None = None

    def train_horizon(self, train_df: pd.DataFrame, val_df: pd.DataFrame, horizon: str) -> dict:
        feat_cols = feature_columns(train_df)
        self.feature_cols = self.feature_cols or feat_cols

        X_train, y_train = train_df[feat_cols], encode_labels(train_df["label"])
        X_val, y_val = val_df[feat_cols], encode_labels(val_df["label"])

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "max_depth": 5,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbosity": -1,
        }
        model = lgb.train(
            params, train_set, num_boost_round=300,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        self.models[horizon] = model

        val_pred = np.argmax(model.predict(X_val), axis=1)
        acc = float((val_pred == y_val).mean())
        return {"horizon": horizon, "val_accuracy": acc, "n_train": len(X_train), "n_val": len(X_val)}

    def predict_proba(self, X: pd.DataFrame, horizon: str) -> np.ndarray:
        model = self.models[horizon]
        return model.predict(X[self.feature_cols], num_iteration=model.best_iteration)

    def predict_latest(self, row: pd.Series, horizon: str) -> dict[str, float]:
        X = pd.DataFrame([row[self.feature_cols]])
        probs = self.predict_proba(X, horizon)[0]
        return probs_to_dict(probs)

    def save(self, save_dir: str = SAVE_DIR) -> None:
        os.makedirs(save_dir, exist_ok=True)
        for horizon, model in self.models.items():
            model.save_model(os.path.join(save_dir, f"{horizon}.txt"))
        with open(os.path.join(save_dir, "meta.json"), "w") as f:
            json.dump({"feature_cols": self.feature_cols, "label_order": LABEL_ORDER}, f)

    @classmethod
    def load(cls, save_dir: str = SAVE_DIR) -> "LightGBMMultiHorizonModel":
        with open(os.path.join(save_dir, "meta.json")) as f:
            meta = json.load(f)
        obj = cls()
        obj.feature_cols = meta["feature_cols"]
        for horizon in HORIZON_SESSIONS:
            path = os.path.join(save_dir, f"{horizon}.txt")
            if os.path.exists(path):
                obj.models[horizon] = lgb.Booster(model_file=path)
        return obj

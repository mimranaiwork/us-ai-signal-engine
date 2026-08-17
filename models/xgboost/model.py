"""XGBoost multi-horizon classifier — one XGBClassifier per horizon (7 total),
each predicting P(BUY)/P(HOLD)/P(SELL). Handles technical/market features
natively; SEC/macro/options/sentiment features (Phase 5/6) slot in for free
once present in the pooled dataset, since this trains on whatever numeric
feature columns exist."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb

from calendar_us.trading_calendar import HORIZON_SESSIONS
from models.common import LABEL_ORDER, encode_labels, feature_columns, probs_to_dict

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "saved", "xgboost")


class XGBoostMultiHorizonModel:
    def __init__(self):
        self.models: dict[str, xgb.XGBClassifier] = {}
        self.feature_cols: list[str] | None = None

    def train_horizon(self, train_df: pd.DataFrame, val_df: pd.DataFrame, horizon: str) -> dict:
        feat_cols = feature_columns(train_df)
        self.feature_cols = self.feature_cols or feat_cols

        X_train, y_train = train_df[feat_cols], encode_labels(train_df["label"])
        X_val, y_val = val_df[feat_cols], encode_labels(val_df["label"])

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            early_stopping_rounds=20,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        self.models[horizon] = model

        val_pred = model.predict(X_val)
        acc = float((val_pred == y_val).mean())
        return {"horizon": horizon, "val_accuracy": acc, "n_train": len(X_train), "n_val": len(X_val)}

    def predict_proba(self, X: pd.DataFrame, horizon: str) -> np.ndarray:
        model = self.models[horizon]
        return model.predict_proba(X[self.feature_cols])

    def predict_latest(self, row: pd.Series, horizon: str) -> dict[str, float]:
        X = pd.DataFrame([row[self.feature_cols]])
        probs = self.predict_proba(X, horizon)[0]
        return probs_to_dict(probs)

    def save(self, save_dir: str = SAVE_DIR) -> None:
        os.makedirs(save_dir, exist_ok=True)
        for horizon, model in self.models.items():
            model.save_model(os.path.join(save_dir, f"{horizon}.json"))
        with open(os.path.join(save_dir, "meta.json"), "w") as f:
            json.dump({"feature_cols": self.feature_cols, "label_order": LABEL_ORDER}, f)

    @classmethod
    def load(cls, save_dir: str = SAVE_DIR) -> "XGBoostMultiHorizonModel":
        with open(os.path.join(save_dir, "meta.json")) as f:
            meta = json.load(f)
        obj = cls()
        obj.feature_cols = meta["feature_cols"]
        for horizon in HORIZON_SESSIONS:
            path = os.path.join(save_dir, f"{horizon}.json")
            if os.path.exists(path):
                model = xgb.XGBClassifier()
                model.load_model(path)
                obj.models[horizon] = model
        return obj

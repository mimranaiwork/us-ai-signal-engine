"""
Unified prediction interface across all four Phase 2 model types — loads
whichever trained models are available on disk and produces the same
ModelHorizonPrediction shape (see signals/scoring.py) regardless of
architecture, so the ensemble layer never needs to know a model's internals.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from calendar_us.trading_calendar import HORIZON_SESSIONS
from models.common import probs_to_dict
from models.lightgbm.model import LightGBMMultiHorizonModel
from models.lstm.model import LSTMMultiHorizonModel
from models.sequence_common import SEQUENCE_FEATURE_COLUMNS
from models.transformer.model import TransformerMultiHorizonModel
from models.xgboost.model import XGBoostMultiHorizonModel
from signals.scoring import ModelHorizonPrediction
from signals.stacking import SAVE_DIR as STACKING_SAVE_DIR, StackingEnsemble

MODEL_LOADERS = {
    "xgboost": XGBoostMultiHorizonModel,
    "lightgbm": LightGBMMultiHorizonModel,
}
SEQUENCE_MODEL_LOADERS = {
    "lstm": LSTMMultiHorizonModel,
    "transformer": TransformerMultiHorizonModel,
}
STACKING_REPORT_PATH = os.path.join(os.path.dirname(__file__), "saved", "stacking_report.json")


class ModelRegistry:
    """Loads every available trained model once and serves predictions for
    a single (feature_row, feature_sequence) pair across all horizons."""

    def __init__(self):
        self.tree_models = {}
        self.sequence_models = {}
        for name, cls in MODEL_LOADERS.items():
            try:
                self.tree_models[name] = cls.load()
            except FileNotFoundError:
                pass
        for name, cls in SEQUENCE_MODEL_LOADERS.items():
            try:
                self.sequence_models[name] = cls.load()
            except FileNotFoundError:
                pass

        # Stacking meta-model (signals/stacking.py, scripts/train_stacking.py)
        # — optional, only used for the horizons stacking_report.json
        # actually measured it beating the weighted average on held-out data
        # (mixed per-horizon, see that script's docstring); missing entirely
        # is fine, generate_signal() just falls back to weighted-average.
        self.stacker: "StackingEnsemble | None" = None
        self.stacking_horizons: set[str] = set()
        try:
            self.stacker = StackingEnsemble.load(STACKING_SAVE_DIR)
            with open(STACKING_REPORT_PATH) as f:
                report = json.load(f)
            self.stacking_horizons = {h for h, v in report["horizons"].items() if v["winner"] == "stacking"}
        except FileNotFoundError:
            pass

    def available_models(self) -> list[str]:
        return list(self.tree_models) + list(self.sequence_models)

    def predict_all(self, latest_row: pd.Series, feature_sequence: np.ndarray | None) -> list[ModelHorizonPrediction]:
        """
        latest_row: the most recent feature row for one symbol (used by tree models).
        feature_sequence: (seq_len, n_seq_features) array ending at the same date
            (used by LSTM/Transformer); None skips sequence models (e.g. insufficient history).
        """
        predictions = []
        for name, model in self.tree_models.items():
            for horizon in HORIZON_SESSIONS:
                if horizon not in model.models:
                    continue
                probs = model.predict_latest(latest_row, horizon)
                predictions.append(ModelHorizonPrediction(model=name, horizon=horizon, probs=probs))

        if feature_sequence is not None:
            batch = feature_sequence[np.newaxis, :, :]  # (1, seq_len, n_features)
            for name, model in self.sequence_models.items():
                probs_by_horizon = model.predict_proba_batch(batch)
                for horizon, probs in probs_by_horizon.items():
                    predictions.append(
                        ModelHorizonPrediction(model=name, horizon=horizon, probs=probs_to_dict(probs[0]))
                    )

        return predictions

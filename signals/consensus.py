"""
Model consensus (do XGBoost/LightGBM/LSTM/Transformer agree?) and horizon
consensus (do 1D..6M agree?) — both in [0, 1], both computed as
1 - stddev-of-scores, since each model/horizon score already lives in
[-1, +1] so their stddev is naturally bounded and interpretable as
"how much disagreement".
"""

from __future__ import annotations

import numpy as np


def model_consensus_for_horizon(predictions, horizon: str) -> float:
    """predictions: list of ModelHorizonPrediction (see signals/scoring.py) —
    typed loosely here to avoid a circular import with scoring.py."""
    relevant = [p.score for p in predictions if p.horizon == horizon]
    if len(relevant) <= 1:
        return 1.0
    std = float(np.std(relevant))
    return max(0.0, 1.0 - std)  # scores live in [-1,1], so std in [0,1] roughly


def horizon_consensus(horizon_scores: dict[str, float]) -> float:
    """How much the seven horizons agree on direction — high when 1D..6M all
    point the same way, low when short and long horizons disagree."""
    values = list(horizon_scores.values())
    if len(values) <= 1:
        return 1.0
    std = float(np.std(values))
    return max(0.0, 1.0 - std)

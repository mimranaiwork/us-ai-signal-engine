"""
Per-horizon and composite scoring (Phase 3, spec sections 31/34/35).

horizon_score = P(BUY) - P(SELL), in [-1, +1]. This is deliberately a linear
combination of the two "directional" class probabilities — HOLD probability
doesn't enter the score directly, it shows up as *low confidence* instead
(see signals/confidence.py), which is the right place for "the model isn't
sure" to live rather than muddying the directional score itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from calendar_us.trading_calendar import HORIZON_SESSIONS
from models.common import probs_to_dict
from signals.consensus import horizon_consensus, model_consensus_for_horizon

if TYPE_CHECKING:
    from signals.stacking import StackingEnsemble

ModelProbs = dict[str, float]  # {"BUY":.., "HOLD":.., "SELL":..}


def horizon_score(probs: ModelProbs) -> float:
    return float(probs["BUY"] - probs["SELL"])


def score_to_label(score: float, buy_threshold: float, sell_threshold: float) -> str:
    if score >= buy_threshold:
        return "BUY"
    if score <= sell_threshold:
        return "SELL"
    return "HOLD"


@dataclass
class ModelHorizonPrediction:
    model: str
    horizon: str
    probs: ModelProbs

    @property
    def score(self) -> float:
        return horizon_score(self.probs)


@dataclass
class CompositeResult:
    symbol: str
    score: float
    horizon_scores: dict[str, float]
    model_consensus: float
    horizon_consensus: float
    per_model_scores: dict[str, dict[str, float]] = field(default_factory=dict)  # model -> horizon -> score
    stacking_used: list[str] = field(default_factory=list)  # horizons where the stacking meta-model actually ran


def model_weighted_average(
    predictions: list[ModelHorizonPrediction], model_weights: dict[str, float], horizon: str,
) -> ModelProbs:
    """Weighted-average ensemble across models for a single horizon (probability
    averaging — one of the ensemble methods called out in spec section 34;
    stacking/meta-model/logistic-ensemble are Phase 2+ training-time choices
    once there's enough validation history to fit a meta-model on)."""
    relevant = [p for p in predictions if p.horizon == horizon]
    if not relevant:
        raise ValueError(f"No predictions for horizon {horizon}")

    total_weight = sum(model_weights.get(p.model, 0.0) for p in relevant)
    if total_weight <= 0:
        total_weight = len(relevant)
        weights = {p.model: 1.0 for p in relevant}
    else:
        weights = {p.model: model_weights.get(p.model, 0.0) for p in relevant}

    avg = {"BUY": 0.0, "HOLD": 0.0, "SELL": 0.0}
    for p in relevant:
        w = weights[p.model] / total_weight
        for label in avg:
            avg[label] += w * p.probs[label]
    return avg


def stacking_ensemble_proba(
    stacker: "StackingEnsemble", horizon: str, predictions: list[ModelHorizonPrediction],
) -> ModelProbs | None:
    """Builds the {model}_{label} feature row the stacking meta-model was
    trained on (signals/stacking.py) from this horizon's ModelHorizonPredictions
    and runs it through. Returns None (caller falls back to weighted-average)
    if this horizon has no fitted meta-model, or if any base model the
    meta-model was trained on is missing this call (e.g. no feature_sequence
    available for LSTM/Transformer) — a stacking model can't score with an
    incomplete input row the way a weighted average degrades gracefully to."""
    if horizon not in stacker.models:
        return None
    feat_cols = stacker.feature_cols[horizon]
    relevant = {p.model: p for p in predictions if p.horizon == horizon}

    row = {}
    for col in feat_cols:
        model, label = col.rsplit("_", 1)
        if model not in relevant:
            return None
        row[col] = relevant[model].probs[label]

    proba = stacker.predict_proba(horizon, pd.DataFrame([row]))[0]
    return probs_to_dict(proba)


def composite_score(horizon_scores: dict[str, float], horizon_weights: dict[str, float]) -> float:
    total_weight = sum(horizon_weights.get(h, 0.0) for h in horizon_scores)
    if total_weight <= 0:
        return float(np.mean(list(horizon_scores.values())))
    return sum(horizon_scores[h] * horizon_weights.get(h, 0.0) for h in horizon_scores) / total_weight


def build_composite(
    symbol: str,
    predictions: list[ModelHorizonPrediction],
    model_weights: dict[str, float],
    horizon_weights: dict[str, float],
    stacker: "StackingEnsemble | None" = None,
    stacking_horizons: set[str] | None = None,
) -> CompositeResult:
    """`stacker`/`stacking_horizons`: per-horizon opt-in to the stacking
    meta-model (scripts/train_stacking.py) in place of the fixed-weight
    average, for horizons where it was shown to actually win on held-out
    data (models/saved/stacking_report.json) — mixed per-horizon results are
    the real, measured outcome, not every horizon benefits equally. Falls
    back to the weighted average per-horizon whenever stacking isn't
    available or a required base model's prediction is missing."""
    stacking_horizons = stacking_horizons or set()
    horizon_scores = {}
    per_model_scores: dict[str, dict[str, float]] = {}
    stacking_used: list[str] = []
    for horizon in HORIZON_SESSIONS:
        relevant = [p for p in predictions if p.horizon == horizon]
        if not relevant:
            continue

        ensembled_probs = None
        if stacker is not None and horizon in stacking_horizons:
            ensembled_probs = stacking_ensemble_proba(stacker, horizon, predictions)
            if ensembled_probs is not None:
                stacking_used.append(horizon)
        if ensembled_probs is None:
            ensembled_probs = model_weighted_average(predictions, model_weights, horizon)

        horizon_scores[horizon] = horizon_score(ensembled_probs)
        for p in relevant:
            per_model_scores.setdefault(p.model, {})[horizon] = p.score

    model_consensus = float(np.mean([
        model_consensus_for_horizon(predictions, h) for h in horizon_scores
    ])) if horizon_scores else 0.0

    return CompositeResult(
        symbol=symbol,
        score=composite_score(horizon_scores, horizon_weights),
        horizon_scores=horizon_scores,
        model_consensus=model_consensus,
        horizon_consensus=horizon_consensus(horizon_scores),
        per_model_scores=per_model_scores,
        stacking_used=stacking_used,
    )

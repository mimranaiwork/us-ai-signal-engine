"""
Confidence scoring (spec section 37): combines model agreement, horizon
agreement, how far each model's probabilities are from a coin-flip,
market-regime certainty, and data quality/liquidity into one 0-100% number.

Deliberately NOT called "probability of profit" (per the spec's own warning)
— it's a measure of how much the ensemble agrees with itself and how clean
the underlying data is, not a calibrated win-rate. "Historical model
performance" and "prediction calibration" (also listed in section 37) need
backtest history to compute and are wired in once Phase 4's backtesting
engine has produced some — until then they're left out of the weighted sum
rather than faked with a placeholder number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CONFIDENCE_WEIGHTS = {
    "model_consensus": 0.30,
    "horizon_consensus": 0.25,
    "probability_strength": 0.25,
    "regime_certainty": 0.10,
    "data_quality": 0.10,
}


@dataclass
class DataQuality:
    sufficient_history: bool
    sufficient_liquidity: bool
    missing_feature_ratio: float  # 0 = no missing features, 1 = all missing

    @property
    def score(self) -> float:
        if not self.sufficient_history or not self.sufficient_liquidity:
            return 0.0
        return max(0.0, 1.0 - self.missing_feature_ratio)


def probability_strength(all_probs: list[dict[str, float]]) -> float:
    """
    How far each (model, horizon) probability distribution is from uniform
    (1/3, 1/3, 1/3) — a model outputting {BUY:0.9,HOLD:0.05,SELL:0.05} is
    "sure"; one outputting {BUY:0.35,HOLD:0.33,SELL:0.32} is not, even if its
    argmax happens to agree with everyone else's.
    """
    if not all_probs:
        return 0.0
    strengths = []
    for probs in all_probs:
        max_prob = max(probs.values())
        # max_prob ranges [1/3, 1] -> rescale to [0, 1]
        strengths.append((max_prob - 1 / 3) / (1 - 1 / 3))
    return float(np.clip(np.mean(strengths), 0.0, 1.0))


def compute_confidence(
    model_consensus: float,
    horizon_consensus: float,
    all_probs: list[dict[str, float]],
    regime_confidence: float,
    data_quality: DataQuality,
    weights: dict[str, float] = CONFIDENCE_WEIGHTS,
) -> float:
    components = {
        "model_consensus": model_consensus,
        "horizon_consensus": horizon_consensus,
        "probability_strength": probability_strength(all_probs),
        "regime_certainty": regime_confidence,
        "data_quality": data_quality.score,
    }
    total_weight = sum(weights.values())
    confidence = sum(components[k] * weights[k] for k in components) / total_weight
    return float(np.clip(confidence, 0.0, 1.0))

import numpy as np
import pandas as pd

from backtesting.optimize import _signals_from_scores


def test_signals_from_scores_thresholds():
    scores = pd.Series([0.5, -0.5, 0.1, -0.1, 0.25, -0.25])
    signals = _signals_from_scores(scores, buy_threshold=0.25, sell_threshold=-0.25)
    assert list(signals) == ["BUY", "SELL", "HOLD", "HOLD", "BUY", "SELL"]  # boundary is inclusive


def test_signals_from_scores_all_hold_for_wide_thresholds():
    scores = pd.Series([0.1, -0.1, 0.05])
    signals = _signals_from_scores(scores, buy_threshold=0.9, sell_threshold=-0.9)
    assert (signals == "HOLD").all()


def test_horizon_weight_presets_sum_to_one():
    from backtesting.optimize import HORIZON_WEIGHT_PRESETS
    for name, weights in HORIZON_WEIGHT_PRESETS.items():
        assert abs(sum(weights.values()) - 1.0) < 1e-6, f"{name} doesn't sum to 1.0"


def test_model_weight_presets_sum_to_one():
    from backtesting.optimize import MODEL_WEIGHT_PRESETS
    for name, weights in MODEL_WEIGHT_PRESETS.items():
        assert abs(sum(weights.values()) - 1.0) < 1e-6, f"{name} doesn't sum to 1.0"

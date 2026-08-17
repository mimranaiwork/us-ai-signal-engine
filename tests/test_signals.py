import pytest

from signals.confidence import DataQuality, compute_confidence, probability_strength
from signals.consensus import horizon_consensus, model_consensus_for_horizon
from signals.market_regime import detect_market_regime
from signals.scoring import (
    ModelHorizonPrediction,
    build_composite,
    composite_score,
    horizon_score,
    model_weighted_average,
    score_to_label,
)


def test_horizon_score_range():
    assert horizon_score({"BUY": 1.0, "HOLD": 0.0, "SELL": 0.0}) == 1.0
    assert horizon_score({"BUY": 0.0, "HOLD": 0.0, "SELL": 1.0}) == -1.0
    assert horizon_score({"BUY": 0.33, "HOLD": 0.34, "SELL": 0.33}) == pytest.approx(0.0, abs=1e-6)


def test_score_to_label_thresholds():
    assert score_to_label(0.3, 0.25, -0.25) == "BUY"
    assert score_to_label(-0.3, 0.25, -0.25) == "SELL"
    assert score_to_label(0.1, 0.25, -0.25) == "HOLD"
    assert score_to_label(0.25, 0.25, -0.25) == "BUY"  # boundary is inclusive


def test_model_weighted_average_respects_weights():
    predictions = [
        ModelHorizonPrediction("xgboost", "1M", {"BUY": 1.0, "HOLD": 0.0, "SELL": 0.0}),
        ModelHorizonPrediction("lightgbm", "1M", {"BUY": 0.0, "HOLD": 0.0, "SELL": 1.0}),
    ]
    # Equal weights -> exactly halfway.
    avg = model_weighted_average(predictions, {"xgboost": 0.5, "lightgbm": 0.5}, "1M")
    assert avg["BUY"] == pytest.approx(0.5)
    assert avg["SELL"] == pytest.approx(0.5)

    # All weight on xgboost -> its prediction dominates.
    avg2 = model_weighted_average(predictions, {"xgboost": 1.0, "lightgbm": 0.0}, "1M")
    assert avg2["BUY"] == pytest.approx(1.0)
    assert avg2["SELL"] == pytest.approx(0.0)


def test_model_consensus_high_when_models_agree():
    predictions = [
        ModelHorizonPrediction("xgboost", "1M", {"BUY": 0.8, "HOLD": 0.1, "SELL": 0.1}),
        ModelHorizonPrediction("lightgbm", "1M", {"BUY": 0.75, "HOLD": 0.15, "SELL": 0.1}),
    ]
    consensus = model_consensus_for_horizon(predictions, "1M")
    assert consensus > 0.9


def test_model_consensus_low_when_models_disagree():
    predictions = [
        ModelHorizonPrediction("xgboost", "1M", {"BUY": 0.9, "HOLD": 0.05, "SELL": 0.05}),
        ModelHorizonPrediction("lightgbm", "1M", {"BUY": 0.05, "HOLD": 0.05, "SELL": 0.9}),
    ]
    consensus = model_consensus_for_horizon(predictions, "1M")
    assert consensus < 0.3


def test_horizon_consensus_all_agree_vs_disagree():
    agreeing = {"1D": 0.5, "1W": 0.55, "2W": 0.52, "1M": 0.48}
    disagreeing = {"1D": 0.8, "1W": -0.8, "2W": 0.1, "1M": -0.5}
    assert horizon_consensus(agreeing) > horizon_consensus(disagreeing)


def test_composite_score_weighted_correctly():
    scores = {"1D": 1.0, "1M": -1.0}
    weights = {"1D": 0.75, "1M": 0.25}
    # Weighted average: 1.0*0.75 + (-1.0)*0.25 = 0.5
    assert composite_score(scores, weights) == pytest.approx(0.5)


def test_build_composite_end_to_end():
    predictions = [
        ModelHorizonPrediction("xgboost", "1D", {"BUY": 0.6, "HOLD": 0.2, "SELL": 0.2}),
        ModelHorizonPrediction("lightgbm", "1D", {"BUY": 0.55, "HOLD": 0.25, "SELL": 0.2}),
        ModelHorizonPrediction("xgboost", "1M", {"BUY": 0.7, "HOLD": 0.2, "SELL": 0.1}),
        ModelHorizonPrediction("lightgbm", "1M", {"BUY": 0.65, "HOLD": 0.2, "SELL": 0.15}),
    ]
    model_weights = {"xgboost": 0.5, "lightgbm": 0.5}
    horizon_weights = {"1D": 0.5, "1M": 0.5}

    result = build_composite("TEST", predictions, model_weights, horizon_weights)
    assert result.symbol == "TEST"
    assert set(result.horizon_scores) == {"1D", "1M"}
    assert result.model_consensus > 0.8  # both models roughly agree here
    assert -1 <= result.score <= 1
    assert result.stacking_used == []  # no stacker passed -> pure weighted average


def _fitted_stacker():
    import pandas as pd

    from signals.stacking import StackingEnsemble

    train_df = pd.DataFrame({
        "symbol": "TEST", "date": pd.date_range("2024-01-01", periods=20),
        "label": ["BUY"] * 10 + ["SELL"] * 10,
        "xgboost_BUY": [0.9] * 10 + [0.1] * 10, "xgboost_HOLD": [0.05] * 20, "xgboost_SELL": [0.05] * 10 + [0.9] * 10,
        "lightgbm_BUY": [0.8] * 10 + [0.2] * 10, "lightgbm_HOLD": [0.1] * 20, "lightgbm_SELL": [0.1] * 10 + [0.7] * 10,
    })
    stacker = StackingEnsemble()
    stacker.fit_horizon("1D", train_df)
    return stacker


def test_build_composite_uses_stacking_for_opted_in_horizon():
    predictions = [
        ModelHorizonPrediction("xgboost", "1D", {"BUY": 0.9, "HOLD": 0.05, "SELL": 0.05}),
        ModelHorizonPrediction("lightgbm", "1D", {"BUY": 0.8, "HOLD": 0.1, "SELL": 0.1}),
        ModelHorizonPrediction("xgboost", "1M", {"BUY": 0.9, "HOLD": 0.05, "SELL": 0.05}),
        ModelHorizonPrediction("lightgbm", "1M", {"BUY": 0.8, "HOLD": 0.1, "SELL": 0.1}),
    ]
    stacker = _fitted_stacker()

    result = build_composite(
        "TEST", predictions, {"xgboost": 0.5, "lightgbm": 0.5}, {"1D": 0.5, "1M": 0.5},
        stacker=stacker, stacking_horizons={"1D"},
    )
    # 1D was opted into stacking and the stacker has a fitted "1D" model -> used.
    assert result.stacking_used == ["1D"]
    # 1M was NOT opted in, even though predictions exist -> falls back untouched.
    assert "1M" not in result.stacking_used


def test_build_composite_falls_back_when_stacker_missing_a_base_model():
    # Stacker was trained on xgboost+lightgbm; only xgboost is present here.
    predictions = [ModelHorizonPrediction("xgboost", "1D", {"BUY": 0.9, "HOLD": 0.05, "SELL": 0.05})]
    stacker = _fitted_stacker()

    result = build_composite(
        "TEST", predictions, {"xgboost": 1.0}, {"1D": 1.0},
        stacker=stacker, stacking_horizons={"1D"},
    )
    assert result.stacking_used == []  # gracefully fell back to weighted average
    assert "1D" in result.horizon_scores  # still produced a score


def test_probability_strength_sure_vs_unsure():
    sure = [{"BUY": 0.9, "HOLD": 0.05, "SELL": 0.05}]
    unsure = [{"BUY": 0.34, "HOLD": 0.33, "SELL": 0.33}]
    assert probability_strength(sure) > probability_strength(unsure)
    assert probability_strength(unsure) == pytest.approx(0.0, abs=0.02)


def test_confidence_low_when_everything_uncertain():
    poor_quality = DataQuality(sufficient_history=False, sufficient_liquidity=True, missing_feature_ratio=0.0)
    confidence = compute_confidence(
        model_consensus=0.1, horizon_consensus=0.1,
        all_probs=[{"BUY": 0.34, "HOLD": 0.33, "SELL": 0.33}],
        regime_confidence=0.1, data_quality=poor_quality,
    )
    assert confidence < 0.3


def test_confidence_high_when_everything_agrees():
    good_quality = DataQuality(sufficient_history=True, sufficient_liquidity=True, missing_feature_ratio=0.0)
    confidence = compute_confidence(
        model_consensus=1.0, horizon_consensus=1.0,
        all_probs=[{"BUY": 0.9, "HOLD": 0.05, "SELL": 0.05}],
        regime_confidence=1.0, data_quality=good_quality,
    )
    assert confidence > 0.85


def test_market_regime_bull_when_all_benchmarks_up():
    inputs = {
        "SPY": {"trend": "UP", "volatility_20d": 0.15},
        "QQQ": {"trend": "UP", "volatility_20d": 0.18},
        "IWM": {"trend": "UP", "volatility_20d": 0.20},
    }
    result = detect_market_regime(inputs)
    assert result.regime == "BULL"
    assert result.confidence == pytest.approx(1.0)


def test_market_regime_bear_when_all_benchmarks_down():
    inputs = {
        "SPY": {"trend": "DOWN", "volatility_20d": 0.15},
        "QQQ": {"trend": "DOWN", "volatility_20d": 0.18},
        "IWM": {"trend": "DOWN", "volatility_20d": 0.20},
    }
    result = detect_market_regime(inputs)
    assert result.regime == "BEAR"


def test_market_regime_high_volatility_overrides_trend():
    inputs = {
        "SPY": {"trend": "UP", "volatility_20d": 0.50},
        "QQQ": {"trend": "UP", "volatility_20d": 0.55},
        "IWM": {"trend": "UP", "volatility_20d": 0.60},
    }
    result = detect_market_regime(inputs)
    assert result.regime == "HIGH_VOLATILITY"


def test_market_regime_sideways_when_mixed():
    inputs = {
        "SPY": {"trend": "UP", "volatility_20d": 0.15},
        "QQQ": {"trend": "DOWN", "volatility_20d": 0.15},
        "IWM": {"trend": "FLAT", "volatility_20d": 0.15},
    }
    result = detect_market_regime(inputs)
    assert result.regime == "SIDEWAYS"


def test_market_regime_breadth_divergence_cuts_confidence():
    """Index-level BULL (all benchmarks up) while market breadth is
    strongly negative (most individual stocks actually declining) is a
    real 'narrow market' warning — must reduce confidence, not silently
    be ignored."""
    inputs = {
        "SPY": {"trend": "UP", "volatility_20d": 0.15},
        "QQQ": {"trend": "UP", "volatility_20d": 0.18},
        "IWM": {"trend": "UP", "volatility_20d": 0.20},
    }
    agreeing = detect_market_regime(inputs, market_breadth_score=0.3)
    diverging = detect_market_regime(inputs, market_breadth_score=-0.4)

    assert agreeing.regime == "BULL"
    assert diverging.regime == "BULL"  # still BULL — breadth informs confidence, doesn't override
    assert diverging.confidence < agreeing.confidence
    assert diverging.detail.get("breadth_divergence") is True
    assert "breadth_divergence" not in agreeing.detail


def test_market_regime_breadth_agreement_does_not_inflate_confidence():
    inputs = {
        "SPY": {"trend": "UP", "volatility_20d": 0.15},
        "QQQ": {"trend": "UP", "volatility_20d": 0.18},
        "IWM": {"trend": "UP", "volatility_20d": 0.20},
    }
    without_breadth = detect_market_regime(inputs)
    with_agreeing_breadth = detect_market_regime(inputs, market_breadth_score=0.5)
    assert with_agreeing_breadth.confidence == pytest.approx(without_breadth.confidence)


def test_market_regime_breadth_ignored_for_sideways_and_high_vol():
    """Breadth adjustment only applies to a directional BULL/BEAR read —
    SIDEWAYS/HIGH_VOLATILITY aren't a 'trend' breadth can confirm or deny."""
    mixed_inputs = {
        "SPY": {"trend": "UP", "volatility_20d": 0.15},
        "QQQ": {"trend": "DOWN", "volatility_20d": 0.15},
        "IWM": {"trend": "FLAT", "volatility_20d": 0.15},
    }
    result = detect_market_regime(mixed_inputs, market_breadth_score=-0.9)
    assert result.regime == "SIDEWAYS"
    assert "breadth_divergence" not in result.detail

import numpy as np
import pandas as pd
import pytest

from monitoring.drift import (
    detect_feature_drift,
    detect_prediction_drift,
    population_stability_index,
    summarize_drift,
)


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    data = pd.Series(rng.normal(0, 1, 1000))
    psi = population_stability_index(data, data)
    assert psi == pytest.approx(0.0, abs=1e-9)


def test_psi_low_for_similar_distributions():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(0, 1, 2000))
    current = pd.Series(rng.normal(0.05, 1.02, 500))  # tiny shift
    psi = population_stability_index(reference, current)
    assert psi < 0.1


def test_psi_high_for_shifted_distribution():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(0, 1, 2000))
    current = pd.Series(rng.normal(5, 1, 500))  # large shift
    psi = population_stability_index(reference, current)
    assert psi > 0.25


def test_psi_high_for_shifted_variance():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(0, 1, 2000))
    current = pd.Series(rng.normal(0, 5, 500))  # same mean, much wider spread
    psi = population_stability_index(reference, current)
    assert psi > 0.1


def test_detect_feature_drift_flags_shifted_column():
    rng = np.random.default_rng(0)
    reference_df = pd.DataFrame({
        "stable_feature": rng.normal(0, 1, 1000),
        "drifted_feature": rng.normal(0, 1, 1000),
    })
    current_df = pd.DataFrame({
        "stable_feature": rng.normal(0, 1, 200),
        "drifted_feature": rng.normal(10, 1, 200),
    })
    results = detect_feature_drift(reference_df, current_df, ["stable_feature", "drifted_feature"])
    by_name = {r.feature: r for r in results}
    assert by_name["stable_feature"].status == "STABLE"
    assert by_name["drifted_feature"].status == "ALERT"


def test_detect_feature_drift_skips_missing_columns():
    reference_df = pd.DataFrame({"a": [1, 2, 3]})
    current_df = pd.DataFrame({"a": [1, 2, 3]})
    results = detect_feature_drift(reference_df, current_df, ["a", "not_present"])
    assert len(results) == 1
    assert results[0].feature == "a"


def test_detect_prediction_drift():
    rng = np.random.default_rng(0)
    reference_scores = pd.Series(rng.uniform(-0.2, 0.2, 1000))
    current_scores = pd.Series(rng.uniform(0.6, 0.9, 200))  # model suddenly much more bullish
    result = detect_prediction_drift(reference_scores, current_scores)
    assert result.feature == "composite_score"
    assert result.status == "ALERT"


def test_summarize_drift():
    from monitoring.drift import DriftResult
    results = [
        DriftResult("a", 0.05, "STABLE"),
        DriftResult("b", 0.15, "WATCH"),
        DriftResult("c", 0.30, "ALERT"),
    ]
    summary = summarize_drift(results)
    assert summary["n_features_checked"] == 3
    assert summary["n_alert"] == 1
    assert summary["n_watch"] == 1
    assert summary["alert_features"] == ["c"]
    assert summary["max_psi"] == pytest.approx(0.30)

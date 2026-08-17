"""Regression test for the SHAP pred_contribs axis-order bug found while
wiring signal_engine.py: XGBoost's multi-class pred_contribs returns shape
(n_rows, n_classes, n_features+1) — indexing it as (n_rows, n_features+1,
n_classes) silently produced a 2-element series instead of raising, which is
exactly the kind of bug that's easy to miss without a shape assertion."""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from explainability.shap import shap_contributions, top_reasons_and_risks


@pytest.fixture(scope="module")
def toy_model():
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame({
        "price_vs_sma50": rng.normal(0, 0.05, n),
        "rsi_14": rng.uniform(20, 80, n),
        "volatility_20d": rng.uniform(0.1, 0.4, n),
    })
    # Deterministic-ish labels so the model actually learns something.
    y = np.where(X["price_vs_sma50"] > 0.02, 2, np.where(X["price_vs_sma50"] < -0.02, 0, 1))
    model = xgb.XGBClassifier(n_estimators=20, max_depth=3, objective="multi:softprob", num_class=3)
    model.fit(X, y)
    return model, X


def test_shap_contributions_shape_matches_features(toy_model):
    model, X = toy_model
    contributions = shap_contributions(model, X.iloc[[0]], class_index=2)
    assert len(contributions) == len(X.columns)
    assert list(contributions.index) == list(X.columns)


def test_shap_contributions_differ_by_class(toy_model):
    model, X = toy_model
    row = X.iloc[[0]]
    buy_contribs = shap_contributions(model, row, class_index=2)
    sell_contribs = shap_contributions(model, row, class_index=0)
    assert not buy_contribs.equals(sell_contribs)


def test_top_reasons_and_risks_returns_known_phrases(toy_model):
    model, X = toy_model
    reasons, risks = top_reasons_and_risks(model, X.iloc[[0]], buy_class_index=2)
    assert isinstance(reasons, list)
    assert isinstance(risks, list)
    # Every returned reason should be a real phrase from the fixed table, not
    # a fabricated string — check membership against the known phrase set.
    from explainability.shap import REASON_PHRASES
    known_phrases = {p for pair in REASON_PHRASES.values() for p in pair}
    assert all(r in known_phrases for r in reasons)

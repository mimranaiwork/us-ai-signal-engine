import numpy as np
import pandas as pd
import pytest

from signals.stacking import StackingEnsemble, prob_columns


def _synthetic_predictions_df(n=200, seed=0):
    """Two base models' BUY/HOLD/SELL probs + a true label that's mostly
    driven by model_a (so the meta-model should learn to weight it higher)."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(["SELL", "HOLD", "BUY"], size=n, p=[0.3, 0.4, 0.3])

    def _probs_biased_toward(true_labels, strength):
        out = {"BUY": [], "HOLD": [], "SELL": []}
        for lab in true_labels:
            base = {"BUY": 0.2, "HOLD": 0.2, "SELL": 0.2}
            base[lab] += strength
            total = sum(base.values())
            for k in base:
                out[k].append(base[k] / total)
        return out

    model_a = _probs_biased_toward(labels, strength=0.6)   # accurate
    model_b = _probs_biased_toward(labels, strength=0.05)  # near-random

    df = pd.DataFrame({"symbol": "TEST", "date": pd.date_range("2024-01-01", periods=n), "label": labels})
    for name, probs in (("model_a", model_a), ("model_b", model_b)):
        for lab in ("BUY", "HOLD", "SELL"):
            df[f"{name}_{lab}"] = probs[lab]
    return df


def test_prob_columns_excludes_identifier_columns():
    df = _synthetic_predictions_df(n=10)
    cols = prob_columns(df)
    assert "symbol" not in cols
    assert "date" not in cols
    assert "label" not in cols
    assert "model_a_BUY" in cols and "model_b_SELL" in cols


def test_fit_horizon_learns_from_accurate_base_model():
    train_df = _synthetic_predictions_df(n=300, seed=1)
    eval_df = _synthetic_predictions_df(n=200, seed=2)

    stacker = StackingEnsemble()
    fit_info = stacker.fit_horizon("1D", train_df)
    assert fit_info["train_accuracy"] > 0.6

    proba = stacker.predict_proba("1D", eval_df)
    assert proba.shape == (len(eval_df), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    from models.common import LABEL_TO_IDX
    y_eval = eval_df["label"].map(LABEL_TO_IDX).to_numpy()
    pred = np.argmax(proba, axis=1)
    accuracy = (pred == y_eval).mean()
    assert accuracy > 0.6  # meta-model should exploit model_a's signal


def test_save_and_load_roundtrip(tmp_path):
    train_df = _synthetic_predictions_df(n=100, seed=3)
    stacker = StackingEnsemble()
    stacker.fit_horizon("1D", train_df)
    stacker.fit_horizon("1W", train_df)

    save_dir = str(tmp_path / "stacking")
    stacker.save(save_dir)

    loaded = StackingEnsemble.load(save_dir)
    assert set(loaded.models.keys()) == {"1D", "1W"}

    eval_df = _synthetic_predictions_df(n=50, seed=4)
    original_proba = stacker.predict_proba("1D", eval_df)
    loaded_proba = loaded.predict_proba("1D", eval_df)
    np.testing.assert_allclose(original_proba, loaded_proba)


def test_predict_proba_uses_feature_cols_captured_at_fit_time():
    train_df = _synthetic_predictions_df(n=100, seed=5)
    stacker = StackingEnsemble()
    stacker.fit_horizon("1D", train_df)

    # eval_df has an extra, unrelated numeric column — predict_proba must
    # still only use the columns captured at fit time, not re-derive them.
    eval_df = _synthetic_predictions_df(n=30, seed=6)
    eval_df["unrelated_extra_col"] = 1.0
    proba = stacker.predict_proba("1D", eval_df)
    assert proba.shape == (len(eval_df), 3)


def test_predict_proba_correctly_aligned_when_a_class_missing_from_training():
    """If a horizon's meta-train split happens to have no SELL rows at all
    (label imbalance is real on this project's data — see
    config/settings.yaml's signal_thresholds comment on why SELL rarely
    fires), sklearn's classes_ shrinks to just [HOLD, BUY] and predict_proba
    returns only 2 columns. That must still land in the right SELL/HOLD/BUY
    slots (index 0/1/2 per models.common.LABEL_TO_IDX), not silently
    misaligned into a 2-wide array."""
    train_df = pd.DataFrame({
        "symbol": "TEST", "date": pd.date_range("2024-01-01", periods=20),
        "label": ["BUY"] * 10 + ["HOLD"] * 10,  # no SELL at all
        "model_a_BUY": [0.9] * 10 + [0.1] * 10,
        "model_a_HOLD": [0.05] * 10 + [0.8] * 10,
        "model_a_SELL": [0.05] * 20,
    })
    stacker = StackingEnsemble()
    stacker.fit_horizon("1D", train_df)

    eval_df = pd.DataFrame({
        "model_a_BUY": [0.9], "model_a_HOLD": [0.05], "model_a_SELL": [0.05],
    })
    proba = stacker.predict_proba("1D", eval_df)
    assert proba.shape == (1, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert proba[0, 0] == 0.0  # SELL (index 0) never seen at fit time -> exactly 0, not misaligned garbage

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtesting.walk_forward import WalkForwardFold, evaluate_fold_sequence
from models.lstm.model import LSTMMultiHorizonModel
from models.sequence_common import SEQUENCE_FEATURE_COLUMNS


@pytest.fixture
def processed_dir(tmp_path, monkeypatch):
    import models.sequence_common as seq_mod

    monkeypatch.setattr(seq_mod, "PROCESSED_DIR", str(tmp_path))

    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)

    features = pd.DataFrame({"date": dates})
    for col in SEQUENCE_FEATURE_COLUMNS:
        features[col] = rng.normal(0, 1, n)
    features.to_csv(tmp_path / "TEST_features.csv", index=False)

    # Only "1D" has real labels; the other six horizons are entirely absent
    # (build_sequences must still work — MISSING_LABEL for those heads).
    labels = pd.DataFrame({
        "date": dates,
        "horizon": ["1D"] * n,
        "label": np.where(rng.random(n) > 0.5, "BUY", "SELL"),
    })
    labels.to_csv(tmp_path / "TEST_labels.csv", index=False)

    return str(tmp_path), dates


def test_evaluate_fold_sequence_trains_and_scores_held_out_test_window(processed_dir):
    _, dates = processed_dir
    fold = WalkForwardFold(
        train_start=dates[0].date(),
        train_end=dates[140].date(),
        val_end=dates[170].date(),
        test_end=(dates[-1] + pd.Timedelta(days=1)).date(),
    )

    result = evaluate_fold_sequence(
        LSTMMultiHorizonModel, ["TEST"], "1D", fold, seq_len=5, epochs=1, hidden_size=8,
    )

    assert "error" not in result
    assert result["horizon"] == "1D"
    assert result["architecture"] == "LSTM"
    assert result["n_test"] > 0
    assert 0.0 <= result["test_classification"]["accuracy"] <= 1.0
    assert result["fold"]["train_start"] == str(fold.train_start)


def test_evaluate_fold_sequence_reports_error_when_fold_too_small(processed_dir):
    _, dates = processed_dir
    # A fold whose test window is a single day can't have >= 20 valid rows.
    tiny_fold = WalkForwardFold(
        train_start=dates[0].date(),
        train_end=dates[10].date(),
        val_end=dates[15].date(),
        test_end=dates[16].date(),
    )
    result = evaluate_fold_sequence(
        LSTMMultiHorizonModel, ["TEST"], "1D", tiny_fold, seq_len=5, epochs=1, hidden_size=8,
    )
    assert "error" in result

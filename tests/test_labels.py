from datetime import date

import numpy as np
import pandas as pd
import pytest

from calendar_us.trading_calendar import HORIZON_SESSIONS
from features.price.price_features import add_price_features
from features.technical.indicators import add_all_technical_indicators
from ingestion.ohlcv_source import SampleCSVSource
from labels import one_day, one_month
from labels.horizons import LabelConfig, generate_all_horizons, generate_labels


@pytest.fixture(scope="module")
def features_df():
    src = SampleCSVSource()
    bars = src.get_bars("AAPL", date(2018, 1, 1), date(2026, 8, 8))
    df = add_price_features(bars)
    df = add_all_technical_indicators(df)
    return df


@pytest.mark.parametrize("horizon,sessions", list(HORIZON_SESSIONS.items()))
def test_future_return_is_exact_row_shift(features_df, horizon, sessions):
    labels = generate_labels(features_df, horizon, LabelConfig(method="fixed", fixed_threshold=0.02))
    expected = features_df["adj_close"].shift(-sessions) / features_df["adj_close"] - 1
    pd.testing.assert_series_equal(
        labels["future_return"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


@pytest.mark.parametrize("horizon,sessions", list(HORIZON_SESSIONS.items()))
def test_tail_rows_have_no_future_return(features_df, horizon, sessions):
    """The last `sessions` rows have no close[T+horizon] available and must be NaN
    — a model that produces a non-NaN label there would be inventing future data."""
    labels = generate_labels(features_df, horizon, LabelConfig(method="fixed"))
    tail = labels["future_return"].iloc[-sessions:]
    assert tail.isna().all()
    assert labels["future_return"].iloc[: -sessions].notna().all()


def test_target_session_date_is_correct_future_date(features_df):
    labels = generate_labels(features_df, "1W", LabelConfig(method="fixed"))
    valid = labels.dropna(subset=["target_session_date"]).reset_index(drop=True)
    dates = features_df["date"].reset_index(drop=True)
    # target_session_date for row i must equal the actual date 5 rows later.
    for i in range(0, len(valid) - 5, 500):  # sample, not exhaustive, for speed
        assert valid.loc[i, "target_session_date"] == dates.iloc[i + 5]


def test_fixed_method_labels_symmetric_threshold(features_df):
    labels = generate_labels(features_df, "1M", LabelConfig(method="fixed", fixed_threshold=0.05))
    valid = labels.dropna(subset=["future_return"])
    assert (valid.loc[valid["label"] == "BUY", "future_return"] > 0.05).all()
    assert (valid.loc[valid["label"] == "SELL", "future_return"] < -0.05).all()
    hold = valid.loc[valid["label"] == "HOLD", "future_return"]
    assert ((hold >= -0.05) & (hold <= 0.05)).all()


def test_all_three_classes_present_for_each_horizon(features_df):
    labels = generate_all_horizons(features_df, LabelConfig(method="volatility"))
    for horizon in HORIZON_SESSIONS:
        classes = set(labels.loc[labels["horizon"] == horizon, "label"].dropna().unique())
        assert classes == {"BUY", "HOLD", "SELL"}, f"{horizon} missing classes: {classes}"


def test_naive_price_comparison_would_be_noisier_than_thresholded():
    """Documents *why* threshold-based labels exist: a naive
    future_price > current_price rule flips on every tiny move, producing far
    more BUY/SELL flips than a volatility-thresholded rule on the same data."""
    rng = np.random.default_rng(0)
    noise_returns = rng.normal(0, 0.001, size=1000)  # tiny day-to-day noise
    naive_labels = np.where(noise_returns > 0, "BUY", "SELL")  # no HOLD possible at all
    thresholded = np.where(noise_returns > 0.02, "BUY", np.where(noise_returns < -0.02, "SELL", "HOLD"))
    assert (naive_labels != "HOLD").all()  # naive rule can never say HOLD
    assert (thresholded == "HOLD").mean() > 0.9  # thresholded rule correctly calls tiny noise HOLD


def test_per_horizon_module_matches_generic_generator(features_df):
    generic = generate_labels(features_df, "1D", LabelConfig())
    via_module = one_day.generate(features_df, LabelConfig())
    pd.testing.assert_frame_equal(generic, via_module)

    generic_1m = generate_labels(features_df, "1M", LabelConfig())
    via_module_1m = one_month.generate(features_df, LabelConfig())
    pd.testing.assert_frame_equal(generic_1m, via_module_1m)

"""
Automated proof that features computed at row T never depend on data after T.

Method: compute features on the full price history, then recompute on a
series truncated to end at T. If any feature function peeked at future rows
(directly, or via a centered/non-causal rolling window), the value at row T
would differ between the two runs. Every feature/indicator column is checked.

This is the general-purpose version of the leakage check the spec calls out
in section 46; it subsumes manual checks for any single indicator since it
compares *every* produced column automatically.
"""

from datetime import date

import pandas as pd
import pytest

from features.price.price_features import add_price_features
from features.technical.indicators import add_all_technical_indicators
from ingestion.ohlcv_source import SampleCSVSource

CUTOFF_ROW = 400  # well past every lookback window (max is 252 for volatility_percentile)


@pytest.fixture(scope="module")
def raw_bars():
    src = SampleCSVSource()
    return src.get_bars("AAPL", date(2018, 1, 1), date(2026, 8, 8)).reset_index(drop=True)


def _build_features(bars: pd.DataFrame) -> pd.DataFrame:
    df = add_price_features(bars)
    df = add_all_technical_indicators(df)
    return df


def test_features_at_T_identical_with_or_without_future_rows(raw_bars):
    full = _build_features(raw_bars)
    truncated = _build_features(raw_bars.iloc[: CUTOFF_ROW + 1])

    row_full = full.iloc[CUTOFF_ROW]
    row_truncated = truncated.iloc[CUTOFF_ROW]

    non_feature_cols = {"date", "open", "high", "low", "close", "adj_close", "volume"}
    feature_cols = [c for c in full.columns if c not in non_feature_cols]

    mismatches = []
    for col in feature_cols:
        a, b = row_full[col], row_truncated[col]
        both_nan = pd.isna(a) and pd.isna(b)
        if both_nan:
            continue
        if pd.isna(a) or pd.isna(b) or abs(float(a) - float(b)) > 1e-9:
            mismatches.append((col, a, b))

    assert not mismatches, (
        f"Feature(s) differ depending on whether future rows exist — this means "
        f"they leak future information: {mismatches}"
    )


def test_removing_a_single_future_row_does_not_change_past_features(raw_bars):
    """A stronger, cheaper variant: drop just the very last row and confirm
    every other row's features are byte-for-byte unchanged."""
    full = _build_features(raw_bars)
    minus_one = _build_features(raw_bars.iloc[:-1])

    common_len = len(minus_one)
    non_feature_cols = {"date", "open", "high", "low", "close", "adj_close", "volume"}
    feature_cols = [c for c in full.columns if c not in non_feature_cols]

    pd.testing.assert_frame_equal(
        full.iloc[:common_len][feature_cols].reset_index(drop=True),
        minus_one[feature_cols].reset_index(drop=True),
        check_exact=False,
        rtol=1e-9,
    )


def test_labels_use_future_data_only_up_to_their_own_horizon(raw_bars):
    """Labels are *supposed* to use future data (that's the target) — but
    changing bars beyond T+horizon must not change the label at T. We perturb
    a far-future row and confirm every label unaffected by that row is
    unchanged."""
    from labels.horizons import LabelConfig, generate_labels

    df = _build_features(raw_bars)
    labels_before = generate_labels(df, "1M", LabelConfig(method="fixed", fixed_threshold=0.02))

    perturbed_bars = raw_bars.copy()
    perturbed_bars.loc[perturbed_bars.index[-1], "adj_close"] *= 1.5  # shock the very last row
    df_perturbed = _build_features(perturbed_bars)
    labels_after = generate_labels(df_perturbed, "1M", LabelConfig(method="fixed", fixed_threshold=0.02))

    horizon_sessions = 21
    unaffected_cutoff = len(raw_bars) - 1 - horizon_sessions  # rows whose target < the perturbed row
    pd.testing.assert_series_equal(
        labels_before["label"].iloc[:unaffected_cutoff].reset_index(drop=True),
        labels_after["label"].iloc[:unaffected_cutoff].reset_index(drop=True),
    )

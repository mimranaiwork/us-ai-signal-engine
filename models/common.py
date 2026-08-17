"""
Shared plumbing for all four Phase 2 model types: canonical feature columns,
label encoding, and pooled (cross-sectional, all-symbols-at-once) dataset
construction from Phase 1's data/processed/{symbol}_features.csv /
{symbol}_labels.csv.

Pooled training (one model per horizon, trained across the whole universe)
rather than one model per stock is deliberate — it's what actually scales to
"thousands of US stocks" per the spec, and lets the model learn general
cross-sectional patterns (e.g. "RSI>70 + below-SMA200" behaves similarly
across tickers) instead of overfitting 40 separate tiny per-symbol datasets.
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd

from calendar_us.trading_calendar import HORIZON_SESSIONS
from universe.universe_loader import Universe

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

LABEL_ORDER = ["SELL", "HOLD", "BUY"]  # index 0,1,2 — ordinal, matches score = P(BUY) - P(SELL) intuition
LABEL_TO_IDX = {l: i for i, l in enumerate(LABEL_ORDER)}

# Non-feature / identifier / leakage-risk columns to exclude from X.
# (`date` is an identifier; raw OHLC are dominated by price level, not
# informative on their own — the model sees adj_close-derived ratios/returns
# instead, which is what actually generalizes across tickers of different price.)
EXCLUDE_COLUMNS = {
    "date", "open", "high", "low", "close", "adj_close", "volume",
    "SPY_trend", "QQQ_trend", "IWM_trend", "DIA_trend",  # categorical strings, handled separately if needed
    # Raw pivot-point price levels (features/technical/pivot_points.py) — same
    # reasoning as raw OHLC above: these are dominated by price level, not
    # informative on their own across tickers. signal_engine.py reads them
    # directly off the latest row for the "levels"/breakout response instead
    # of training on them.
    "pivot", "resistance_1", "resistance_2", "resistance_3",
    "support_1", "support_2", "support_3",
}

# industry_* columns are NaN for every row of a symbol whose industry has
# fewer than 2 peers in the universe (see features/industry/industry_features.py) —
# a structural "no peer group" gap, not a transient warmup NaN like the other
# feature columns. Dropping on them like everything else would silently
# remove ~40% of symbols (AAPL, AMZN, TSLA, ... — anything without >=1 peer)
# from every pooled training set. Filled with 0 (neutral "no relative signal")
# instead so those symbols keep training on their other features.
INDUSTRY_NO_PEER_FILL_COLUMNS = {
    "industry_return", "n_peers", "industry_volatility", "industry_momentum", "stock_vs_industry_1d",
}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]


def load_symbol_frame(symbol: str) -> pd.DataFrame:
    feat_path = os.path.join(PROCESSED_DIR, f"{symbol}_features.csv")
    lbl_path = os.path.join(PROCESSED_DIR, f"{symbol}_labels.csv")
    if not (os.path.exists(feat_path) and os.path.exists(lbl_path)):
        raise FileNotFoundError(
            f"No processed data for {symbol}. Run scripts/build_phase1_dataset.py first."
        )
    features = pd.read_csv(feat_path, parse_dates=["date"])
    labels = pd.read_csv(lbl_path, parse_dates=["date", "target_session_date"])
    features["symbol"] = symbol
    labels["symbol"] = symbol
    return features, labels


def build_pooled_dataset(symbols: list[str], horizon: str) -> pd.DataFrame:
    """
    Returns one row per (symbol, date) with feature columns + a `label` column
    (SELL/HOLD/BUY) for the given horizon, pooled across all symbols. Rows
    with NaN features (warmup period) or NaN label (tail with no future data,
    or label undefined) are dropped.
    """
    if horizon not in HORIZON_SESSIONS:
        raise ValueError(f"Unknown horizon '{horizon}'")

    frames = []
    for symbol in symbols:
        try:
            features, labels = load_symbol_frame(symbol)
        except FileNotFoundError:
            continue
        horizon_labels = labels[labels["horizon"] == horizon][["date", "label", "symbol"]]
        merged = features.merge(horizon_labels, on=["date", "symbol"], how="inner")
        frames.append(merged)

    if not frames:
        raise RuntimeError(f"No data found for any symbol for horizon {horizon}")

    pooled = pd.concat(frames, ignore_index=True)
    fill_cols = [c for c in INDUSTRY_NO_PEER_FILL_COLUMNS if c in pooled.columns]
    pooled[fill_cols] = pooled[fill_cols].fillna(0.0)
    feat_cols = feature_columns(pooled)
    pooled = pooled.dropna(subset=feat_cols + ["label"])
    return pooled.sort_values("date").reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame, train_end: date, val_end: date
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simple time-based split (train < train_end <= val < val_end <= test).
    Phase 2's training baseline — Phase 4 replaces this with a full
    walk-forward harness (multiple rolling train/val/test windows)."""
    d = df["date"].dt.date
    train = df[d < train_end]
    val = df[(d >= train_end) & (d < val_end)]
    test = df[d >= val_end]
    return train, val, test


def default_universe_symbols() -> list[str]:
    return Universe().all_symbols(active_only=True)


def encode_labels(label_series: pd.Series) -> np.ndarray:
    return label_series.map(LABEL_TO_IDX).to_numpy()


def probs_to_dict(probs: np.ndarray) -> dict[str, float]:
    """probs: array of length 3 in LABEL_ORDER order -> {"BUY":..,"HOLD":..,"SELL":..}."""
    return {label: float(probs[idx]) for label, idx in LABEL_TO_IDX.items()}

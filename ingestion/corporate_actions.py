"""
Corporate-action handling: load the corporate_actions table/CSV and validate
that adjusted close prices don't contain unexplained jumps consistent with an
un-adjusted split (the single most common source of "fake" 2x/4x/etc. returns
in naive backtests).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

CORPORATE_ACTIONS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "market", "sample_corporate_actions.csv"
)

# A single-session adjusted-close move bigger than this, with no matching
# corporate-action record on file, is flagged as suspicious (likely an
# un-adjusted split/reverse-split slipping through).
SUSPICIOUS_JUMP_THRESHOLD = 0.35


def load_corporate_actions(path: str = CORPORATE_ACTIONS_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["symbol", "action_date", "action_type", "ratio", "cash_amount", "details"])
    df = pd.read_csv(path, parse_dates=["action_date"])
    return df


def find_unexplained_jumps(bars: pd.DataFrame, actions: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    bars: OHLCV_COLUMNS dataframe for one symbol, sorted ascending by date.
    Returns the subset of rows whose day-over-day adj_close return exceeds
    SUSPICIOUS_JUMP_THRESHOLD with no corporate action recorded on that date.
    """
    if bars.empty:
        return bars.iloc[0:0]

    df = bars.copy()
    df["ret"] = df["adj_close"].pct_change()
    action_dates = set(pd.to_datetime(actions.loc[actions["symbol"] == symbol, "action_date"]).dt.date)

    def _is_unexplained(row) -> bool:
        if pd.isna(row["ret"]):
            return False
        if abs(row["ret"]) < SUSPICIOUS_JUMP_THRESHOLD:
            return False
        return row["date"].date() not in action_dates

    mask = df.apply(_is_unexplained, axis=1)
    return df.loc[mask, ["date", "close", "adj_close", "ret"]]


def apply_split_adjustment(bars: pd.DataFrame, action_date, ratio: float) -> pd.DataFrame:
    """
    Back-adjusts open/high/low/close/volume for all sessions strictly before
    `action_date` by `ratio` (e.g. ratio=4.0 for a 4-for-1 split), matching
    how adj_close is conventionally computed. Used when a source only
    provides raw prices and the split has to be applied manually.
    """
    df = bars.copy()
    action_date = pd.to_datetime(action_date)
    before = df["date"] < action_date
    for col in ["open", "high", "low", "close", "adj_close"]:
        df.loc[before, col] = df.loc[before, col] / ratio
    df.loc[before, "volume"] = (df.loc[before, "volume"] * ratio).astype("int64")
    return df

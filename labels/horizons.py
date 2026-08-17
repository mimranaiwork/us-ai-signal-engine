"""
Seven-horizon BUY/HOLD/SELL label generation.

Horizons are trading-session counts (1/5/10/21/42/63/126 — see
calendar_us.trading_calendar.HORIZON_SESSIONS), NOT calendar days. Because
`df` here is already indexed one-row-per-trading-session (see
ingestion/ohlcv_source.py), "N trading sessions ahead of row T" is exactly
row T+N — no calendar arithmetic needed at label-generation time, which is
both simpler and correct (calendar-day math would silently misalign horizons
across holidays/weekends).

Never use `future_price > current_price` as the label — that's the "excessive
noise" trap the spec calls out (near-zero moves get force-labeled BUY/SELL as
often as real moves). Every method below requires the move to clear a
volatility-aware threshold before it counts as BUY/SELL; anything inside the
threshold is HOLD.

Five labeling methodologies are implemented (fixed / atr / volatility /
market_relative / risk_adjusted). Phase 1 ships `volatility` as the default;
picking the empirically best one requires walk-forward backtesting against
trained models, which is a Phase 2+/training-phase task — there's nothing to
walk-forward-validate against yet with no model to score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from calendar_us.trading_calendar import HORIZON_SESSIONS

LabelMethod = str  # "fixed" | "atr" | "volatility" | "market_relative" | "risk_adjusted"


@dataclass(frozen=True)
class LabelConfig:
    method: LabelMethod = "volatility"
    fixed_threshold: float = 0.02          # "fixed" method: flat return threshold
    atr_multiplier: float = 1.5            # "atr" method: threshold = atr_percent * multiplier
    vol_multiplier: float = 0.5            # "volatility" method: threshold = horizon_vol * multiplier
    risk_adjusted_z: float = 0.5           # "risk_adjusted": threshold on return/horizon-vol "z-score"


def _future_return(px: pd.Series, horizon_sessions: int) -> pd.Series:
    """future_return[T] = close[T + horizon] / close[T] - 1, via row shift
    (row T+N *is* "N trading sessions after T" — see module docstring)."""
    return px.shift(-horizon_sessions) / px - 1


def _target_session_date(dates: pd.Series, horizon_sessions: int) -> pd.Series:
    return dates.shift(-horizon_sessions)


def _label_from_threshold(future_return: pd.Series, threshold: pd.Series | float) -> pd.Series:
    label = pd.Series("HOLD", index=future_return.index)
    label = label.mask(future_return > threshold, "BUY")
    label = label.mask(future_return < -threshold, "SELL")
    return label


def generate_labels(
    df: pd.DataFrame,
    horizon_name: str,
    config: LabelConfig | None = None,
    benchmark_future_return: pd.Series | None = None,
) -> pd.DataFrame:
    """
    df must already carry the price/technical feature columns needed by the
    chosen method (`atr_percent` for "atr", `volatility_20d` for "volatility"/
    "risk_adjusted", and be pre-aligned by date with `benchmark_future_return`
    for "market_relative"). See features/price and features/technical.

    Returns a frame with: symbol-agnostic columns
        date, horizon, future_return, threshold_used, labeling_method, label,
        target_session_date
    """
    config = config or LabelConfig()
    if horizon_name not in HORIZON_SESSIONS:
        raise ValueError(f"Unknown horizon '{horizon_name}'. Valid: {list(HORIZON_SESSIONS)}")
    horizon_sessions = HORIZON_SESSIONS[horizon_name]

    px = df["adj_close"]
    future_return = _future_return(px, horizon_sessions)
    target_date = _target_session_date(df["date"], horizon_sessions)

    method = config.method
    if method == "fixed":
        threshold = pd.Series(config.fixed_threshold, index=df.index)
        label = _label_from_threshold(future_return, threshold)

    elif method == "atr":
        if "atr_percent" not in df.columns:
            raise ValueError("atr method requires 'atr_percent' column (see features.price_features)")
        threshold = df["atr_percent"] * config.atr_multiplier
        label = _label_from_threshold(future_return, threshold)

    elif method == "volatility":
        if "volatility_20d" not in df.columns:
            raise ValueError("volatility method requires 'volatility_20d' column (see features.technical.indicators)")
        horizon_vol = df["volatility_20d"] * np.sqrt(horizon_sessions / 252)
        threshold = horizon_vol * config.vol_multiplier
        label = _label_from_threshold(future_return, threshold)

    elif method == "market_relative":
        if benchmark_future_return is None:
            raise ValueError("market_relative method requires benchmark_future_return")
        relative_return = future_return - benchmark_future_return
        if "volatility_20d" in df.columns:
            horizon_vol = df["volatility_20d"] * np.sqrt(horizon_sessions / 252)
            threshold = horizon_vol * config.vol_multiplier
        else:
            threshold = pd.Series(config.fixed_threshold, index=df.index)
        label = _label_from_threshold(relative_return, threshold)
        future_return = relative_return  # the label reflects the relative move

    elif method == "risk_adjusted":
        if "volatility_20d" not in df.columns:
            raise ValueError("risk_adjusted method requires 'volatility_20d' column")
        horizon_vol = (df["volatility_20d"] * np.sqrt(horizon_sessions / 252)).replace(0, np.nan)
        risk_adjusted_return = future_return / horizon_vol
        threshold = pd.Series(config.risk_adjusted_z, index=df.index)
        label = _label_from_threshold(risk_adjusted_return, threshold)
        threshold = threshold * horizon_vol  # report threshold back in return units

    else:
        raise ValueError(f"Unknown labeling method '{method}'")

    return pd.DataFrame({
        "date": df["date"],
        "horizon": horizon_name,
        "future_return": future_return,
        "threshold_used": threshold if isinstance(threshold, pd.Series) else pd.Series(threshold, index=df.index),
        "labeling_method": method,
        "label": label,
        "target_session_date": target_date,
    })


def generate_all_horizons(
    df: pd.DataFrame,
    config: LabelConfig | None = None,
    benchmark_future_returns: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Runs generate_labels for all seven horizons and concatenates the results."""
    config = config or LabelConfig()
    frames = []
    for horizon_name in HORIZON_SESSIONS:
        bench_ret = None
        if config.method == "market_relative" and benchmark_future_returns is not None:
            bench_ret = benchmark_future_returns.get(horizon_name)
        frames.append(generate_labels(df, horizon_name, config, bench_ret))
    return pd.concat(frames, ignore_index=True)

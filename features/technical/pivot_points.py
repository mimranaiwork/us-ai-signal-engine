"""
Classic (Standard/Floor Trader) pivot points + a breakout proximity read —
not part of the original 55-section spec, added on request for
resistance/support levels and a "how close to breaking out" read.

Computed daily from the immediately preceding COMPLETED trading session's
high/low/close (shift(1) before deriving) — the same "Daily" convention most
charting platforms use by default for a Pivot Points indicator, including
TradingView's, which this project's frontend is styled after. Backward-
looking only: row T's levels are knowable at the START of session T, before
that session's own price action — no leakage.
"""

from __future__ import annotations

import pandas as pd

# Within this % of a resistance level counts as "near" a breakout — a round,
# documented threshold, not backtested/optimized. Same honesty caveat as
# other un-validated thresholds in this project (e.g.
# signals/market_regime.py's HIGH_VOLATILITY_THRESHOLD).
BREAKOUT_PROXIMITY_PCT = 0.02


def add_pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    prev_close = df["close"].shift(1)

    pivot = (prev_high + prev_low + prev_close) / 3
    df["pivot"] = pivot
    df["resistance_1"] = 2 * pivot - prev_low
    df["support_1"] = 2 * pivot - prev_high
    df["resistance_2"] = pivot + (prev_high - prev_low)
    df["support_2"] = pivot - (prev_high - prev_low)
    df["resistance_3"] = prev_high + 2 * (pivot - prev_low)
    df["support_3"] = prev_low - 2 * (prev_high - pivot)
    return df


def breakout_assessment(latest_row: pd.Series) -> dict:
    """Where the latest close sits relative to its own pivot levels: the
    nearest resistance/support above/below price, whether price is "near
    breakout" (within BREAKOUT_PROXIMITY_PCT of resistance, still below it),
    and a target/expected-profit if it does reach that resistance level.

    This is a deterministic geometric read of price vs. pivot levels, NOT a
    model output or a probability — same honesty class as
    signals/stock_regime.py: real and computed, but not walk-forward-
    validated as predictive of anything."""
    price = latest_row.get("adj_close")
    level_cols = ["resistance_3", "resistance_2", "resistance_1", "pivot", "support_1", "support_2", "support_3"]
    if pd.isna(price) or any(pd.isna(latest_row.get(c)) for c in level_cols):
        return {"available": False, "reason": "insufficient history for pivot levels"}

    price = float(price)
    resistances = sorted(float(latest_row[c]) for c in ("resistance_1", "resistance_2", "resistance_3"))
    supports = sorted((float(latest_row[c]) for c in ("support_1", "support_2", "support_3")), reverse=True)

    resistances_above = [r for r in resistances if r > price]
    supports_below = [s for s in supports if s < price]
    nearest_resistance = min(resistances_above) if resistances_above else None
    nearest_support = max(supports_below) if supports_below else None

    near_breakout = False
    if nearest_resistance is not None:
        distance_pct = (nearest_resistance - price) / price
        near_breakout = 0 <= distance_pct <= BREAKOUT_PROXIMITY_PCT

    target = nearest_resistance
    expected_profit_pct = round((target - price) / price * 100, 2) if target is not None else None

    return {
        "available": True,
        "current_price": round(price, 2),
        "pivot": round(float(latest_row["pivot"]), 2),
        "resistance": {
            "r1": round(float(latest_row["resistance_1"]), 2),
            "r2": round(float(latest_row["resistance_2"]), 2),
            "r3": round(float(latest_row["resistance_3"]), 2),
        },
        "support": {
            "s1": round(float(latest_row["support_1"]), 2),
            "s2": round(float(latest_row["support_2"]), 2),
            "s3": round(float(latest_row["support_3"]), 2),
        },
        "nearest_resistance": round(nearest_resistance, 2) if nearest_resistance is not None else None,
        "nearest_support": round(nearest_support, 2) if nearest_support is not None else None,
        "near_breakout": near_breakout,
        "target": round(target, 2) if target is not None else None,
        "expected_profit_pct": expected_profit_pct,
    }

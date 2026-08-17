"""
Generates a fully offline, synthetic OHLCV dataset so the rest of Phase 1
(features, labels, tests, API) can run without network access or API keys.

Produces one CSV per symbol under data/market/sample/{symbol}.csv, for:
  - every active symbol in data/universe/stocks.csv
  - benchmark ETFs: SPY, QQQ, IWM, DIA
  - sector ETFs: XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE, XLC

Also writes data/market/sample_corporate_actions.csv with one synthetic split
event, so corporate-action handling has something real to exercise.

Price paths are geometric Brownian motion, seeded deterministically per
symbol (hash of the ticker) so the dataset is reproducible across runs.
This is synthetic data for pipeline testing — NOT real historical prices.
"""

from __future__ import annotations

import hashlib
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from calendar_us.trading_calendar import trading_sessions  # noqa: E402
from universe.universe_loader import Universe  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "market", "sample")
CORP_ACTIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "market", "sample_corporate_actions.csv")

START = date(2018, 1, 1)
END = date(2026, 8, 8)

BENCHMARK_ETFS = ["SPY", "QQQ", "IWM", "DIA"]
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]

# Symbol -> (annual drift, annual volatility) so different names behave
# differently (e.g. semis more volatile than staples), still all synthetic.
DEFAULT_DRIFT = 0.10
DEFAULT_VOL = 0.30

SPLIT_SYMBOL = "NVDA"
SPLIT_DATE = date(2022, 6, 15)
SPLIT_RATIO = 4.0


def _seed_for(symbol: str) -> int:
    return int(hashlib.sha256(symbol.encode()).hexdigest(), 16) % (2**32)


def _gbm_path(symbol: str, sessions: list[date], start_price: float,
              drift: float, vol: float) -> pd.DataFrame:
    rng = np.random.default_rng(_seed_for(symbol))
    n = len(sessions)
    dt = 1 / 252
    shocks = rng.normal((drift - 0.5 * vol**2) * dt, vol * np.sqrt(dt), size=n)
    log_prices = np.log(start_price) + np.cumsum(shocks)
    close = np.exp(log_prices)

    intraday_vol = vol * np.sqrt(dt) * 0.6
    open_ = close * (1 + rng.normal(0, intraday_vol, size=n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, intraday_vol, size=n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, intraday_vol, size=n)))
    volume = rng.integers(1_000_000, 20_000_000, size=n)

    df = pd.DataFrame({
        "date": sessions,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "adj_close": close,
        "volume": volume,
    })
    return df


def _apply_synthetic_split(df: pd.DataFrame, split_date: date, ratio: float) -> pd.DataFrame:
    """Split forward in time: raw close continues its path, but from
    split_date onward the *nominal* price is divided by `ratio`, and
    everything before is back-adjusted in adj_close (close stays "as traded")."""
    df = df.copy()
    on_or_after = df["date"] >= split_date
    for col in ["open", "high", "low", "close"]:
        df.loc[on_or_after, col] = df.loc[on_or_after, col] / ratio
    df.loc[on_or_after, "volume"] = (df.loc[on_or_after, "volume"] * ratio).astype("int64")
    # adj_close is continuous / back-adjusted across the whole series
    df["adj_close"] = df["close"]
    before = ~on_or_after
    df.loc[before, "adj_close"] = df.loc[before, "close"] / ratio
    return df


def generate_all(out_dir: str = OUT_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)
    sessions = trading_sessions(START, END)
    print(f"Generating {len(sessions)} trading sessions from {START} to {END}")

    universe = Universe()
    symbols = universe.all_symbols(active_only=True)
    all_symbols = sorted(set(symbols) | set(BENCHMARK_ETFS) | set(SECTOR_ETFS))

    for symbol in all_symbols:
        start_price = 50 + (_seed_for(symbol) % 400)
        drift = DEFAULT_DRIFT + ((_seed_for(symbol) % 21) - 10) / 200
        vol = DEFAULT_VOL + ((_seed_for(symbol) % 21) - 10) / 200
        vol = max(vol, 0.12)

        df = _gbm_path(symbol, sessions, start_price, drift, vol)
        if symbol == SPLIT_SYMBOL:
            df = _apply_synthetic_split(df, SPLIT_DATE, SPLIT_RATIO)

        df.to_csv(os.path.join(out_dir, f"{symbol}.csv"), index=False)

    pd.DataFrame([{
        "symbol": SPLIT_SYMBOL,
        "action_date": SPLIT_DATE.isoformat(),
        "action_type": "SPLIT",
        "ratio": SPLIT_RATIO,
        "cash_amount": "",
        "details": f"Synthetic {int(SPLIT_RATIO)}-for-1 split for sample-data testing",
    }]).to_csv(CORP_ACTIONS_PATH, index=False)

    print(f"Wrote {len(all_symbols)} symbol CSVs to {out_dir}")
    print(f"Wrote corporate actions to {CORP_ACTIONS_PATH}")


if __name__ == "__main__":
    generate_all()

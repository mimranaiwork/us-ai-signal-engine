"""
Options features (spec section 18): implied volatility, IV rank/percentile,
put/call ratio, volumes, open interest, and derived sentiment.

Real historical options chain data (IV, open interest, put/call volume) has
no free public source comparable to SEC EDGAR or Yahoo-style equity
quotes — it requires a paid vendor (CBOE DataShop, ORATS, OPRA feed
resellers, etc.). Per the spec's own instruction ("do not make options
features mandatory if historical data quality is insufficient"), this
module is built against a clean `OptionsDataSource` interface — mirroring
`ingestion/ohlcv_source.py`'s pattern — with only a `SyntheticOptionsSource`
implementation shipped. It generates IV as a plausible function of the
underlying's own *realized* volatility (real, already-computed data) plus
noise, NOT arbitrary random numbers, so the derived ratios below are
internally consistent even though the IV itself isn't real. Swap in a paid
vendor by implementing the same interface — nothing downstream changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

IV_HISTORY_WINDOW = 252  # ~1 year, standard IV-rank lookback


@dataclass
class OptionsSnapshot:
    date: object
    iv: float
    call_volume: float
    put_volume: float
    open_interest: float

    @property
    def put_call_ratio(self) -> float:
        return self.put_volume / self.call_volume if self.call_volume else float("nan")


class OptionsDataSource(ABC):
    @abstractmethod
    def get_daily_snapshots(self, symbol: str, price_history: pd.DataFrame) -> pd.DataFrame:
        """price_history: OHLCV_COLUMNS dataframe (see ingestion.ohlcv_source) for
        the underlying. Returns a DataFrame with columns:
        date, iv, call_volume, put_volume, open_interest."""
        raise NotImplementedError


class SyntheticOptionsSource(OptionsDataSource):
    """IV = trailing realized volatility * (1 + mean-reverting skew noise).
    Put/call volume skews toward puts after down days (a real, well-known
    behavioral pattern — hedging demand rises with recent drawdowns) plus
    noise. Seeded per-symbol for reproducibility, same convention as
    scripts/generate_sample_data.py."""

    def __init__(self, seed_salt: str = "options"):
        self.seed_salt = seed_salt

    def get_daily_snapshots(self, symbol: str, price_history: pd.DataFrame) -> pd.DataFrame:
        import hashlib
        seed = int(hashlib.sha256(f"{symbol}_{self.seed_salt}".encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)

        df = price_history.copy().reset_index(drop=True)
        returns = df["adj_close"].pct_change()
        realized_vol = returns.rolling(20).std() * np.sqrt(252)

        skew_noise = pd.Series(rng.normal(0, 0.05, len(df))).rolling(5, min_periods=1).mean()
        iv = (realized_vol * (1 + skew_noise)).clip(lower=0.05)

        recent_return = returns.rolling(5).mean().fillna(0)
        put_skew = (-recent_return * 10).clip(-0.5, 0.5)  # more puts after recent weakness
        base_volume = rng.uniform(1000, 10000, len(df))
        call_volume = base_volume * (1 - put_skew).clip(lower=0.1)
        put_volume = base_volume * (1 + put_skew).clip(lower=0.1)
        open_interest = base_volume * rng.uniform(5, 20, len(df))

        return pd.DataFrame({
            "date": df["date"], "iv": iv,
            "call_volume": call_volume, "put_volume": put_volume,
            "open_interest": open_interest,
        })


def add_options_features(snapshots: pd.DataFrame, realized_vol: pd.Series) -> pd.DataFrame:
    """snapshots: output of get_daily_snapshots(). realized_vol: the
    underlying's own volatility_20d series (see features.technical.indicators),
    same length/order, for iv_vs_realized_volatility."""
    df = snapshots.copy()
    df["put_call_ratio"] = df["put_volume"] / df["call_volume"]
    df["total_options_volume"] = df["call_volume"] + df["put_volume"]

    df["iv_rank"] = df["iv"].rolling(IV_HISTORY_WINDOW, min_periods=60).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5, raw=False
    )
    df["iv_percentile"] = df["iv"].rolling(IV_HISTORY_WINDOW, min_periods=60).apply(
        lambda x: (x < x.iloc[-1]).mean(), raw=False
    )
    df["iv_vs_realized_volatility"] = df["iv"] - realized_vol.reset_index(drop=True)

    # options_sentiment: low put/call (call-heavy) + IV below its own recent
    # norm (complacent, not fearful) -> bullish; the reverse -> bearish.
    pc_z = (df["put_call_ratio"] - df["put_call_ratio"].rolling(60, min_periods=20).mean()) / \
           df["put_call_ratio"].rolling(60, min_periods=20).std()
    df["options_sentiment"] = (-pc_z).clip(-3, 3) / 3  # roughly [-1, 1]

    volume_z = (df["total_options_volume"] - df["total_options_volume"].rolling(20, min_periods=10).mean()) / \
               df["total_options_volume"].rolling(20, min_periods=10).std()
    df["unusual_options_activity"] = volume_z.abs() > 2

    return df

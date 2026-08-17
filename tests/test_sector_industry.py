import os

import numpy as np
import pandas as pd
import pytest

from features.industry.industry_features import add_stock_vs_industry, compute_industry_peer_returns
from features.sector.sector_features import add_stock_vs_sector, compute_sector_features


def _synthetic_etf_bars(n=60, drift=0.001, seed=0):
    rng = np.random.default_rng(seed)
    prices = 100 * (1 + rng.normal(drift, 0.01, n)).cumprod()
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "date": dates, "open": prices, "high": prices * 1.005, "low": prices * 0.995,
        "close": prices, "adj_close": prices, "volume": np.full(n, 1_000_000),
    })


# ── Sector ───────────────────────────────────────────────────────────────────

def test_compute_sector_features_shape():
    etf_bars = _synthetic_etf_bars()
    result = compute_sector_features(etf_bars)
    for col in ("sector_return_1d", "sector_return_1w", "sector_return_1m", "sector_momentum", "trend"):
        assert col in result.columns
    assert len(result) == len(etf_bars)


def test_stock_outperforming_sector_shows_positive_relative_strength():
    sector_bars = _synthetic_etf_bars(drift=0.0005, seed=1)   # modest sector drift
    stock_bars = _synthetic_etf_bars(drift=0.01, seed=2)      # stock rallies much harder

    sector_features = compute_sector_features(sector_bars)
    stock_df = pd.DataFrame({"date": stock_bars["date"], "daily_return": stock_bars["adj_close"].pct_change()})

    result = add_stock_vs_sector(stock_df, sector_features)
    assert result["stock_vs_sector_21d"].dropna().iloc[-1] > 0


def test_stock_underperforming_sector_shows_negative_relative_strength():
    sector_bars = _synthetic_etf_bars(drift=0.01, seed=1)     # sector rallies hard
    stock_bars = _synthetic_etf_bars(drift=-0.005, seed=2)    # stock declines

    sector_features = compute_sector_features(sector_bars)
    stock_df = pd.DataFrame({"date": stock_bars["date"], "daily_return": stock_bars["adj_close"].pct_change()})

    result = add_stock_vs_sector(stock_df, sector_features)
    assert result["stock_vs_sector_21d"].dropna().iloc[-1] < 0


# ── Industry ─────────────────────────────────────────────────────────────────

@pytest.fixture
def processed_dir(tmp_path):
    dates = pd.date_range("2024-01-01", periods=30)
    # Two peers in the same (synthetic) industry, one lone stock in another.
    a = pd.DataFrame({"date": dates, "daily_return": np.full(30, 0.01)})
    b = pd.DataFrame({"date": dates, "daily_return": np.full(30, 0.03)})
    c = pd.DataFrame({"date": dates, "daily_return": np.full(30, -0.01)})  # lone stock, no peers
    for symbol, df in [("PEER_A", a), ("PEER_B", b), ("LONELY", c)]:
        df.to_csv(os.path.join(tmp_path, f"{symbol}_features.csv"), index=False)
    return str(tmp_path)


class _FakeStock:
    def __init__(self, industry_code):
        self.industry_code = industry_code


class _FakeUniverse:
    """Avoids depending on the real data/universe/*.csv — this test wants
    full control over which symbols share an industry."""
    def __init__(self, industry_map: dict[str, str]):
        self._map = industry_map

    def all_symbols(self, active_only=True):
        return list(self._map.keys())

    def get(self, symbol):
        return _FakeStock(self._map[symbol])


def test_industry_peer_average_computed_correctly(processed_dir):
    universe = _FakeUniverse({"PEER_A": "WIDGETS", "PEER_B": "WIDGETS", "LONELY": "GADGETS"})
    result = compute_industry_peer_returns(universe, processed_dir=processed_dir)

    assert "WIDGETS" in result
    assert "GADGETS" not in result  # only 1 member -> not enough peers, correctly excluded

    widgets = result["WIDGETS"]
    assert widgets["industry_return"].iloc[0] == pytest.approx((0.01 + 0.03) / 2)
    assert (widgets["n_peers"] == 2).all()


def test_stock_vs_industry_relative_strength(processed_dir):
    universe = _FakeUniverse({"PEER_A": "WIDGETS", "PEER_B": "WIDGETS"})
    industry_returns = compute_industry_peer_returns(universe, processed_dir=processed_dir)["WIDGETS"]

    stock_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=30),
        "daily_return": np.full(30, 0.03),  # matches PEER_B, above the 0.02 peer average
    })
    result = add_stock_vs_industry(stock_df, industry_returns)
    assert result["stock_vs_industry_1d"].iloc[0] == pytest.approx(0.03 - 0.02)


def test_stock_vs_industry_returns_nan_when_no_peer_group():
    stock_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "daily_return": [0.01, 0.02, -0.01, 0.0, 0.03],
    })
    result = add_stock_vs_industry(stock_df, None)
    assert result["industry_return"].isna().all()
    assert result["stock_vs_industry_1d"].isna().all()

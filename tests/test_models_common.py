import os

import numpy as np
import pandas as pd
import pytest

from models.common import build_pooled_dataset


@pytest.fixture
def processed_dir(tmp_path, monkeypatch):
    import models.common as common_mod

    monkeypatch.setattr(common_mod, "PROCESSED_DIR", str(tmp_path))

    dates = pd.date_range("2024-01-01", periods=10)

    # HAS_PEERS: real industry_return values (a symbol whose industry has >=2 peers).
    has_peers = pd.DataFrame({
        "date": dates,
        "adj_close": np.linspace(100, 110, 10),
        "rsi_14": np.full(10, 50.0),
        "industry_return": np.full(10, 0.01),
        "n_peers": np.full(10, 2),
        "stock_vs_industry_1d": np.full(10, 0.001),
    })
    has_peers.to_csv(tmp_path / "HAS_PEERS_features.csv", index=False)

    # NO_PEERS: industry columns entirely NaN (lone symbol in its industry) —
    # every other feature is fine, so this symbol should still survive pooling.
    no_peers = pd.DataFrame({
        "date": dates,
        "adj_close": np.linspace(200, 210, 10),
        "rsi_14": np.full(10, 55.0),
        "industry_return": np.full(10, np.nan),
        "n_peers": np.full(10, np.nan),
        "stock_vs_industry_1d": np.full(10, np.nan),
    })
    no_peers.to_csv(tmp_path / "NO_PEERS_features.csv", index=False)

    for symbol in ("HAS_PEERS", "NO_PEERS"):
        labels = pd.DataFrame({
            "date": dates,
            "target_session_date": dates,
            "horizon": ["1D"] * 10,
            "label": ["HOLD"] * 10,
            "symbol": symbol,
        })
        labels.to_csv(tmp_path / f"{symbol}_labels.csv", index=False)

    return str(tmp_path)


def test_symbol_without_industry_peers_is_not_dropped_from_pooled_dataset(processed_dir):
    pooled = build_pooled_dataset(["HAS_PEERS", "NO_PEERS"], "1D")
    assert set(pooled["symbol"].unique()) == {"HAS_PEERS", "NO_PEERS"}


def test_no_peer_group_industry_columns_filled_with_zero_not_dropped(processed_dir):
    pooled = build_pooled_dataset(["HAS_PEERS", "NO_PEERS"], "1D")
    no_peer_rows = pooled[pooled["symbol"] == "NO_PEERS"]
    assert not no_peer_rows.empty
    assert (no_peer_rows["industry_return"] == 0.0).all()
    assert (no_peer_rows["n_peers"] == 0.0).all()
    assert (no_peer_rows["stock_vs_industry_1d"] == 0.0).all()


def test_symbol_with_peers_keeps_real_industry_values(processed_dir):
    pooled = build_pooled_dataset(["HAS_PEERS", "NO_PEERS"], "1D")
    has_peer_rows = pooled[pooled["symbol"] == "HAS_PEERS"]
    assert np.allclose(has_peer_rows["industry_return"].to_numpy(), 0.01)

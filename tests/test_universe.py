from datetime import date

import pytest

from universe.universe_loader import Universe


@pytest.fixture(scope="module")
def universe():
    return Universe()


def test_loads_without_error(universe):
    assert len(universe.stocks) > 0
    assert len(universe.exchanges) > 0
    assert len(universe.sectors) > 0
    assert len(universe.industries) > 0


def test_all_symbols_active_only_excludes_delisted(universe):
    active = universe.all_symbols(active_only=True)
    assert "AAPL" in active
    assert "LEHMQ" not in active
    assert "FB" not in active


def test_all_symbols_including_delisted(universe):
    everyone = universe.all_symbols(active_only=False)
    assert "LEHMQ" in everyone
    assert "FB" in everyone


def test_survivorship_bias_point_in_time_membership(universe):
    # Lehman existed and should be "in the universe" as of 2005, but was
    # removed (bankruptcy, 2008-09-15) and must NOT appear in a 2020 snapshot —
    # this is exactly the check a naive "just use today's active list" backtest
    # would get wrong.
    assert "LEHMQ" in universe.as_of(date(2005, 1, 1))
    assert "LEHMQ" not in universe.as_of(date(2020, 1, 1))

    # FB existed as a ticker until 2022-06-09; must not appear afterward.
    assert "FB" in universe.as_of(date(2020, 1, 1))
    assert "FB" not in universe.as_of(date(2023, 1, 1))

    # A stock always active should appear on both dates.
    assert "AAPL" in universe.as_of(date(2020, 1, 1))
    assert "AAPL" in universe.as_of(date(2023, 1, 1))


def test_get_stock(universe):
    aapl = universe.get("AAPL")
    assert aapl.symbol == "AAPL"
    assert aapl.active is True

    lehman = universe.get("LEHMQ")
    assert lehman.active is False
    assert lehman.delisted_reason == "BANKRUPTCY"


def test_sector_etf_lookup(universe):
    assert universe.sector_etf("TECHNOLOGY") == "XLK"
    assert universe.sector_etf("FINANCIALS") == "XLF"


def test_unknown_symbol_raises(universe):
    with pytest.raises(KeyError):
        universe.get("NOT_A_REAL_TICKER")

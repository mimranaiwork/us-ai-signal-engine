"""Point-in-time Shariah universe: N-PORT parsing, ticker mapping, membership lookups."""

import pandas as pd

from execution.shariah_history import Membership, is_equity, norm_name, parse_nport, resolve_ticker

NPORT = """<?xml version="1.0" encoding="UTF-8"?>
<edgarSubmission xmlns="http://www.sec.gov/edgar/nport" xmlns:com="http://www.sec.gov/edgar/common">
  <headerData><submissionType>NPORT-P</submissionType></headerData>
  <formData>
    <genInfo><seriesName>SP Funds S&amp;P 500 Sharia Industry Exclusions ETF</seriesName>
      <seriesId>S000067283</seriesId><repPdEnd>2023-11-30</repPdEnd><repPdDate>2023-02-28</repPdDate></genInfo>
    <invstOrSecs>
      <invstOrSec><name>Apple Inc</name><lei>X</lei><title>Apple Inc</title><cusip>037833100</cusip>
        <identifiers><isin value="US0378331005"/><ticker value="AAPL"/></identifiers>
        <valUSD>1000</valUSD><pctVal>10.5</pctVal><assetCat>EC</assetCat></invstOrSec>
      <invstOrSec><name>Abiomed Inc</name><title>ABIOMED CVR</title><cusip>003654100</cusip>
        <identifiers><isin value="US0036541003"/></identifiers><valUSD>0</valUSD><pctVal>0</pctVal><assetCat>EC</assetCat></invstOrSec>
      <invstOrSec><name>Pool Corp</name><title>Pool Corp</title><cusip>73278L105</cusip>
        <identifiers><isin value="US73278L1052"/></identifiers><valUSD>50</valUSD><pctVal>0.5</pctVal><assetCat>EC</assetCat></invstOrSec>
      <invstOrSec><name>First American Treasury Obligations Fund</name><title>X</title><cusip>31846V328</cusip>
        <identifiers><isin value="US31846V3289"/></identifiers><valUSD>20</valUSD><pctVal>0.2</pctVal><assetCat>STIV</assetCat></invstOrSec>
    </invstOrSecs>
  </formData>
</edgarSubmission>"""


def test_parse_nport_reads_series_period_and_holdings():
    doc = parse_nport(NPORT)
    assert doc["series_id"] == "S000067283" and doc["period_end"] == "2023-02-28"
    assert [h["name"] for h in doc["holdings"]][:3] == ["Apple Inc", "Abiomed Inc", "Pool Corp"]
    assert doc["holdings"][0]["ticker"] == "AAPL" and doc["holdings"][0]["pct"] == 10.5


def test_equity_filter_drops_cash_funds_and_cvrs():
    h = parse_nport(NPORT)["holdings"]
    assert [x["name"] for x in h if is_equity(x)] == ["Apple Inc", "Pool Corp"]


def test_ticker_resolution_order():
    h = parse_nport(NPORT)["holdings"]
    assert resolve_ticker(h[0], {}, {}) == ("AAPL", "filing")
    assert resolve_ticker(h[2], {"73278L105": "POOL"}, {}) == ("POOL", "cusip")
    assert resolve_ticker(h[2], {}, {norm_name("POOL CORP"): "POOL"}) == ("POOL", "name")
    assert resolve_ticker({"name": "Gone Ltd", "cusip": "X"}, {}, {}) == ("", "unmatched")


def test_name_normalisation_ignores_legal_suffixes():
    assert norm_name("Alphabet Inc. - Class A") == norm_name("ALPHABET INC")
    assert norm_name("Procter & Gamble Co/The") == norm_name("PROCTER AND GAMBLE CO")


def test_membership_applies_each_snapshot_until_the_next():
    hist = pd.DataFrame({"snapshot_date": ["2021-03-31", "2021-03-31", "2021-06-30", "2021-06-30"],
                         "ticker": ["AAA", "BBB", "AAA", "CCC"]})
    m = Membership(hist)
    assert m.on("2020-01-02") == {"AAA", "BBB"}          # before first snapshot -> first snapshot
    assert m.on("2021-05-01") == {"AAA", "BBB"}
    assert m.on("2021-06-30") == {"AAA", "CCC"}          # BBB left, CCC joined
    assert m.eligible("2022-01-01", "CCC") and not m.eligible("2022-01-01", "BBB")
    mask = m.mask(pd.DatetimeIndex(["2021-04-01", "2021-07-01"]), ["AAA", "BBB", "CCC"])
    assert mask.values.tolist() == [[True, True, False], [True, False, True]]
    assert m.all == {"AAA", "BBB", "CCC"}


def test_overrides_renames_and_cusip_carry_forward():
    from execution.shariah_history import fill_by_cusip
    assert resolve_ticker({"name": "Facebook Inc"}, {}, {}) == ("META", "override")
    assert resolve_ticker({"name": "McDonald's Corp"}, {}, {}) == ("MCD", "override")
    assert resolve_ticker({"ticker": "FB"}, {}, {}) == ("META", "filing")
    rows = [{"ticker": "EQR", "cusip": "29476L107", "matched_by": "filing"},
            {"ticker": "", "cusip": "29476l107", "matched_by": "unmatched"}]
    assert fill_by_cusip(rows) == 1 and rows[1]["ticker"] == "EQR" and rows[1]["matched_by"] == "cusip_history"


def test_zero_value_leftovers_are_not_holdings():
    assert not is_equity({"name": "ABIOMED INC", "title": "ABIOMED INC", "asset_cat": "EC", "value_usd": 0.0, "pct": 0.0})
    assert is_equity({"name": "Pool Corp", "asset_cat": "EC", "value_usd": 50.0, "pct": 0.5})

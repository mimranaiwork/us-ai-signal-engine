"""
Point-in-time Shariah universe: which stocks SPUS actually held on each past
date, rebuilt from the fund's quarterly SEC N-PORT filings (series
S000067283). Used by the backtests so a stock only counts from when it was
really in the Shariah index — removing the survivorship bias of applying
today's list to the past.

Pure functions only (parsing, ticker mapping, lookups); the network part is
scripts/build_shariah_history.py.

Membership rule: a quarterly snapshot (the fund's holdings at the report
period end) applies from that date until the next snapshot. Dates before the
first snapshot use the first snapshot.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from bisect import bisect_right

import pandas as pd

HISTORY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "universe", "shariah_history.csv")
TICKER_RE = re.compile(r"^[A-Z]{1,5}([.\-][A-Z])?$")
_SUFFIXES = re.compile(
    r"\b(INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|PLC|NV|N V|SA|AG|SE|LP|L P|HOLDINGS?|GROUP|"
    r"THE|CLASS [A-Z]|CL [A-Z]|COM|COMMON STOCK|ORDINARY SHARES?|SHS|NEW|DEL|DE)\b")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(el, name):
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def _text(el, name):
    c = _child(el, name)
    return (c.text or "").strip() if c is not None and c.text else ""


def parse_nport(xml_text: str) -> dict:
    """-> {series_id, period_end, holdings: [{name, title, cusip, isin, ticker, value_usd, pct, asset_cat}]}"""
    root = ET.fromstring(xml_text)
    series_id, period_end, holdings = "", "", []
    for el in root.iter():
        t = _local(el.tag)
        if t == "seriesId" and not series_id:
            series_id = (el.text or "").strip()
        elif t == "repPdDate" and not period_end:
            period_end = (el.text or "").strip()
        elif t == "invstOrSec":
            ids = _child(el, "identifiers")
            isin = ticker = ""
            if ids is not None:
                for i in ids:
                    n = _local(i.tag)
                    if n == "isin":
                        isin = i.get("value", "")
                    elif n == "ticker":
                        ticker = i.get("value", "")
            def num(name):
                try:
                    return float(_text(el, name))
                except ValueError:
                    return None
            holdings.append({"name": _text(el, "name"), "title": _text(el, "title"), "cusip": _text(el, "cusip"),
                             "isin": isin, "ticker": ticker.strip().upper(), "value_usd": num("valUSD"),
                             "pct": num("pctVal"), "asset_cat": _text(el, "assetCat")})
    return {"series_id": series_id, "period_end": period_end, "holdings": holdings}


def norm_name(name: str) -> str:
    s = re.sub(r"[^A-Z0-9 ]", " ", (name or "").upper().replace("&", " AND "))
    s = _SUFFIXES.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def resolve_ticker(h: dict, cusip_map: dict[str, str], name_map: dict[str, str]) -> tuple[str, str]:
    """Best ticker for one holding and how it was found: the filing's own
    ticker, then the CUSIP from the issuer's current holdings file, then an
    exact normalised-name match against SEC's company list. ('', 'unmatched') otherwise."""
    t = (h.get("ticker") or "").replace("/", ".").replace(" ", ".")
    if TICKER_RE.match(t):
        return t.replace("-", "."), "filing"
    c = (h.get("cusip") or "").upper()
    if c and c in cusip_map:
        return cusip_map[c], "cusip"
    for key in (norm_name(h.get("name", "")), norm_name(h.get("title", ""))):
        if key and key in name_map:
            return name_map[key], "name"
    return "", "unmatched"


def is_equity(h: dict) -> bool:
    cat = (h.get("asset_cat") or "").upper()
    text = f"{h.get('name') or ''} {h.get('title') or ''}".upper()
    if cat and cat not in ("EC", "EP"):          # EC common equity, EP preferred
        return False
    return not re.search(r"MONEY MARKET|TREASURY|\bCASH\b|\bCVR\b|CONTINGENT VALUE|\bRIGHTS?\b", text)


class Membership:
    """Fast 'which tickers were in the Shariah index on date d' lookup."""

    def __init__(self, history: pd.DataFrame):
        h = history[history["ticker"].astype(str).str.len() > 0].copy()
        h["snapshot_date"] = pd.to_datetime(h["snapshot_date"])
        snaps = h.groupby("snapshot_date")["ticker"].apply(lambda s: frozenset(s)).sort_index()
        self.dates = list(snaps.index)
        self.sets = list(snaps.values)
        self.all = frozenset().union(*self.sets) if self.sets else frozenset()

    def on(self, d) -> frozenset:
        if not self.dates:
            return frozenset()
        i = bisect_right(self.dates, pd.Timestamp(d)) - 1
        return self.sets[max(i, 0)]

    def eligible(self, d, symbol: str) -> bool:
        return symbol in self.on(d)

    def mask(self, index: pd.DatetimeIndex, columns) -> pd.DataFrame:
        """Boolean dates x symbols frame: True where the symbol was a member."""
        cols = list(columns)
        out = pd.DataFrame(False, index=index, columns=cols)
        for d in index:
            mem = self.on(d)
            out.loc[d] = [c in mem for c in cols]
        return out


def load_membership(path: str = HISTORY_PATH) -> Membership | None:
    if not os.path.exists(path):
        return None
    return Membership(pd.read_csv(path, dtype={"ticker": str}).fillna({"ticker": ""}))

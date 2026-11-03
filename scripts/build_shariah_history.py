"""
Rebuild SPUS's historical holdings from its quarterly SEC N-PORT filings and
write data/universe/shariah_history.csv (one row per holding per snapshot).

Filings are found by SPUS's series ID through EDGAR's company browse feed; if
that returns nothing, it falls back to scanning the Tidal ETF Trust's NPORT-P
filings and keeping those whose seriesId matches.

Needs SEC_EDGAR_USER_AGENT ("Name email") and access to sec.gov — runs on
GitHub Actions (.github/workflows/shariah-history.yml).
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import urllib.request

import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from execution.shariah_history import HISTORY_PATH, fill_by_cusip, is_equity, norm_name, parse_nport, resolve_ticker  # noqa: E402
from execution.shariah_universe import SPUS_HOLDINGS_URL, parse_holdings_csv  # noqa: E402

SERIES_ID = "S000067283"
TRUST_CIK = "1742912"
UA = os.environ.get("SEC_EDGAR_USER_AGENT", "")
_last = [0.0]


def get(url: str, ua: str | None = None, timeout: int = 30) -> str:
    wait = 0.15 - (time.monotonic() - _last[0])
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.monotonic()
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua or UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as exc:
            if attempt == 3:
                raise
            print(f"  retry {attempt + 1} {url}: {exc}")
            time.sleep(2 * (attempt + 1))


def filings_by_series() -> list[str]:
    """Folder URLs of NPORT-P filings for the series, via EDGAR's atom feed."""
    out, start = [], 0
    while True:
        feed = get(f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={SERIES_ID}&type=NPORT-P"
                   f"&dateb=&owner=include&count=100&start={start}&output=atom")
        hrefs = re.findall(r"<filing-href>(.*?)</filing-href>", feed)
        out += [h.rsplit("/", 1)[0] for h in hrefs]
        if len(hrefs) < 100:
            return sorted(set(out))
        start += 100


def filings_by_trust_scan() -> list[str]:
    sub = json.loads(get(f"https://data.sec.gov/submissions/CIK{int(TRUST_CIK):010d}.json"))
    pages = [sub["filings"]["recent"]] + [json.loads(get(f"https://data.sec.gov/submissions/{f['name']}"))
                                         for f in sub["filings"].get("files", [])]
    out = []
    for p in pages:
        for form, acc in zip(p["form"], p["accessionNumber"]):
            if form == "NPORT-P":
                out.append(f"https://www.sec.gov/Archives/edgar/data/{TRUST_CIK}/{acc.replace('-', '')}")
    return out


def main():
    if not UA or "@" not in UA:
        sys.exit("Set SEC_EDGAR_USER_AGENT to 'Your Name you@example.com'.")

    folders = filings_by_series()
    source = "series feed"
    if not folders:
        folders, source = filings_by_trust_scan(), "trust scan"
    print(f"{len(folders)} candidate NPORT-P filings ({source})")

    snaps = {}
    for i, f in enumerate(folders):
        try:
            doc = parse_nport(get(f + "/primary_doc.xml"))
        except Exception as exc:
            print(f"  skip {f}: {exc}")
            continue
        if doc["series_id"] != SERIES_ID or not doc["period_end"]:
            continue
        snaps[doc["period_end"]] = (f, doc["holdings"])    # an amendment for the same period replaces the original
        if i % 20 == 0:
            print(f"  {i + 1}/{len(folders)} filings read, {len(snaps)} SPUS snapshots")
    print(f"{len(snaps)} SPUS quarterly snapshots: {min(snaps)} → {max(snaps)}")

    # ticker maps: today's issuer file (CUSIP -> ticker) and SEC's company list (name -> ticker)
    cusip_map = {}
    try:
        cur = pd.read_csv(io.StringIO(get(SPUS_HOLDINGS_URL, ua="Mozilla/5.0")))
        tcol = next(c for c in cur.columns if re.search(r"ticker|symbol", c, re.I))
        ccol = next(c for c in cur.columns if re.search(r"cusip", c, re.I))
        valid = set(parse_holdings_csv(cur.to_csv(index=False)))
        for t, c in zip(cur[tcol].astype(str), cur[ccol].astype(str)):
            t = t.strip().upper().replace("/", ".")
            if t in valid:
                cusip_map[c.strip().upper()] = t
    except Exception as exc:
        print(f"issuer CUSIP map unavailable: {exc}")
    names = json.loads(get("https://www.sec.gov/files/company_tickers.json"))
    name_map = {}
    for v in names.values():
        name_map.setdefault(norm_name(v["title"]), v["ticker"].upper().replace("-", "."))

    rows, how_counts = [], {}
    for period, (folder, holdings) in sorted(snaps.items()):
        for h in holdings:
            if not is_equity(h):
                continue
            ticker, how = resolve_ticker(h, cusip_map, name_map)
            how_counts[how] = how_counts.get(how, 0) + 1
            rows.append({"snapshot_date": period, "ticker": ticker, "name": h["name"], "cusip": h["cusip"],
                         "isin": h["isin"], "pct": h["pct"], "matched_by": how, "filing": folder})
    filled = fill_by_cusip(rows)
    how_counts["cusip_history"] = filled
    how_counts["unmatched"] = how_counts.get("unmatched", 0) - filled
    df = pd.DataFrame(rows)
    df.to_csv(HISTORY_PATH, index=False)

    per = df.groupby("snapshot_date").agg(holdings=("name", "size"), matched=("ticker", lambda s: (s.str.len() > 0).sum()))
    print(per.to_string())
    print("matched by:", how_counts)
    um = df[df["ticker"] == ""]["name"].value_counts()
    print(f"unmatched names ({len(um)}):", ", ".join(um.index[:40]))
    print(f"wrote {HISTORY_PATH}: {len(df)} rows, {df['ticker'].replace('', pd.NA).nunique()} distinct tickers")


if __name__ == "__main__":
    main()

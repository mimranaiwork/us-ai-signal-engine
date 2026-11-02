"""
Tradeable Shariah universe for the execution layer.

Primary source: the SPUS ETF's published daily holdings (SP Funds S&P 500
Sharia Industry Exclusions ETF, tracking the S&P 500 Shariah index). That
is a professionally screened list (business activity + financial ratios,
reviewed by the index's Shariah board) covering ~215 large caps — broader
than this project's own universe, and refreshed by the issuer as names
pass/fail screening.

Secondary check: this project's own DIY screen
(features/compliance/sharia_screen.py, via scripts/run_sharia_screen.py's
report). Where that report says a symbol is NOT compliant, it is removed
even if SPUS holds it — i.e. a name must not be failed by either source.
The DIY screen only covers this project's tracked universe, so for most
SPUS names it simply has no opinion.

Neither source is a religious ruling; see sharia_screen.py's docstring.
"""

from __future__ import annotations

import io
import json
import os
import re
import urllib.request

import pandas as pd

SPUS_HOLDINGS_URL = "https://www.sp-funds.com/wp-content/uploads/data/TidalFG_Holdings_SPUS.csv"
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
FALLBACK_PATH = os.path.join(REPO_ROOT, "data", "universe", "shariah_spus.txt")
DIY_REPORT_PATH = os.path.join(REPO_ROOT, "models", "saved", "sharia_screen_report.json")
MIN_EXPECTED_HOLDINGS = 100  # a parse returning fewer is treated as a broken download, not a real list

TICKER_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")


def parse_holdings_csv(text: str) -> list[str]:
    """Issuer CSVs name the column StockTicker/Ticker/Symbol depending on
    administrator; non-equity rows (cash, CVRs, CUSIP-only lines) are
    dropped by the ticker pattern."""
    df = pd.read_csv(io.StringIO(text))
    col = next((c for c in df.columns if re.search(r"ticker|symbol", str(c), re.I)), None)
    if col is None:
        return []
    out: list[str] = []
    for t in df[col].dropna().astype(str):
        t = t.strip().upper().replace("/", ".").replace(" ", ".")
        if TICKER_RE.match(t) and t not in out:
            out.append(t)
    return out


def read_symbol_file(path: str) -> list[str]:
    syms: list[str] = []
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            syms += [t for t in line.split() if TICKER_RE.match(t)]
    return list(dict.fromkeys(syms))


def write_symbol_file(path: str, symbols: list[str], header: str) -> None:
    with open(path, "w") as f:
        f.write(f"# {header}\n")
        for i in range(0, len(symbols), 20):
            f.write(" ".join(symbols[i:i + 20]) + "\n")


def _http_get(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def load_spus_universe(url: str = SPUS_HOLDINGS_URL, fallback_path: str = FALLBACK_PATH,
                       fetch=_http_get, today: str | None = None) -> tuple[list[str], str]:
    """Returns (symbols, human-readable source). On a good download the
    fallback file is refreshed so the next offline run is current."""
    try:
        syms = parse_holdings_csv(fetch(url))
        if len(syms) < MIN_EXPECTED_HOLDINGS:
            raise ValueError(f"only {len(syms)} tickers parsed")
        write_symbol_file(fallback_path, syms, f"SPUS holdings downloaded {today or ''}".strip())
        return syms, f"SPUS live holdings ({len(syms)} stocks)"
    except Exception as exc:
        syms = read_symbol_file(fallback_path)
        return syms, f"saved SPUS list ({len(syms)} stocks; download failed: {exc})"


def diy_non_compliant(report_path: str | None = None) -> set[str]:
    """Symbols this project's own screen explicitly marked non-compliant.
    Missing report -> empty set (no opinion), mirroring api/main.py's
    _load_sharia_lookup degrade-gracefully behavior."""
    report_path = report_path or DIY_REPORT_PATH  # resolved at call time, not import time
    if not os.path.exists(report_path):
        return set()
    with open(report_path) as f:
        report = json.load(f)
    return {r["symbol"] for r in report.get("results", []) if r.get("compliant") is False}


def build_universe(exclude: list[str] | None = None, **kwargs) -> tuple[list[str], str, list[str]]:
    """Returns (symbols, source_description, removed_by_diy_screen)."""
    syms, source = load_spus_universe(**kwargs)
    removed = sorted(set(syms) & diy_non_compliant())
    drop = set(removed) | set(exclude or [])
    return [s for s in syms if s not in drop], source, removed

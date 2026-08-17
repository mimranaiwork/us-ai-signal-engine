"""
Loads and validates the configurable US stock universe (data/universe/*.csv).

Survivorship-bias note: `stocks.csv` carries `active`, `date_added`,
`date_removed`, and `delisted_reason` columns specifically so historical
backtests can reconstruct "what the universe actually looked like on date X"
instead of filtering to today's survivors. Phase 1 ships only two illustrative
delisted rows (Lehman Brothers, Facebook's ticker retirement) to prove the
schema works end-to-end — populating a comprehensive historical delisted-stock
dataset requires a paid vendor feed (e.g. CRSP, Sharadar) and is a Phase 5+
ingestion task, not something to fabricate here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "universe")

REQUIRED_STOCK_COLUMNS = {
    "symbol", "company_name", "exchange", "sector_code", "industry_code",
    "market_cap", "country", "active", "date_added", "date_removed", "delisted_reason",
}


@dataclass(frozen=True)
class Stock:
    symbol: str
    company_name: str
    exchange: str
    sector_code: str
    industry_code: str
    market_cap: float
    country: str
    active: bool
    date_added: date | None
    date_removed: date | None
    delisted_reason: str | None


class Universe:
    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.exchanges = self._read_csv("exchanges.csv")
        self.sectors = self._read_csv("sectors.csv")
        self.industries = self._read_csv("industries.csv")
        self.stocks = self._read_csv("stocks.csv")
        self._validate()

    def _read_csv(self, filename: str) -> pd.DataFrame:
        path = os.path.join(self.data_dir, filename)
        return pd.read_csv(path, comment="#", skip_blank_lines=True)

    def _validate(self) -> None:
        missing = REQUIRED_STOCK_COLUMNS - set(self.stocks.columns)
        if missing:
            raise ValueError(f"stocks.csv missing required columns: {sorted(missing)}")

        valid_exchanges = set(self.exchanges["exchange_code"])
        bad_exchange = set(self.stocks["exchange"]) - valid_exchanges
        if bad_exchange:
            raise ValueError(f"Unknown exchange code(s) in stocks.csv: {sorted(bad_exchange)}")

        valid_sectors = set(self.sectors["sector_code"])
        bad_sector = set(self.stocks["sector_code"]) - valid_sectors
        if bad_sector:
            raise ValueError(f"Unknown sector code(s) in stocks.csv: {sorted(bad_sector)}")

        valid_industries = set(self.industries["industry_code"])
        bad_industry = set(self.stocks["industry_code"]) - valid_industries
        if bad_industry:
            raise ValueError(f"Unknown industry code(s) in stocks.csv: {sorted(bad_industry)}")

        # Every industry must roll up to a sector that actually exists.
        merged = self.industries.merge(
            self.sectors, on="sector_code", how="left", indicator=True
        )
        orphans = merged.loc[merged["_merge"] == "left_only", "industry_code"]
        if len(orphans):
            raise ValueError(f"industries.csv references unknown sector_code(s): {list(orphans)}")

        dupes = self.stocks["symbol"][self.stocks["symbol"].duplicated()]
        if len(dupes):
            raise ValueError(f"Duplicate symbol(s) in stocks.csv: {list(dupes)}")

    def all_symbols(self, active_only: bool = True) -> list[str]:
        df = self.stocks
        if active_only:
            df = df[df["active"] == True]  # noqa: E712
        return sorted(df["symbol"].tolist())

    def as_of(self, as_of_date: date) -> list[str]:
        """
        Point-in-time universe membership: symbols whose [date_added, date_removed)
        window covers as_of_date. This is what backtests should use — not
        `all_symbols()` — to avoid survivorship bias.
        """
        df = self.stocks.copy()
        df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce").dt.date
        df["date_removed"] = pd.to_datetime(df["date_removed"], errors="coerce").dt.date

        def _covers(row) -> bool:
            added = row["date_added"]
            removed = row["date_removed"]
            if added is not None and added is not pd.NaT and as_of_date < added:
                return False
            if removed is not None and removed is not pd.NaT and as_of_date >= removed:
                return False
            return True

        mask = df.apply(_covers, axis=1)
        return sorted(df.loc[mask, "symbol"].tolist())

    def get(self, symbol: str) -> Stock:
        row = self.stocks.loc[self.stocks["symbol"] == symbol]
        if row.empty:
            raise KeyError(f"Unknown symbol: {symbol}")
        r = row.iloc[0]

        def _parse_date(v):
            if pd.isna(v):
                return None
            return datetime.strptime(str(v), "%Y-%m-%d").date()

        return Stock(
            symbol=r["symbol"],
            company_name=r["company_name"],
            exchange=r["exchange"],
            sector_code=r["sector_code"],
            industry_code=r["industry_code"],
            market_cap=float(r["market_cap"]),
            country=r["country"],
            active=bool(r["active"]),
            date_added=_parse_date(r["date_added"]),
            date_removed=_parse_date(r["date_removed"]),
            delisted_reason=None if pd.isna(r["delisted_reason"]) else r["delisted_reason"],
        )

    def sector_etf(self, sector_code: str) -> str:
        row = self.sectors.loc[self.sectors["sector_code"] == sector_code]
        if row.empty:
            raise KeyError(f"Unknown sector_code: {sector_code}")
        return row.iloc[0]["sector_etf"]

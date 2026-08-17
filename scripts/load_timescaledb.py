"""
Loads real data into the actually-running TimescaleDB/Postgres container:
universe reference tables (exchanges/sectors/industries/stocks), real OHLCV
bars (data/market/real/*.csv), and generated labels (data/processed/*.csv).

This is what turns database/schema.sql + database/models.py from "defined
but never exercised" into something with real rows in it — the schema and
ORM models were written in Phase 1 but nothing ever actually connected to
Postgres until this script.

Run schema.sql first (via psql, see docker-compose.yml's header comment),
then: DATABASE_URL=postgresql://postgres:postgres@localhost:5432/us_ai_signal_engine python scripts/load_timescaledb.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from calendar_us.trading_calendar import HORIZON_SESSIONS  # noqa: E402
from models.common import PROCESSED_DIR  # noqa: E402
from universe.universe_loader import Universe  # noqa: E402

REAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "market", "real")


def _engine():
    url = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}".format(
            user=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=os.environ.get("POSTGRES_PORT", "5432"),
            db=os.environ.get("POSTGRES_DB", "us_ai_signal_engine"),
        ),
    )
    return create_engine(url)


def load_universe(engine, universe: Universe) -> None:
    universe.exchanges.rename(columns={}).to_sql("exchanges", engine, if_exists="append", index=False)
    universe.sectors.to_sql("sectors", engine, if_exists="append", index=False)
    universe.industries.to_sql("industries", engine, if_exists="append", index=False)

    stocks = universe.stocks.rename(columns={"exchange": "exchange_code"}).copy()
    stocks["active"] = stocks["active"].astype(bool)
    stocks.to_sql("stocks", engine, if_exists="append", index=False)
    print(f"Loaded {len(universe.exchanges)} exchanges, {len(universe.sectors)} sectors, "
          f"{len(universe.industries)} industries, {len(stocks)} stocks")


def load_ohlcv(engine, symbols: list[str]) -> int:
    total = 0
    for symbol in symbols:
        path = os.path.join(REAL_DATA_DIR, f"{symbol}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=["date"])

        # yfinance occasionally returns an incomplete/still-forming bar for
        # the most recent session (fetched intraday, before the close print
        # settles) — NULL close/adj_close on exactly the last row. The
        # schema correctly requires NOT NULL on these columns; drop rather
        # than fail the whole load, since the day's real close will be
        # complete on the next fetch.
        required = ["open", "high", "low", "close", "adj_close", "volume"]
        incomplete = df[required].isna().any(axis=1)
        if incomplete.any():
            print(f"  {symbol}: skipping {incomplete.sum()} incomplete row(s) "
                  f"({df.loc[incomplete, 'date'].dt.date.tolist()})")
            df = df[~incomplete]

        rows = pd.DataFrame({
            "symbol": symbol,
            "session_date": df["date"].dt.date,
            "open": df["open"], "high": df["high"], "low": df["low"],
            "close": df["close"], "adj_close": df["adj_close"], "volume": df["volume"].astype("int64"),
            "source": "yfinance",
        })
        rows.to_sql("ohlcv_bars", engine, if_exists="append", index=False, method="multi", chunksize=1000)
        total += len(rows)
    print(f"Loaded {total} OHLCV bars for {len(symbols)} symbols")
    return total


def load_labels(engine, symbols: list[str]) -> int:
    total = 0
    for symbol in symbols:
        path = os.path.join(PROCESSED_DIR, f"{symbol}_labels.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=["date", "target_session_date"])
        df = df.dropna(subset=["label"])
        rows = pd.DataFrame({
            "symbol": symbol,
            "session_date": df["date"].dt.date,
            "horizon": df["horizon"],
            "future_return": df["future_return"],
            "threshold_used": df["threshold_used"],
            "labeling_method": df["labeling_method"],
            "label": df["label"],
            "target_session_date": df["target_session_date"].dt.date,
        })
        rows.to_sql("labels", engine, if_exists="append", index=False, method="multi", chunksize=1000)
        total += len(rows)
    print(f"Loaded {total} label rows for {len(symbols)} symbols")
    return total


def main():
    engine = _engine()
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("Connected to database.")

    universe = Universe()
    with engine.connect() as conn:
        existing = conn.execute(text("SELECT count(*) FROM stocks")).scalar()
    if existing and existing > 0:
        print(f"stocks table already has {existing} rows — skipping universe load (schema.sql ran once already?)")
    else:
        load_universe(engine, universe)

    symbols = universe.all_symbols(active_only=True)
    load_ohlcv(engine, symbols)
    load_labels(engine, symbols)

    with engine.connect() as conn:
        counts = {
            table: conn.execute(text(f"SELECT count(*) FROM {table}")).scalar()
            for table in ("exchanges", "sectors", "industries", "stocks", "ohlcv_bars", "labels")
        }
    print("Final row counts:", counts)


if __name__ == "__main__":
    main()

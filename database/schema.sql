-- Phase 1 schema: universe, calendar-aware OHLCV, corporate actions, labels.
-- Requires the TimescaleDB extension for hypertable partitioning on time-series
-- tables (ohlcv_bars, market_features, labels). Everything else is plain Postgres.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ── Reference / universe ─────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS exchanges (
    exchange_code   TEXT PRIMARY KEY,
    exchange_name   TEXT NOT NULL,
    mic             TEXT NOT NULL,
    country         TEXT NOT NULL,
    timezone        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sectors (
    sector_code     TEXT PRIMARY KEY,
    sector_name     TEXT NOT NULL,
    sector_etf      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS industries (
    industry_code   TEXT PRIMARY KEY,
    industry_name   TEXT NOT NULL,
    sector_code     TEXT NOT NULL REFERENCES sectors(sector_code)
);

-- Point-in-time universe membership lives in `stocks`, NOT just "today's list" —
-- date_added/date_removed/active let backtests reconstruct historical
-- membership and avoid survivorship bias. See universe/universe_loader.py.
CREATE TABLE IF NOT EXISTS stocks (
    symbol          TEXT PRIMARY KEY,
    company_name    TEXT NOT NULL,
    exchange_code   TEXT NOT NULL REFERENCES exchanges(exchange_code),
    sector_code     TEXT NOT NULL REFERENCES sectors(sector_code),
    industry_code   TEXT NOT NULL REFERENCES industries(industry_code),
    market_cap      NUMERIC,
    country         TEXT NOT NULL,
    active          BOOLEAN NOT NULL DEFAULT TRUE,
    date_added      DATE,
    date_removed    DATE,
    delisted_reason TEXT
);

-- Symbol history: tracks ticker renames (e.g. FB -> META) so historical bars
-- filed under an old ticker can still be joined to the current company record.
CREATE TABLE IF NOT EXISTS symbol_changes (
    id              SERIAL PRIMARY KEY,
    old_symbol      TEXT NOT NULL,
    new_symbol      TEXT NOT NULL,
    effective_date  DATE NOT NULL,
    reason          TEXT
);

-- ── Price data ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ohlcv_bars (
    symbol          TEXT NOT NULL,
    session_date    DATE NOT NULL,
    open            NUMERIC NOT NULL,
    high            NUMERIC NOT NULL,
    low             NUMERIC NOT NULL,
    close           NUMERIC NOT NULL,
    adj_close       NUMERIC NOT NULL,
    volume          BIGINT NOT NULL,
    source          TEXT NOT NULL DEFAULT 'unknown',
    PRIMARY KEY (symbol, session_date)
);
SELECT create_hypertable('ohlcv_bars', 'session_date', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol ON ohlcv_bars (symbol, session_date DESC);

-- Raw corporate actions, used to reconcile/validate adjusted prices and to
-- exclude action-driven jumps from being mistaken for real returns.
CREATE TABLE IF NOT EXISTS corporate_actions (
    id              SERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL,
    action_date     DATE NOT NULL,
    action_type     TEXT NOT NULL CHECK (action_type IN
                        ('SPLIT', 'REVERSE_SPLIT', 'DIVIDEND', 'SPINOFF', 'MERGER', 'ACQUISITION', 'TICKER_CHANGE')),
    ratio           NUMERIC,          -- e.g. split ratio (4.0 for a 4-for-1 split)
    cash_amount     NUMERIC,          -- e.g. dividend per share
    details         TEXT
);
CREATE INDEX IF NOT EXISTS idx_corp_actions_symbol ON corporate_actions (symbol, action_date);

-- ── US market / benchmark context ──────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS market_features (
    benchmark       TEXT NOT NULL,      -- SPY, QQQ, IWM, DIA
    session_date    DATE NOT NULL,
    return_1d       NUMERIC,
    rsi_14          NUMERIC,
    volatility_20d  NUMERIC,
    trend           TEXT,               -- UP / DOWN / FLAT (vs SMA50/SMA200)
    PRIMARY KEY (benchmark, session_date)
);
SELECT create_hypertable('market_features', 'session_date', if_not_exists => TRUE);

-- ── Labels (seven horizons) ─────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS labels (
    symbol            TEXT NOT NULL,
    session_date      DATE NOT NULL,
    horizon           TEXT NOT NULL CHECK (horizon IN ('1D','1W','2W','1M','2M','3M','6M')),
    future_return     NUMERIC,
    threshold_used    NUMERIC,
    labeling_method   TEXT NOT NULL,   -- fixed | atr | volatility | market_relative | risk_adjusted
    label             TEXT NOT NULL CHECK (label IN ('BUY','HOLD','SELL')),
    target_session_date DATE,          -- close[T + horizon]'s actual session date
    PRIMARY KEY (symbol, session_date, horizon, labeling_method)
);
SELECT create_hypertable('labels', 'session_date', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_labels_symbol_horizon ON labels (symbol, horizon, session_date DESC);

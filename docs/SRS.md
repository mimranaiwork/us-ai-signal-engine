# Software Requirements Specification
## US AI Signal Engine — Multi-Horizon Stock BUY/HOLD/SELL Signal System

**Version:** 1.0
**Status:** Living document — reflects the system as built and verified, not the original spec's aspirations
**Last updated:** 2026-08-13

---

## 1. Introduction

### 1.1 Purpose

This document specifies the functional and non-functional requirements for the US AI Signal Engine: a system that produces multi-horizon (1 day to 6 months) BUY/HOLD/SELL trading signals for US-listed equities, using an ensemble of machine-learning models trained on real market, fundamental, and sentiment data, and exposes those signals through a REST API and a web frontend.

It is written after the system was built, not before — every requirement below is either **Implemented** (built, tested, verified against real data) or explicitly marked **Not implemented** / **Partially implemented** with the reason. This is a deliberate departure from a conventional pre-build SRS: the original design brief (a 55-section prompt) is treated as the requirements *source*, and this document is the requirements *record* of what actually exists.

### 1.2 Scope

The system covers US equities on NYSE, Nasdaq, and NYSE American. In its current form it operates over a **curated, hand-maintained universe of 110 symbols (an S&P 100-anchored large-cap set)** (see Appendix A) — it is explicitly not a scanner over the full ~6,000-symbol US market; that would require different (always-on, higher-volume) data infrastructure than what's built.

In scope:
- Point-in-time-correct historical data ingestion (OHLCV, fundamentals, corporate actions)
- Feature engineering (technical, price, market, sector, industry, macro, options, sentiment)
- Multi-horizon label generation
- Four independent model architectures (XGBoost, LightGBM, LSTM, Transformer) plus a stacking meta-model
- Ensemble scoring, confidence calibration, and regime detection
- Walk-forward backtesting and validation
- A REST API and a React frontend (watchlist, screener, charting, track record)
- Scheduled nightly data refresh and model retraining

Out of scope (see §6 for detail): true portfolio/position management, order execution, personalized investment advice, real-time (sub-daily) market data, and coverage beyond the 110-symbol universe without manual expansion.

### 1.3 Definitions and Acronyms

| Term | Meaning |
|---|---|
| Horizon | A forward-looking trading-session window: 1D=1, 1W=5, 2W=10, 1M=21, 2M=42, 3M=63, 6M=126 sessions |
| Signal | The final BUY / HOLD / SELL classification for a symbol at a point in time |
| Composite score | Horizon-weighted average of per-horizon scores, range [-1, +1] |
| Model consensus | 1 − stddev of the four base models' scores for a horizon (agreement measure) |
| Horizon consensus | 1 − stddev of the composite score across horizons |
| Point-in-time correctness | No feature or label may use information not actually available as of its timestamp (the central data-integrity discipline of this project) |
| Walk-forward validation | Expanding-window retrain/validate/test, as opposed to a single static split |
| Stacking | A logistic-regression meta-model that blends the four base models' outputs, as an alternative to a fixed weighted average |
| Pivot levels | Classic daily support/resistance (R1–R3, S1–S3) computed from the prior session's H/L/C |

### 1.4 References

- Original design brief: "Claude AI Prompt — US Stock Multi-Horizon AI BUY/HOLD/SELL Engine" (55 sections; internal, not reproduced here)
- `README.md` — phase-by-phase build narrative and honesty notes
- `docs/TECHNICAL_DESIGN.md` — companion document to this SRS

---

## 2. Overall Description

### 2.1 Product Perspective

The system is a standalone application, not an extension of an existing product. It has three deployable components:

1. **Backend** — Python/FastAPI service (`api/`) backed by a feature/label pipeline, four trained model architectures, and a Redis cache.
2. **Frontend** — a React/Vite single-page app (`frontend/`) consuming the backend's REST API.
3. **Scheduled jobs** — a Celery worker + beat scheduler that refreshes data and retrains models nightly.

Supporting infrastructure: PostgreSQL/TimescaleDB (OHLCV storage, exercised via `scripts/load_timescaledb.py`), Redis (response caching + Celery broker), MLflow (experiment tracking).

### 2.2 Product Functions (summary)

- Ingest real OHLCV, fundamentals, and corporate-actions data for the tracked universe
- Compute ~150 engineered features per symbol per session (technical, price, market-relative, sector, industry, macro, options-proxy, sentiment, pivot levels)
- Generate seven-horizon volatility-adjusted BUY/HOLD/SELL labels
- Train and serve four independent model architectures plus a stacking ensemble
- Produce a single, explainable signal per symbol with confidence, consensus, regime, and risk fields
- Backtest and walk-forward-validate every model and ensembling method against real data
- Serve signals, historical prices, a full-universe screener, and validation results via REST
- Present all of the above in a TradingView-styled web UI
- Refresh data and retrain models on a nightly schedule, automatically
- Screen the tracked universe for a mechanical, unofficial approximation of Sharia compliance (business activity + AAOIFI-style financial ratios)

### 2.3 User Classes and Characteristics

| User class | Description | Primary interface |
|---|---|---|
| Interactive trader/analyst | Wants a signal card, charts, and a watchlist for symbols they follow | Frontend — Watchlist view |
| Screener user | Wants to rank the whole tracked universe by signal/gain/breakout | Frontend — Screener view |
| System evaluator | Wants to see whether the models actually work, historically | Frontend — Track Record view |
| Integrator | Wants programmatic access to signals/prices/screener data | REST API directly |
| Operator | Runs and monitors the data pipeline, retraining, and drift | CLI scripts, Celery, `/api/v1/monitoring/drift/{symbol}` |

No user class requires authentication in the current build — see §4.3 (Security) for what that implies and doesn't imply.

### 2.4 Operating Environment

- **Backend:** Python 3.12, FastAPI + Uvicorn, tested on macOS (Intel) in this build; Docker images build for Linux via `docker-compose.yml`
- **Frontend:** any modern browser; built with Vite, no IE/legacy support
- **Data store:** PostgreSQL 15+/TimescaleDB extension; Redis 7+
- **ML runtime:** XGBoost, LightGBM, PyTorch (CPU; MPS/Metal avoided in the scheduled-job path — see §4.2)
- **External services:** Yahoo Finance (via `yfinance`, unauthenticated), SEC EDGAR (requires a configured User-Agent, no API key)

### 2.5 Design and Implementation Constraints

- **No paid data vendors.** Every data source is free (yfinance, SEC EDGAR). This directly constrains what's achievable — see §6.
- **Small, curated universe by design**, not a technical limitation of the codebase — the pipeline itself is symbol-agnostic and has been extended from 40 → 46 → 110 symbols on request without architecture changes.
- **CPU-only ML on the current deployment target.** LSTM/Transformer training is deliberately kept small (hidden_size=32, 2–3 epochs in scheduled/CI paths) to stay tractable without a GPU.
- **Intel Mac PyTorch ceiling**: torch pinned to 2.2.2, numpy<2, scipy<1.14 for ABI compatibility on the development machine used to build this.

### 2.6 Assumptions and Dependencies

- yfinance's free endpoints remain available and stable in schema (a real risk — yfinance's news payload shape has already changed once historically; code defensively handles both shapes where it matters)
- SEC EDGAR's submissions API remains free and unauthenticated (User-Agent header required, no key)
- Users understand this is a research/decision-support tool, not investment advice (see §6, Non-Goals)

---

## 3. Functional Requirements

Each requirement is tagged **[✓ Implemented]**, **[~ Partial]**, or **[✗ Not implemented]**. IDs are stable identifiers for cross-reference with the Technical Design document.

### FR-1 — Universe & Reference Data

| ID | Requirement | Status |
|---|---|---|
| FR-1.1 | The system shall maintain a survivorship-bias-aware universe (symbol, sector, industry, exchange, market cap, active flag, date added/removed, delisted reason) | ✓ Implemented — `data/universe/stocks.csv`, `universe/universe_loader.py` |
| FR-1.2 | The system shall support point-in-time universe membership queries (`as_of(date)`) | ✓ Implemented |
| FR-1.3 | The universe shall be extensible without code changes (add a CSV row + fetch data) | ✓ Implemented — demonstrated live: INTC, MRVL, ARM, WDAY, TEAM added post-launch |
| FR-1.4 | The system shall expose universe reference data via API | ✓ Implemented — `GET /api/v1/universe`, `GET /api/v1/universe/{symbol}` |

### FR-2 — Data Ingestion

| ID | Requirement | Status |
|---|---|---|
| FR-2.1 | The system shall ingest daily OHLCV bars for every tracked symbol plus benchmark indices (SPY/QQQ/IWM/DIA) and sector ETFs | ✓ Implemented — `ingestion/ohlcv_source.py::YFinanceSource`, `scripts/fetch_real_data.py` |
| FR-2.2 | The system shall ingest SEC EDGAR fundamentals (10-K/10-Q) with point-in-time filing dates | ✓ Implemented — `ingestion/sec_edgar.py` |
| FR-2.3 | The system shall ingest macro/market-indicator series (VIX, 10Y yield, DXY, Gold, Oil, Copper, Bitcoin) | ✓ Implemented — `scripts/fetch_macro_data.py` |
| FR-2.4 | The system shall ingest real-time news headlines for sentiment display | ✓ Implemented — `features/sentiment/news_sentiment.py::fetch_recent_news` (live, current-only) |
| FR-2.5 | The system shall ingest options chain data (IV, put/call ratio) | ~ Partial — machinery is real and tested; underlying data is a documented synthetic proxy (no free options vendor exists) |
| FR-2.6 | The system shall ingest macro **economic** releases (CPI, GDP, unemployment) | ✗ Not implemented — requires FRED with point-in-time vintage handling; explicitly not faked |
| FR-2.7 | Incomplete/still-forming trading-session rows shall be detected and excluded, not silently included as NULL or zero | ✓ Implemented — fixed in three independent places during development (TimescaleDB loader, prices API, chart rendering) after being caught as a real bug |

### FR-3 — Feature Engineering

| ID | Requirement | Status |
|---|---|---|
| FR-3.1 | Compute standard technical indicators: SMA/EMA (5–200), RSI (7/14/21), MACD, Bollinger Bands, ATR, ADX/+DI/-DI, stochastic, Williams %R, CCI, MFI, OBV, volume ratio | ✓ Implemented — `features/technical/indicators.py` |
| FR-3.2 | Compute classic daily pivot points (Pivot, R1–R3, S1–S3) and a breakout-proximity assessment with target/expected-gain | ✓ Implemented — `features/technical/pivot_points.py` |
| FR-3.3 | Compute price-derived features (returns, gap, range, rolling volatility) — no raw price levels used as model inputs | ✓ Implemented — `features/price/price_features.py` |
| FR-3.4 | Compute market-relative features (relative strength vs. SPY/QQQ/IWM/DIA, market regime inputs) | ✓ Implemented — `features/market/market_features.py` |
| FR-3.5 | Compute market breadth (advancers/decliners, % above SMA50/200, new highs/lows) across the universe | ✓ Implemented — `features/market/market_breadth.py` |
| FR-3.6 | Compute sector-relative features (stock vs. its sector ETF) | ✓ Implemented — `features/sector/sector_features.py` |
| FR-3.7 | Compute industry peer-group relative features | ✓ Implemented — `features/industry/industry_features.py`, excludes industries with <2 members rather than fabricating a peer average |
| FR-3.8 | Compute macro-relative features (VIX regime, macro series changes joined onto each symbol) | ✓ Implemented — `features/macro/macro_features.py` |
| FR-3.9 | Compute fundamentals ratios (P/E, margins, growth) with point-in-time filing-date joins | ✓ Implemented — `features/fundamentals/fundamentals.py` |
| FR-3.10 | Compute earnings timing (days since/to next earnings, real or cadence-estimated, with an actual calendar date) | ✓ Implemented — `features/earnings/earnings.py` |
| FR-3.11 | Compute options-derived sentiment (put/call ratio, IV rank/percentile) | ~ Partial — synthetic proxy, documented |
| FR-3.12 | Compute live news sentiment (lexicon-based scoring + topic classification) | ✓ Implemented — `features/sentiment/news_sentiment.py` |
| FR-3.13 | No feature shall use information not available as of its own timestamp (no leakage) | ✓ Implemented and tested — dedicated leakage tests across fundamentals, earnings, sentiment, sequence windowing |

### FR-4 — Labeling

| ID | Requirement | Status |
|---|---|---|
| FR-4.1 | Generate BUY/HOLD/SELL labels for seven horizons (1D…6M) | ✓ Implemented — `labels/horizons.py` |
| FR-4.2 | Support multiple labeling methodologies (fixed threshold, ATR-based, volatility-adjusted, market-relative, risk-adjusted) | ✓ Implemented — all five in `labels/horizons.py`; `volatility` is the configured default |
| FR-4.3 | Labels shall use trading-session counts, not calendar days | ✓ Implemented — `calendar_us/trading_calendar.py` |

### FR-5 — Model Training

| ID | Requirement | Status |
|---|---|---|
| FR-5.1 | Train an XGBoost classifier per horizon, pooled cross-sectionally across the universe | ✓ Implemented — `models/xgboost/model.py` |
| FR-5.2 | Train a LightGBM classifier per horizon, pooled cross-sectionally | ✓ Implemented — `models/lightgbm/model.py` |
| FR-5.3 | Train an LSTM sequence model jointly across all seven horizons (shared encoder, per-horizon heads) | ✓ Implemented — `models/lstm/model.py` |
| FR-5.4 | Train a Transformer sequence model, same multi-task design | ✓ Implemented — `models/transformer/model.py` |
| FR-5.5 | Train a stacking meta-model (logistic regression over the four base models' outputs) per horizon, as an ensemble alternative | ✓ Implemented — `signals/stacking.py`; empirically wins 3 of 7 horizons vs. the fixed weighted average |
| FR-5.6 | A PPO reinforcement-learning agent for position sizing | ✗ Not implemented — explicitly out of scope per the source spec (directional classification is not PPO's intended role); a config weight placeholder exists but is inert |
| FR-5.7 | Raw price-level columns (OHLC, pivot levels) shall never be used as model training features | ✓ Implemented and tested — `models/common.py::EXCLUDE_COLUMNS`; caught and fixed as a real bug when pivot levels were first added |

### FR-6 — Ensemble & Signal Generation

| ID | Requirement | Status |
|---|---|---|
| FR-6.1 | Combine per-horizon, per-model probabilities into a single per-horizon score via P(BUY) − P(SELL) | ✓ Implemented — `signals/scoring.py` |
| FR-6.2 | Combine per-horizon scores into one composite score via horizon-weighted average | ✓ Implemented, walk-forward-optimized weights |
| FR-6.3 | Compute model consensus and horizon consensus as agreement measures | ✓ Implemented — `signals/consensus.py` |
| FR-6.4 | Compute a calibrated confidence score from consensus, regime clarity, and data quality | ✓ Implemented — `signals/confidence.py` |
| FR-6.5 | Detect market regime (Bull/Bear/Sideways/High-Volatility) from benchmark trend + breadth | ✓ Implemented — `signals/market_regime.py` |
| FR-6.6 | Detect stock-specific regime (trending/sideways/high-volatility) independent of the market regime | ✓ Implemented — `signals/stock_regime.py` |
| FR-6.7 | Convert composite score to BUY/HOLD/SELL via configurable, walk-forward-optimized thresholds | ✓ Implemented |
| FR-6.8 | Apply conservative HOLD overrides (low consensus, low confidence, uncertain regime, insufficient history/liquidity, imminent earnings) | ✓ Implemented — `signals/signal_engine.py` |
| FR-6.9 | Per-horizon, opt in to the stacking meta-model only where it's empirically shown to outperform, falling back to the weighted average elsewhere or on missing inputs | ✓ Implemented |
| FR-6.10 | Provide SHAP-based top reasons/risks for a signal | ✓ Implemented — `explainability/shap.py`, native XGBoost `pred_contribs` |
| FR-6.11 | Flag risk level, incorporating earnings-imminence and volatility | ✓ Implemented |
| FR-6.12 | Provide a breakout assessment (nearest support/resistance, near-breakout flag, target, expected % gain) | ✓ Implemented — `features/technical/pivot_points.py::breakout_assessment` |

### FR-7 — Backtesting & Validation

| ID | Requirement | Status |
|---|---|---|
| FR-7.1 | Simulate a long-only trading strategy from model signals, with transaction costs, vs. Buy & Hold and benchmarks | ✓ Implemented — `backtesting/engine.py` |
| FR-7.2 | Report classification and trading metrics (accuracy, F1, Sharpe, Sortino, max drawdown, win rate, profit factor) | ✓ Implemented — `backtesting/metrics.py` |
| FR-7.3 | Walk-forward validate XGBoost/LightGBM across expanding-window folds | ✓ Implemented — `backtesting/walk_forward.py::evaluate_fold` |
| FR-7.4 | Walk-forward validate LSTM/Transformer across expanding-window folds (genuine per-fold retraining, not a single static split) | ✓ Implemented for a representative subset (3 of 7 horizons, 2 most recent folds) — full sweep is architecturally supported but not run by default due to cost |
| FR-7.5 | Grid-search horizon weights, model weights, and BUY/SELL thresholds against Sharpe, confirmed on a held-out window | ✓ Implemented — `backtesting/optimize.py` |
| FR-7.6 | Compare the stacking ensemble against the fixed weighted average on genuinely held-out data | ✓ Implemented — `scripts/train_stacking.py` |

### FR-8 — API

| ID | Requirement | Status |
|---|---|---|
| FR-8.1 | `GET /health` — liveness + loaded-model check | ✓ |
| FR-8.2 | `GET /api/v1/universe`, `GET /api/v1/universe/{symbol}` — universe reference data | ✓ |
| FR-8.3 | `GET /api/v1/signals/{symbol}` — the full signal (cached, 5 min TTL) | ✓ |
| FR-8.4 | `GET /api/v1/prices/{symbol}` — historical OHLCV + RSI + Bollinger + ATR + pivot levels for charting | ✓ |
| FR-8.5 | `GET /api/v1/quotes?symbols=...` — lightweight batch last-price/change for watchlists, without running the ML ensemble | ✓ |
| FR-8.6 | `GET /api/v1/screener` — signals for the entire tracked universe in one call (cached, 10 min TTL) | ✓ |
| FR-8.7 | `GET /api/v1/track-record` — walk-forward/backtest/stacking-comparison results | ✓ |
| FR-8.8 | `GET /api/v1/monitoring/drift/{symbol}` — PSI-based feature drift vs. history | ✓ |
| FR-8.9 | `GET /api/v1/calendar/sessions` — trading sessions in a date range | ✓ |
| FR-8.10 | `GET /api/v1/compliance/sharia` — full Sharia-compliance screen report for the tracked universe | ✓ Implemented — see FR-11 |
| FR-8.11 | API authentication / rate limiting | ✗ Not implemented — see §4.3 |

### FR-9 — Frontend

| ID | Requirement | Status |
|---|---|---|
| FR-9.1 | Watchlist view: symbol card with live price/change/%change per row | ✓ Implemented |
| FR-9.2 | Signal card: BUY/HOLD/SELL badge, composite score gauge, per-horizon breakdown | ✓ Implemented |
| FR-9.3 | Price chart: candlestick (daily) with volume subplot | ✓ Implemented |
| FR-9.4 | RSI indicator sub-panel | ✓ Implemented |
| FR-9.5 | Bollinger Bands overlay on the price chart | ✓ Implemented |
| FR-9.6 | Chart type toggle: Candlestick / Line / Renko / Point & Figure | ✓ Implemented — Renko and P&F are genuine price-based (not time-based) constructions, not re-skinned line charts |
| FR-9.7 | Support/resistance ladder with breakout target and expected gain % | ✓ Implemented |
| FR-9.8 | Live news & sentiment panel | ✓ Implemented |
| FR-9.9 | Key stats panel: sector, market cap, volume, next earnings date | ✓ Implemented |
| FR-9.10 | Screener view: sortable/filterable table across the whole tracked universe, ranked by gain %/signal/breakout | ✓ Implemented |
| FR-9.10a | Screener Sharia-compliance column, "compliant only" filter, and mandatory methodology disclaimer | ✓ Implemented — see FR-11 |
| FR-9.11 | Track Record view: real historical backtest/walk-forward/stacking results | ✓ Implemented |
| FR-9.12 | Auto-refresh (watchlist every 5 min, screener every 2 min), user-toggleable | ✓ Implemented |
| FR-9.13 | User accounts, saved preferences across devices | ✗ Not implemented — watchlist persists to `localStorage` only |

### FR-10 — Scheduled Operations

| ID | Requirement | Status |
|---|---|---|
| FR-10.1 | Nightly data refresh (re-fetch OHLCV + macro, rebuild features, re-augment) | ✓ Implemented — `training/tasks.py::refresh_data`, verified end-to-end |
| FR-10.2 | Nightly model retraining | ✓ Implemented — `training/tasks.py::retrain_all` |
| FR-10.3 | Retraining shall run only after that night's data refresh completes, never on stale data or racing an independent schedule | ✓ Implemented — `training/tasks.py::nightly_pipeline` chains them in-process |
| FR-10.4 | Cached API responses shall be invalidated after a data refresh | ✓ Implemented — `monitoring/cache.py::clear_pattern` |
| FR-10.5 | Scheduled jobs shall not crash on the deployment host | ✓ Implemented — fixed a real macOS PyTorch/fork incompatibility during development (Celery `worker_pool="solo"`) |
| FR-10.6 | Intraday (sub-daily) data refresh | ✗ Not implemented by design — the underlying data source (daily OHLCV bars) doesn't change intraday, so more frequent polling would not produce different results |

### FR-11 — Sharia Compliance Screening

| ID | Requirement | Status |
|---|---|---|
| FR-11.1 | Screen each universe symbol on business activity (exclude conventional banking/lending, insurance, tobacco, alcohol, gambling, adult entertainment, pork products by industry code) | ✓ Implemented — `features/compliance/sharia_screen.py::business_activity_screen` |
| FR-11.2 | Screen each non-excluded symbol on AAOIFI-style financial ratios computed from real SEC XBRL data: debt/market-cap < 33%, (cash + short-term investments)/market-cap < 33%, interest-income/revenue < 5% | ✓ Implemented — `features/compliance/sharia_screen.py::financial_ratio_screen` |
| FR-11.3 | A ratio that cannot be computed from available filing data shall be reported as unknown, never silently assumed to pass or fail | ✓ Implemented and unit-tested (`tests/test_sharia_screen.py`) |
| FR-11.4 | Results for finance-adjacent, low-XBRL-coverage industries (asset managers, payment networks) whose filings use bank-style tags this screen doesn't recognize shall carry a materially stronger warning than an ordinary missing-data case | ✓ Implemented — `FINANCE_ADJACENT_LOW_COVERAGE_INDUSTRIES`; regression-tested against a real finding (GS/BLK/V/MA all returned zero computable ratios) |
| FR-11.5 | The screen shall be run as a batch job against SEC EDGAR (rate-limited, quarterly-cadence data), not live per API request | ✓ Implemented — `scripts/run_sharia_screen.py`, output cached to `models/saved/sharia_screen_report.json` |
| FR-11.6 | Every surface presenting a result (API response, UI) shall carry an explicit, un-missable disclaimer that this is an unofficial mechanical approximation, not a certified Sharia compliance determination by a qualified Islamic finance board | ✓ Implemented — `methodology_note` field in both endpoints; mandatory footer disclaimer in `Screener.jsx` |

---

## 4. Non-Functional Requirements

### 4.1 Performance

- A single signal request completes in low single-digit seconds on cache miss (four-model inference + SHAP); <50ms on cache hit.
- A full-universe screener scan (110 symbols) completes in ~30 seconds on cache miss; served from a 10-minute cache otherwise.
- Nightly data refresh (~125 tickers incl. benchmarks/sector ETFs/macro) completes in ~2.5 minutes; full refresh + retrain (including LSTM/Transformer) in under 10 minutes on the CPU-only development machine.

### 4.2 Reliability & Availability

- Every external dependency (SEC EDGAR, news fetch, market breadth, Redis) degrades gracefully on failure — the system never fails a signal request because a non-essential enrichment is unavailable; it omits that field and logs the reason.
- Redis is an optimization, not a hard dependency: the API functions correctly, just slower, with Redis down.
- The scheduled-job runtime shall not crash the process on platform-specific incompatibilities (see FR-10.5).

### 4.3 Security

- No user authentication or authorization is implemented on the API or frontend. This is an explicit, current-state limitation, not a design goal — see §6.
- No secrets are committed to the repository; SEC EDGAR requires a User-Agent (not a secret) via environment variable; no other credentials are required by any integrated data source.
- CORS is currently open (`allow_origins=["*"]`) for local development; this must be restricted before any non-local deployment.

### 4.4 Data Integrity

- Point-in-time correctness is enforced and tested at every stage that could otherwise leak future information: fundamentals (filing-date joins), earnings (forward-projected estimates only), sentiment (as-of windowing), sequence windowing (trailing-only).
- Survivorship bias is addressed structurally (delisted symbols retained with `active=false` and a reason), though the delisted-symbol sample data is illustrative, not a comprehensive historical delisting dataset.

### 4.5 Maintainability

- 194 automated tests across 25 files, covering every feature/label/model/ensemble/API-adjacent module; run in ~13 seconds.
- Every fixed bug during development has an accompanying regression test (see `docs/TECHNICAL_DESIGN.md` §13 for the full list).
- Configuration (horizon weights, model weights, thresholds, labeling method) lives in `config/settings.yaml`, not hardcoded, and is re-derivable via `backtesting/optimize.py`.

### 4.6 Usability

- The frontend requires no installation beyond `npm install` + a running backend; no build step is required for local development (Vite dev server with HMR).
- Every "why" in the UI is documented inline where the number could be misread (e.g., pivot levels are explicitly labeled as a geometric read, not a model prediction; news sentiment is explicitly labeled as display-only).

---

## 5. External Interface Requirements

### 5.1 REST API

Base URL: `http://localhost:8000` (configurable via `VITE_API_BASE` on the frontend). All responses are JSON. See §3, FR-8 for the endpoint list; see `docs/TECHNICAL_DESIGN.md` §9 for full request/response schemas.

### 5.2 Frontend UI

Single-page React application, three primary views (Watchlist, Screener, Track Record) reachable via sidebar tabs, styled dark-theme/TradingView-inspired. See `docs/TECHNICAL_DESIGN.md` §10 for the component tree.

### 5.3 Third-Party Integrations

| Service | Purpose | Auth |
|---|---|---|
| Yahoo Finance (`yfinance`) | OHLCV, macro series, live news | None |
| SEC EDGAR | 10-K/10-Q fundamentals, filing dates | User-Agent header (email), no key |
| Redis | Response cache, Celery broker/backend | None (local) |
| MLflow | Training run tracking | None (local) |

---

## 6. Non-Goals and Known Limitations

Stated explicitly so they are never mistaken for oversights:

1. **Not investment advice.** The system produces model outputs, not recommendations; no part of it is intended to satisfy suitability, fiduciary, or regulatory requirements for investment advice.
2. **Not a full-market scanner.** 110 symbols by design; expanding it is a manual, per-symbol data-onboarding step, not a config flag.
3. **Not real-time.** Daily-bar granularity; nightly refresh; no intraday quotes or streaming data.
4. **Not authenticated.** No user accounts, no API keys, no rate limiting — appropriate for local/research use, not for public deployment as-is.
5. **No real options data.** Synthetic proxy only; the feature machinery is real, the inputs are not.
6. **No real economic-release macro data** (CPI, GDP, unemployment) — would need FRED with point-in-time vintage handling.
7. **PPO / position sizing is not built** — a directional PPO classifier was explicitly rejected as out of scope; the config weight is a placeholder for a future portfolio-allocation agent, not a classifier.
8. **Backtested performance is not a guarantee of future performance** — the models show a persistent long bias learned from a mostly-uptrending 2018–2026 training period; this is disclosed, not hidden, in `config/settings.yaml`.

---

## Appendix A — Tracked Universe (110 symbols)

AAPL, ABBV, ABT, ACN, ADBE, AMAT, AMD, AMGN, AMT, AMZN, ARM, AVGO, AXP, BA, BAC, BKNG, BLK, BMY, BNY, BRK-B, C, CAT, CL, CMCSA, COF, COP, COST, CRM, CSCO, CVS, CVX, DAL, DE, DHR, DIS, DOW, DUK, EMR, FDX, GD, GE, GEV, GILD, GM, GOOGL, GS, HD, HON, IBM, INTC, INTU, ISRG, JNJ, JPM, KO, LIN, LLY, LMT, LOW, LRCX, MA, MCD, MDLZ, MDT, META, MMM, MO, MRK, MRVL, MS, MSFT, MU, NBIS, NEE, NFLX, NKE, NOW, NVDA, ORCL, PANW, PEP, PFE, PG, PLD, PLTR, PM, QCOM, RTX, SBUX, SCHW, SHOP, SO, SPG, T, TEAM, TMO, TMUS, TSLA, TXN, UBER, UNH, UNP, UPS, USB, V, VZ, WDAY, WFC, WMT, XOM

Anchored on the S&P 100 index (fetched live, August 2026) plus pre-existing hand-picked names that predate that expansion and aren't themselves S&P 100 constituents: NBIS, ARM, WDAY, TEAM, DAL, SHOP, PANW, MRVL.

Plus two illustrative delisted entries (LEHMQ, FB→META) demonstrating the survivorship-bias-aware schema — not a comprehensive historical delisting dataset.

## Appendix B — Configuration Snapshot

See `config/settings.yaml` for the authoritative, current values (horizon weights, model weights, signal thresholds, liquidity floor, earnings risk window) — reproduced here would drift out of sync with reality faster than this document is updated.

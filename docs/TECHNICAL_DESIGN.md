# Technical Design Document
## US AI Signal Engine

**Companion to:** `docs/SRS.md`
**Last updated:** 2026-08-13

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          SCHEDULED (Celery, nightly)                     │
│  refresh_data → build_phase1_dataset → augment_features → retrain_all    │
└──────────────────────────────┬────────────────────────────────────────┘
                                 │ writes
                                 ▼
   ┌───────────────┐   ┌─────────────────┐   ┌──────────────────┐
   │ yfinance /     │──▶│ data/market/    │──▶│ features/*        │
   │ SEC EDGAR      │   │ real/*.csv      │   │ (technical, price, │
   └───────────────┘   └─────────────────┘   │  market, sector,   │
                                 │             │  industry, macro,  │
                                 ▼             │  fundamentals,     │
                        ┌─────────────────┐    │  earnings, options,│
                        │ data/processed/ │◀──│  sentiment, pivot)  │
                        │ {symbol}_       │   └──────────────────┘
                        │ features.csv /  │
                        │ _labels.csv     │
                        └────────┬────────┘
                                 │ read by
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
        ┌────────────┐  ┌──────────────┐  ┌─────────────┐
        │ models/     │  │ signals/      │  │ backtesting/ │
        │ xgboost,    │─▶│ signal_engine │  │ (walk-forward,│
        │ lightgbm,   │  │ (ensemble +   │  │  optimize)    │
        │ lstm,       │  │  regime +     │  └─────────────┘
        │ transformer,│  │  confidence)  │
        │ stacking    │  └───────┬──────┘
        └────────────┘          │
                                  ▼
                        ┌─────────────────┐        ┌──────────┐
                        │ api/main.py      │◀──────▶│ Redis     │
                        │ (FastAPI)        │        │ (cache)   │
                        └────────┬────────┘        └──────────┘
                                  │ REST
                                  ▼
                        ┌─────────────────┐
                        │ frontend/         │
                        │ (React + Vite)     │
                        └─────────────────┘
```

### 1.1 Component boundaries

| Component | Responsibility | Does NOT do |
|---|---|---|
| `ingestion/` | Pull raw data from external sources into a canonical shape | Feature computation |
| `features/` | Turn raw data into model-ready columns | Labeling, model training |
| `labels/` | Turn future price paths into BUY/HOLD/SELL targets | Feature computation |
| `models/` | Train/predict per architecture | Ensembling |
| `signals/` | Combine model outputs into one signal | Model training |
| `backtesting/` | Historical validation | Live serving |
| `api/` | HTTP surface | Business logic (delegates to `signals/`, `features/`) |
| `training/` | Orchestration + scheduling | Feature/model logic (imports from `models/`, `scripts/`) |
| `frontend/` | Presentation | Any computation the backend should own |

---

## 2. Technology Stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI + Uvicorn | Async-capable, typed, minimal ceremony for a small surface |
| Tree models | XGBoost 2.x, LightGBM 4.x | Industry-standard gradient boosting, native NaN handling, fast CPU inference |
| Sequence models | PyTorch (CPU) | LSTM/Transformer encoders, shared multi-task heads |
| Meta-model | scikit-learn `LogisticRegression` | Simple, transparent, cheap to retrain |
| Cache/broker | Redis | Single piece of infra serving both response caching and Celery |
| Scheduler | Celery (worker + beat) | Standard Python task scheduling, already paired with Redis |
| Time-series storage | PostgreSQL + TimescaleDB extension | Hypertable for OHLCV; exercised, not required for the API's hot path (which reads CSV) |
| Experiment tracking | MLflow | Local file-backed tracking of training runs |
| Frontend | React 18 + Vite, no UI framework | Hand-rolled SVG charts, no charting-library dependency |
| Testing | pytest | 209 tests, ~17s full run |

---

## 3. Data Architecture

### 3.1 Directory layout (data-relevant)

```
data/
  universe/           stocks.csv, sectors.csv, industries.csv, exchanges.csv
  market/
    sample/           synthetic GBM data (offline dev/demo)
    real/             real OHLCV from yfinance, one CSV per symbol + benchmarks + sector ETFs + macro series
  processed/           {symbol}_features.csv, {symbol}_labels.csv — the model-ready output
  news/                 (intentionally empty — no fabricated sample news, see features/sentiment/)
```

### 3.2 Pipeline stages (in dependency order)

1. **`scripts/fetch_real_data.py`** — yfinance → `data/market/real/{symbol}.csv`, for universe + BENCHMARKS + SECTOR_ETFS (~125 tickers currently: 110 universe + 4 benchmarks + 11 sector ETFs).
2. **`scripts/fetch_macro_data.py`** — yfinance → `data/market/real/{VIX,US10Y,DXY,GOLD,OIL,COPPER,BITCOIN}.csv`.
3. **`scripts/build_phase1_dataset.py`** — reads step 1/2 output, computes price + technical + market-relative features, generates seven-horizon labels → `data/processed/{symbol}_{features,labels}.csv`.
4. **`scripts/augment_features.py`** — post-processes step 3's output *in place*, adding sector/industry/macro columns (needs step 3 to have already run once, since sector/industry features are derived from other symbols' own computed `daily_return`).
5. **Training** (`training/train.py` or `training/tasks.py::retrain_all`) — reads `data/processed/*` only; never touches raw data.
6. **Live serving** (`signals/signal_engine.py::generate_signal`) — reads the *latest row* of `data/processed/{symbol}_features.csv` plus the loaded model files; no recomputation of history per request.

### 3.3 Point-in-time correctness discipline

This is the project's central data-integrity principle, enforced at four points:

- **Fundamentals** (`features/fundamentals/fundamentals.py`): joins are keyed on SEC filing date, not period-end date; `_keep_matching_period()` filters out YTD-vs-quarterly period mismatches within the same filing (a real bug found and fixed — was producing a nonsensical 71% AAPL operating margin before the fix).
- **Earnings** (`features/earnings/earnings.py`): `estimate_next_filing_date()` only ever projects *forward* from the last known filing, capped at 2 cycles, specifically to prevent a stale overdue estimate from producing a negative `days_to_earnings` that would read as "imminent" forever (a regression test locks this in).
- **Sentiment** (`features/sentiment/news_sentiment.py::sentiment_features_as_of`): only articles published on/before `as_of_date` are visible, mirroring the fundamentals pattern — but note live news display (`live_sentiment_snapshot`) is a *separate*, intentionally non-point-in-time path (see §7).
- **Sequence windowing** (`models/sequence_common.py`): trailing-only windows; a window ending at session T never sees T+1.

### 3.4 Known data gaps (see SRS §6 for the requirement-level statement)

- Options data: synthetic proxy (`features/options/options_features.py::SyntheticOptionsSource`), deterministic per symbol (seeded), clearly not real market IV.
- Macro economic releases (CPI/GDP/unemployment): not implemented; would need FRED's `realtime_start`/`realtime_end` vintage fields.
- News: real headlines, but **current-only** (yfinance's `.news` carries no historical archive) — cannot be used for point-in-time training features, only live display.

---

## 4. Feature Engineering Detail

~150 columns per symbol per session at time of writing. Full column groups:

| Group | Module | Example columns | Point-in-time risk |
|---|---|---|---|
| Price | `features/price/price_features.py` | `daily_return`, `log_return`, `gap_return`, `atr_14`, `atr_percent` | None (backward-looking rolling only) |
| Technical | `features/technical/indicators.py` | `sma_*`, `ema_*`, `rsi_*`, `macd*`, `bollinger_*`, `adx_14`, `plus_di_14`, `minus_di_14`, `stochastic_*`, `williams_r`, `cci_20`, `mfi_14`, `obv`, `volume_ratio` | None |
| Pivot levels | `features/technical/pivot_points.py` | `pivot`, `resistance_1..3`, `support_1..3` | None — derived from prior session only; **excluded from model training features** (raw price levels, same reasoning as raw OHLC) |
| Market | `features/market/market_features.py` | `relative_strength_vs_SPY_*`, `{bench}_trend` | None |
| Breadth | `features/market/market_breadth.py` | `market_breadth_score` (used in regime detection, not per-symbol features) | None |
| Sector | `features/sector/sector_features.py` | `sector_return_*`, `stock_vs_sector_*` | None |
| Industry | `features/industry/industry_features.py` | `industry_return`, `stock_vs_industry_1d`, `n_peers` | Structural NaN for <2-peer industries — filled with neutral 0.0 **only at the pooled-training step** (`models/common.py::INDUSTRY_NO_PEER_FILL_COLUMNS`), never fabricated in the feature file itself |
| Macro | `features/macro/macro_features.py` | `vix_level`, `vix_change_1d/1w/1m`, same for us10y/dxy/gold/oil/copper/bitcoin | None (same-day market data, no revision lag) |
| Fundamentals | `features/fundamentals/fundamentals.py` | P/E, margins, YoY growth ratios | Filing-date joined |
| Earnings | `features/earnings/earnings.py` | `days_since_earnings`, `days_to_earnings`, `next_earnings_date`, `days_to_earnings_is_estimate` | Forward-only estimates |
| Options (proxy) | `features/options/options_features.py` | `put_call_ratio`, `iv_rank`, `iv_percentile` | Synthetic — no real-data risk since it isn't real data |
| Sentiment | `features/sentiment/news_sentiment.py` | `news_sentiment_{1,3,7,30}d`, `sentiment_momentum` (training path) | As-of windowed |

### 4.1 Feature exclusion list (never used as model inputs)

`models/common.py::EXCLUDE_COLUMNS`: `date`, raw OHLCV, `{bench}_trend` (categorical), `pivot`, `resistance_1..3`, `support_1..3`. Rationale: raw price levels are dominated by scale, not informative on their own, and don't generalize across tickers of different price — the model instead sees ratio/return-derived features. This was caught as a live bug when pivot columns were added (they would otherwise have degraded cross-sectional generalization silently) and fixed before it shipped.

---

## 5. Labeling Methodology

`labels/horizons.py`, five selectable methods (config: `labels.method`, default `volatility`):

| Method | Logic |
|---|---|
| `fixed` | Forward return vs. a flat threshold (`fixed_threshold`, default 2%) |
| `atr` | Forward return vs. `atr_multiplier × ATR` |
| `volatility` | Forward return vs. `vol_multiplier × realized volatility` (the configured default) |
| `market_relative` | Forward return vs. the benchmark's forward return |
| `risk_adjusted` | Forward return z-scored against its own volatility, threshold `risk_adjusted_z` |

Labels are generated for all seven horizons independently per row; a row near the end of history has no valid label for longer horizons and is dropped at training time (`build_pooled_dataset`), not filled.

---

## 6. Model Architecture Detail

### 6.1 Tree models (XGBoost, LightGBM)

- One classifier per horizon (7 total per architecture), trained on the **pooled cross-sectional dataset** (all symbols' rows for that horizon together) — not one model per stock. This is deliberate: it's what scales to a large universe and lets the model learn general cross-sectional patterns instead of overfitting per-symbol.
- `feature_columns()` auto-selects every numeric column not in `EXCLUDE_COLUMNS` — new feature groups (sector/industry/macro/pivot) flow into training automatically without code changes to the model layer.
- Native `predict_proba` NaN handling; no imputation needed for structurally-missing features (e.g., industry peer columns for isolated industries, filled to 0 upstream as documented in §4).

### 6.2 Sequence models (LSTM, Transformer)

- Shared design (`models/sequence_common.py`): one encoder, seven small linear heads (one per horizon) — a genuine multi-task architecture, not seven independent models.
- Fixed, curated 34-column feature set (`SEQUENCE_FEATURE_COLUMNS`) — deliberately smaller than the tree models' ~140 usable columns, keeping `seq_len × n_features` tractable on CPU.
- `seq_len=60` default (trailing 60 trading sessions).
- Multi-head loss masks out horizons with no valid label for a given window (`MISSING_LABEL = -1`, `CrossEntropyLoss(ignore_index=-1)`) rather than dropping the sample entirely.

### 6.3 Stacking meta-model

- One `LogisticRegression` per horizon, input = the four base models' `[P(BUY), P(HOLD), P(SELL)]` (12 features when all four are present).
- Trained on the base models' **VAL-split predictions** (never used for base-model gradient/tree fitting — early-stopping monitoring only), evaluated on the fully held-out TEST split.
- `predict_proba` maps explicitly through `clf.classes_` rather than assuming a fixed 3-wide output — a real bug (caught by a test) where a horizon's meta-train split lacking one label class (e.g., no SELL rows) would otherwise silently misalign the output columns.
- **Empirical result** (current 110-symbol universe): wins 1D/1W/2W by macro-F1; the weighted average wins 1M/2M/3M/6M. `signal_engine.py` applies this per-horizon, never globally, and falls back to the weighted average whenever a required base model's prediction is missing (a stacker can't score a partially-missing input row the way a weighted average degrades to fewer models automatically).

### 6.4 Model registry (serving-time)

`models/predict.py::ModelRegistry` loads every available trained model once at process start: tree models (dict, keyed by name), sequence models (dict), and the stacker (+ its per-horizon winner set, read from `stacking_report.json`). Missing model files degrade to "not available," never a hard failure.

---

## 7. Ensemble & Signal Scoring — Formulas

```
horizon_score(probs)      = P(BUY) - P(SELL)                              range [-1, 1]

model_weighted_average     = Σ(model_weight_i × probs_i) / Σ(model_weight_i)
                              over models present for that horizon

composite_score            = Σ(horizon_score_h × horizon_weight_h) / Σ(horizon_weight_h)

model_consensus(horizon)   = 1 - stddev(scores across models present)

horizon_consensus          = 1 - stddev(composite-relevant horizon scores)

confidence                 = f(model_consensus, horizon_consensus, probability_strength,
                                 regime.confidence, data_quality)   — signals/confidence.py

signal                     = BUY  if composite_score >= buy_threshold
                              SELL if composite_score <= sell_threshold
                              HOLD otherwise, OR if any conservative-HOLD condition trips:
                                - model_consensus < 0.4
                                - horizon_consensus < 0.4
                                - confidence < 0.35
                                - regime == HIGH_VOLATILITY and regime.confidence < 0.5
                                - insufficient price history
                                - liquidity below floor (20d avg $ volume < config threshold)
                                - earnings within risk_window_days (config, default 5)
```

### 7.1 Market regime detection

`signals/market_regime.py`: majority vote across benchmark ETF trends (Bull/Bear/Sideways), with a `HIGH_VOLATILITY` override, and a breadth-divergence adjustment — if market breadth disagrees with the benchmark-implied trend by more than a fixed threshold, confidence is cut (not the regime flipped): `_breadth_confidence_adjustment()`.

### 7.2 Stock-specific regime

`signals/stock_regime.py`: priority order — HIGH_VOLATILITY override, then ADX<25 → LOW_VOLATILITY/SIDEWAYS, then ADX≥25 → directional vote (price-vs-SMA200, momentum, market trend, sector trend).

### 7.3 Live news sentiment (display path — distinct from the training-time path in §3.3/§4)

`features/sentiment/news_sentiment.py::live_sentiment_snapshot`: fetches current yfinance headlines, scores each with the same lexicon scorer used for point-in-time training features, returns overall sentiment + per-headline breakdown. Explicitly **not** wired into the composite score — display-only, because there's no historical archive to have validated it against.

---

## 8. Backtesting & Validation Methodology

### 8.1 Walk-forward folds

`backtesting/walk_forward.py::generate_folds`: expanding-window, train window always starts at `data_start` and grows; validation/test windows roll forward by `step_years` (defaults: 3yr min train, 1yr val, 1yr test, 1yr step) — e.g. `2018-2021 train | 2021-2022 val | 2022-2023 test`, then `2018-2022 | 2022-2023 | 2023-2024`, etc.

### 8.2 Tree-model walk-forward

`evaluate_fold()`: retrains XGBoost fresh per fold on `[train_start, train_end)`, validates on `[train_end, val_end)`, reports classification metrics on the genuinely out-of-sample `[val_end, test_end)`. Cheap enough (seconds) to run the full 7-horizon × all-folds sweep.

### 8.3 Sequence-model walk-forward

`evaluate_fold_sequence()`: same discipline, genuinely retrains LSTM/Transformer per fold (not re-scoring a static split). Materially more expensive (minutes per fold-horizon-architecture), so `scripts/run_sequence_walk_forward.py` runs a representative subset by default (3 of 7 horizons, 2 most recent folds) — the harness itself supports the full sweep given the compute budget.

### 8.4 Threshold/weight optimization

`backtesting/optimize.py`: grid search over 4 horizon-weight presets × 4 model-weight presets × 5 BUY/SELL thresholds, scored by average Sharpe on a validation window, confirmed on a held-out test window. Current winner: `back_loaded` horizon weights (longer horizons weighted more), `spec_default` model-weight ratio, thresholds ±0.10.

### 8.5 Stacking comparison

`scripts/train_stacking.py`: fits the stacker on VAL-split base-model predictions, evaluates both the stacker and the weighted average on the same TEST-split rows, per horizon — apples-to-apples by construction (same eval set for both).

### 8.6 Trading simulation

`backtesting/engine.py::run_backtest`: long-only, transaction costs applied via `backtesting/costs.py::TransactionCostModel`. Regression-tested for a real off-by-one bug (day-0 entry cost was silently skipped in the original loop).

---

## 9. API Reference

Base: `http://localhost:8000`. All JSON. `symbol` path params are case-insensitive (upper-cased server-side).

| Method | Path | Params | Cache TTL | Notes |
|---|---|---|---|---|
| GET | `/health` | — | — | `{status, models_loaded}` |
| GET | `/api/v1/universe` | `active_only=true` | — | `{count, symbols[]}` |
| GET | `/api/v1/universe/{symbol}` | — | — | Full stock reference record |
| GET | `/api/v1/signals/{symbol}` | — | 300s | Full signal payload — see §9.1 |
| GET | `/api/v1/prices/{symbol}` | `days=180` | — | OHLCV + RSI + Bollinger + ATR + pivot levels, oldest→newest |
| GET | `/api/v1/quotes` | `symbols=A,B,C` | — | `[{symbol, available, price, change, change_pct, volume, date}]` |
| GET | `/api/v1/screener` | `refresh=false` | 600s | Signal summary for every tracked symbol |
| GET | `/api/v1/track-record` | — | — | Backtest/walk-forward/stacking comparison results |
| GET | `/api/v1/monitoring/drift/{symbol}` | `recent_sessions=20` | — | PSI per feature vs. full history |
| GET | `/api/v1/calendar/sessions` | `start`, `end` | — | Trading sessions in range |
| GET | `/api/v1/compliance/sharia` | — | — | Full Sharia-compliance screen report — see §9.2 |

### 9.1 `GET /api/v1/signals/{symbol}` response shape

```json
{
  "symbol": "NVDA", "as_of": "2026-08-12", "signal": "HOLD", "score": 0.091,
  "confidence": 0.70, "model_consensus": 0.87, "horizon_consensus": 0.92,
  "market_regime": "BULL", "stock_regime": "SIDEWAYS",
  "sector": "TECHNOLOGY", "industry": "SEMICONDUCTORS",
  "company_name": "NVIDIA Corporation", "market_cap": 2900000000000,
  "volume": 100991600, "avg_volume_20d": 125780000,
  "risk": "MEDIUM",
  "horizons": {"1D": {"signal": "BUY", "score": 0.114}, "...": "..."},
  "reasons": ["..."], "risks": ["..."],
  "earnings": {"days_since_earnings": null, "days_to_earnings": null,
               "days_to_earnings_is_estimate": null, "next_earnings_date": null},
  "stacking_used": ["1D", "1W", "2W"],
  "levels": {"available": true, "current_price": 217.5, "pivot": 219.49,
             "resistance": {"r1": 222.2, "r2": 226.86, "r3": 229.57},
             "support": {"s1": 214.83, "s2": 212.12, "s3": 207.46},
             "nearest_resistance": 222.2, "near_breakout": false,
             "target": 222.2, "expected_profit_pct": 2.16},
  "news": {"available": true, "overall_sentiment": 0.13,
           "positive_count": 2, "negative_count": 0, "neutral_count": 6,
           "headlines": ["..."]},
  "cached": false
}
```

### 9.2 Sharia compliance screen (`features/compliance/sharia_screen.py`)

An unofficial, mechanical DIY screen — not a certified religious determination (no free certified vendor is accessible to this project; see SRS FR-11 and the mandatory disclaimer text shipped with every surface that shows a result). Two-part screen, both parts must pass:

1. **Business activity** — `business_activity_screen(industry_code)` excludes industries whose core business is inherently non-compliant (`EXCLUDED_INDUSTRIES = {BANKS, INSURANCE, TOBACCO, ALCOHOL, GAMBLING, ADULT_ENTERTAINMENT, PORK_PRODUCTS}`). A hit short-circuits — `screen_symbol` never calls SEC EDGAR for an excluded symbol.
2. **Financial ratios** — `financial_ratio_screen(values, market_cap)` computes, from real SEC XBRL fundamentals (`features/fundamentals/fundamentals.py`, point-in-time via `as_of_fundamentals`):
   - `debt / market_cap < 33%` (long-term + short-term debt)
   - `(cash + short-term investments) / market_cap < 33%`
   - `interest_income / revenue < 5%`

   Any ratio whose inputs are entirely absent from the filing is reported `known: False, passed: None` — never assumed to pass. Only a *known* failing ratio fails the overall screen.

`FINANCE_ADJACENT_LOW_COVERAGE_INDUSTRIES = {ASSET_MANAGEMENT, PAYMENTS}` — a real finding during development: Goldman Sachs, BlackRock, Visa, and Mastercard all came back with every ratio unknown, because their filings use bank-style XBRL tags (deposits, trading liabilities) this project's concept-candidate list doesn't recognize. `compliant: True` there reflects total absence of data, not a real pass, so these get a materially stronger note than an ordinary missing-data case (regression-tested in `tests/test_sharia_screen.py`).

Run as an offline batch job (`scripts/run_sharia_screen.py`), never live per API request — SEC EDGAR is rate-limited and fundamentals only change quarterly. Output: `models/saved/sharia_screen_report.json` (`{as_of, methodology_note, summary, results[]}`), served as-is by `GET /api/v1/compliance/sharia` and joined per-symbol into `GET /api/v1/screener`'s `sharia_compliant`/`sharia_note` fields.

---

## 10. Frontend Architecture

### 10.1 Component tree

```
App.jsx (view state: watchlist | screener | track-record)
├── Sidebar.jsx        — nav tabs, symbol search, watchlist rows (+ live quotes)
└── main
    ├── [watchlist] SignalCard.jsx
    │     ├── SignalBadge.jsx, ScoreGauge.jsx
    │     ├── KeyStats.jsx
    │     ├── ConsensusMeters.jsx
    │     ├── HorizonBreakdown.jsx
    │     ├── PriceChart.jsx        — candlestick/line/Renko/P&F + volume + RSI + BB overlay
    │     ├── SupportResistance.jsx
    │     ├── NewsSentiment.jsx
    │     └── ReasonsRisks.jsx
    ├── [screener] Screener.jsx     — sortable/filterable table, full universe
    └── [track-record] TrackRecord.jsx — backtest/walk-forward/stacking result tables
```

### 10.2 State management

Plain React state + hooks; no external state library. `App.jsx` owns `watchlist` (persisted to `localStorage`), `signalsBySymbol` cache, and `activeSymbol`. No global store — each view fetches what it needs directly via `api.js`.

### 10.3 Chart internals (`PriceChart.jsx` + `lib/chartTransforms.js`)

- Hand-rolled SVG, no charting library dependency (deliberate — keeps the bundle small and avoids a heavy dependency for a fixed, well-understood feature set).
- Three stacked panels sharing one x-scale: price (240px), volume (60px), RSI (80px).
- **Candlestick/Line**: direct bar-indexed rendering; Bollinger Bands rendered as a filled polygon between upper/lower series *plus* separate stroked boundary paths (a fix — the fill alone at low opacity was unreadable as a "band").
- **Renko** (`toRenkoBricks`): ATR-sized bricks (`defaultBoxSize` = latest `atr_14`), classic brick construction — walks the price series, emitting a new brick each time price moves a full brick size in either direction.
- **Point & Figure** (`toPointFigureColumns`): box size = 0.6× ATR, 3-box reversal, classic column construction (extend current column while price keeps moving the same direction; new column on a ≥3-box reversal). Rendered as solid colored column rectangles (a documented simplification — not individual X/O glyphs).
- **Support/resistance overlay**: R1/S1 rendered at full weight (tier 1); R2/S2, R3/S3 fade progressively (tier 2/3) — a fix for three same-color dashed lines blending into one solid stripe when the levels sit close together in price. Labels are de-collision-adjusted (`declutterLabelPositions`) with a minimum 13px gap and a short leader line back to the true price level when they had to move.

### 10.4 Auto-refresh

- Watchlist: 5-minute interval, matches the backend's per-signal cache TTL (polling faster would just re-fetch the same cached response).
- Screener: 2-minute interval, "silent" (no loading-spinner flicker) background poll that doesn't clear a working table on a transient failure.
- Both are user-toggleable (default ON).

---

## 11. Scheduled Jobs (Celery)

### 11.1 Tasks

| Task | Purpose | Triggered |
|---|---|---|
| `training.tasks.refresh_data` | Fetch OHLCV + macro, rebuild features/labels, re-augment, clear `signal:*`/`screener:*` cache | Nightly (chained), or manually via `.delay()` |
| `training.tasks.retrain_all` | Retrain all four architectures on current `data/processed/*` | Nightly (chained), or manually |
| `training.tasks.nightly_pipeline` | Runs `refresh_data` then `retrain_all` **in that order, in-process** | The single `beat_schedule` entry |

### 11.2 Why `nightly_pipeline` exists rather than two independent crons

`retrain_all` trains on whatever CSVs already exist on disk — it never fetches anything itself. Two independent nightly cron entries could race or drift out of order (retrain firing before or without a same-night refresh), silently training on stale data. `nightly_pipeline` calls both functions synchronously in one task, guaranteeing order without depending on timing assumptions between two schedule entries.

### 11.3 A real bug found and fixed: macOS PyTorch + Celery prefork

Celery's default worker pool (`prefork`) forks a child process per task. On macOS, forking a process **after** PyTorch has touched its Metal/MPS backend crashes the child with `SIGABRT` (`+[MPSGraphObject initialize] may have been in progress in another thread when fork() was called`). This was not a theoretical concern — it happened for real: the scheduled `nightly_pipeline` fired on its cron time during development, ran `refresh_data` successfully, then crashed partway through `retrain_all`'s LSTM training step.

**Fix**: `app.conf.worker_pool = "solo"` in `training/celery_app.py` — runs tasks in-process with no forking, eliminating the whole class of bug. The tradeoff (no intra-worker task concurrency) costs nothing here, since `beat` only ever schedules one job at a time. Verified by re-running the exact task that crashed (`nightly_pipeline`, including the LSTM/Transformer retrain step) to completion under the fix.

### 11.4 Schedule

`crontab(hour=18, minute=0)`, timezone `America/New_York` — 6pm ET, giving data vendors time to finalize the day's adjusted close after the 4pm ET market close. Deliberately not more frequent: the underlying data source is daily-bar OHLCV, which doesn't change intraday, so a faster refresh cadence wouldn't produce different results.

---

## 12. Testing Strategy

209 tests across 26 files (`tests/`), ~17 second full run. Coverage philosophy:

- **Business logic is tested directly and thoroughly** (features, labels, scoring, consensus, confidence, regime detection, backtesting engine/metrics, Celery task orchestration via mocks).
- **Thin orchestration layers are verified live, not unit-tested** — the FastAPI endpoints (`api/main.py`) and the Celery task *execution* itself have no dedicated unit-test file; they were verified by actually running them against the live stack (curl, browser, real Celery dispatch) at each step, matching this project's existing convention rather than introducing new test infrastructure (e.g. `httpx`/`TestClient`) for a thin wrapper over already-tested logic.
- **External network calls are mocked in unit tests** (yfinance in sentiment tests, SEC EDGAR client in earnings tests) so the suite runs offline and deterministically; the real integrations are exercised separately, live, when the feature is built (documented in commit-adjacent narrative in `README.md`).
- **Every fixed bug has a regression test** — see §13.

---

## 13. Known Bugs Found and Fixed (chronological, selected)

A running honesty log — this project's development process treats a caught-and-fixed bug as a normal, expected part of building against real data, not something to hide:

| Bug | Root cause | Fix | Test |
|---|---|---|---|
| GTN sandbox Cloudflare block (unrelated .NET project, same session) | Default `urllib` User-Agent blocked | Browser-like User-Agent header | — |
| XBRL period mismatch (71% AAPL operating margin) | 10-Q reports both quarterly and YTD figures with the same filed date; no period-length filter | `_keep_matching_period()` | `tests/test_fundamentals.py` |
| Earnings estimate negative-days bug | Overdue single-cycle estimate could land in the past relative to `as_of_date` | Forward-projection loop, capped at 2 cycles | `tests/test_fundamentals.py` |
| SHAP axis-order bug | Assumed `(n_rows, n_features+1, n_classes)`, actual shape is `(n_rows, n_classes, n_features+1)` | Corrected indexing | — |
| Backtest off-by-one | Day-0 entry transaction cost silently skipped | Explicit day-0 handling | `tests/test_backtesting.py` |
| MLflow `UnboundLocalError` | Local `import mlflow.xgboost` shadowed the module-level `mlflow` name for the whole function | Moved submodule imports to module level | — |
| Docker credential-helper hang | `~/.docker/config.json`'s `credsStore: desktop` hung every pull indefinitely | Removed the key (user-approved) | — |
| TimescaleDB NULL close | Still-forming (incomplete) trading-session rows from yfinance | Detect and skip incomplete rows before load | — |
| **Industry-peer training-data loss** | `dropna()` across *all* feature columns silently dropped every row for any symbol whose industry had <2 peers — 16 of 40 symbols (40%) | Neutral-fill (`0.0`) only the structurally-NaN industry columns at the pooled-training step | `tests/test_models_common.py` |
| **Stacking class-misalignment** | sklearn's `predict_proba` output width shrinks when a meta-train split is missing a label class; assumed a fixed 3-wide output | Map explicitly through `clf.classes_` | `tests/test_stacking.py` |
| **Pivot columns leaking into model features** | New raw-price-level columns (pivot/R1-3/S1-3) would have been auto-included as XGBoost/LightGBM training inputs, degrading cross-sectional generalization the same way raw OHLC would | Added to `EXCLUDE_COLUMNS` before it shipped | — |
| **Prices-endpoint null price** | Today's still-forming session came through with `close: null`, breaking the frontend chart's min/max math | Drop incomplete rows server-side (`dropna`) + defensive client-side filter | — |
| **Celery + PyTorch fork crash (macOS)** | Prefork worker pool forks after PyTorch/MPS initialization; forking after Metal/Objective-C runtime init is unsafe | `worker_pool="solo"` | Verified via live re-run of the crashing task to completion |

---

## 14. Deployment

`docker-compose.yml` services: `postgres` (TimescaleDB), `redis`, `mlflow`, `api`, `worker` (Celery), `beat` (Celery). Local development in this build ran the same processes bare (venv + `uvicorn`, `npm run dev`, `celery worker`/`beat`) rather than through Compose, for faster iteration — the Compose definitions are the intended production packaging.

## 15. Configuration Reference

`config/settings.yaml` — see inline comments in that file for the *why* behind every non-obvious value (horizon weights, model weights, and signal thresholds are all walk-forward-optimized with the search methodology documented directly above each value, not asserted).

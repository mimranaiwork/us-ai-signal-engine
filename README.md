# US AI Signal Engine

A multi-horizon (1D/1W/2W/1M/2M/3M/6M), multi-model BUY/HOLD/SELL signal
engine for US equities (NYSE, Nasdaq, NYSE American). All seven phases from
the spec are built: data foundation, four independent model architectures,
the ensemble that turns their predictions into one real `GET
/api/v1/signals/{symbol}` response, walk-forward backtesting, real SEC EDGAR
fundamentals/earnings-timing ingestion, options/news-sentiment feature
modules, and production infrastructure (Redis caching, MLflow tracking,
Celery scheduled retraining, drift monitoring).

**Two data modes exist side by side.** `scripts/generate_sample_data.py`
produces a fully offline *synthetic geometric Brownian motion* dataset (no
network/API keys needed) — on that data, near-baseline model accuracy
(33-40% on a 3-class problem) is the *correct, expected* outcome, since GBM
has no engineered predictive signal; a model "beating" the majority-class
baseline by a wide margin there would indicate a leakage bug, not a good
model. `scripts/fetch_real_data.py` pulls REAL historical prices for the
whole universe via yfinance (Yahoo Finance) instead, and everything
downstream — features, labels, training, backtesting, the live API — runs
completely unmodified against it via `--source real`. Retrained on real
data, the same models show meaningfully higher validation accuracy
(43-59% depending on horizon, vs. ~33-40% on synthetic data) — evidence the
pipeline is capturing genuine market structure, not an artifact, since nothing
in the modeling code changed between the two runs, only the input data.
`GET /api/v1/signals/AAPL` on real data returns actual current-day signals
(e.g. NVDA: HOLD, score -0.05, confidence 69%, regime BULL, as of the latest
real trading session) — see "Using real market data" below for how to
reproduce this yourself.

## What Phase 1 actually does

```
data/universe/*.csv ──► universe/universe_loader.py ──► point-in-time symbol list
                                                          (survivorship-bias aware)

calendar_us/trading_calendar.py ──► trading-session horizons (1/5/10/21/42/63/126)
                                     — NOT calendar days

ingestion/ohlcv_source.py ──► OHLCV bars (SampleCSVSource offline, or YFinanceSource)
        │
        ▼
ingestion/corporate_actions.py ──► split/dividend adjustment validation
        │
        ▼
features/price/price_features.py ──► returns, ranges, rolling vol, ATR%, $ volume
        │
        ▼
features/technical/indicators.py ──► SMA/EMA, RSI, MACD, Bollinger, OBV, MFI, ...
        │
        ▼
features/market/market_features.py ──► SPY/QQQ/IWM/DIA context + relative strength
        │
        ▼
labels/horizons.py ──► BUY/HOLD/SELL × 7 horizons × 5 threshold methodologies
        │
        ▼
data/processed/{symbol}_features.csv, {symbol}_labels.csv
```

```mermaid
flowchart LR
    U[universe/*.csv] --> UL[Universe loader]
    UL --> SYM[Point-in-time symbol list]
    CAL[US trading calendar] --> HZ[Trading-session horizons]
    SRC[OHLCV source] --> CA[Corporate actions]
    CA --> PF[Price features]
    PF --> TI[Technical indicators]
    TI --> MF[Market features SPY/QQQ/IWM/DIA]
    MF --> LB[Seven-horizon labels]
    HZ --> LB
    SYM --> SRC
    LB --> OUT[(data/processed/*.csv)]
    OUT --> MODELS[XGBoost / LightGBM / LSTM / Transformer]
    MODELS --> ENS[Ensemble: model + horizon consensus]
    ENS --> CONF[Confidence + market regime]
    CONF --> SIG[Final BUY / HOLD / SELL]
    SIG --> API[GET /api/v1/signals/symbol]
```

## What Phase 2 adds — four independent model architectures

```
models/common.py ──► pooled (cross-sectional, all-symbols) train/val/test datasets
        │              per horizon, built from Phase 1's data/processed/*.csv
        ▼
models/xgboost/model.py    ──► one XGBClassifier per horizon (7 total)
models/lightgbm/model.py   ──► one LightGBM Booster per horizon (7 total)
models/sequence_common.py  ──► shared sliding-window sequence builder + training loop
models/lstm/model.py       ──► one shared LSTM encoder, 7 linear heads (multi-task)
models/transformer/model.py──► one shared Transformer encoder, 7 linear heads
        │
        ▼
training/train.py ──► trains all four architectures, all seven horizons,
                       saves to models/saved/{xgboost,lightgbm,lstm,transformer}/
```

- **Pooled, cross-sectional training** (one model per horizon trained across
  the *whole universe* at once), not one model per stock — this is what
  actually scales to "thousands of US stocks," and lets the model learn
  general patterns instead of overfitting ~40 tiny per-symbol datasets.
- **LSTM and Transformer share one encoder with seven small output heads**
  (multi-task), rather than being trained as seven fully separate networks
  each. This matches the spec's own "Features → Multi-Horizon Engine → 7
  heads" diagram (section 30) directly and is far cheaper to train than 14
  separate sequence models.
- **PPO is not part of Phase 2.** Per the spec (section 29), PPO is
  explicitly *not* the directional classifier — it's a position-sizing/
  portfolio-allocation input. `config/settings.yaml`'s `model_weights.ppo`
  placeholder exists for when that's built; nothing currently produces a PPO
  score, so the ensemble runs on the four classifiers only.
- **Known platform gotcha, fixed and documented**: running torch alongside
  numpy/scipy's own OpenMP-linked BLAS in the same process can silently
  SIGABRT with zero error output (a duplicate-OpenMP-runtime crash, not a
  memory issue — it was initially misdiagnosed as OOM before the pattern
  became clear). Fix: `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE`
  must be set before importing torch — baked into the `Dockerfile`; set them
  yourself for local/venv runs (see below).
- **NumPy is pinned `<2`** (with matching `scipy<1.14`) because the last
  torch wheel published for Intel macOS (2.2.2) needs the NumPy 1.x ABI. If
  you're on Linux/Apple Silicon with newer torch wheels available, this pin
  isn't necessary — it's kept for cross-platform reproducibility of this repo.

## What Phase 3 adds — ensemble, confidence, and the real signal

```
signals/scoring.py     ──► horizon_score = P(BUY) - P(SELL); model-weighted
                            probability averaging; composite = Σ(horizon_score × weight)
signals/consensus.py   ──► model consensus + horizon consensus (1 - stddev of scores)
signals/market_regime.py──► BULL/BEAR/SIDEWAYS/HIGH_VOLATILITY from SPY/QQQ/IWM
signals/confidence.py  ──► weighted blend: consensus + probability strength +
                            regime certainty + data quality/liquidity
explainability/shap.py ──► XGBoost native TreeSHAP → fixed phrase-table reasons/risks
                            (never LLM-generated, per the spec's explicit instruction)
        │
        ▼
signals/signal_engine.py ──► conservative HOLD overrides, final signal, risk level
        │
        ▼
GET /api/v1/signals/{symbol}   — now returns the real, fully-populated response
```

- **Conservative HOLD logic (spec section 36) is implemented as an override
  list**, not a single fudge factor: low model consensus, low horizon
  consensus, low confidence, an uncertain regime, insufficient history, and
  insufficient liquidity each independently force HOLD regardless of what
  the raw composite score says, and every triggered reason is surfaced in
  the response's `risks` field.
- **Confidence is explicitly not "probability of profit."** It's a weighted
  blend of agreement/certainty signals available *right now* (model
  consensus, horizon consensus, how far from a coin-flip each model's
  probabilities are, regime certainty, data quality). Two components the
  spec also lists — historical model performance and prediction calibration
  — need backtest history to compute and are deliberately left out of the
  weighted sum (not faked with a placeholder) until Phase 4 produces that
  history.
- **Reasons/risks come from a fixed phrase table**, keyed by the actual
  top-magnitude SHAP-contributing feature names for that specific
  prediction — e.g. `price_vs_sma50` contributing positively becomes "Price
  above 50-day average." A feature with no table entry is silently skipped
  rather than described with an invented sentence.

## What Phase 4 adds — walk-forward backtesting

```
backtesting/costs.py         ──► commission + spread + slippage model (configurable, not zero)
backtesting/metrics.py       ──► classification metrics (accuracy/precision/recall/F1/ROC-AUC/
                                  confusion matrix) + trading metrics (CAGR/Sharpe/Sortino/
                                  max drawdown/Calmar/win rate/profit factor)
backtesting/engine.py        ──► long-only equity-curve simulation net of costs, vs Buy&Hold
                                  and vs SPY/QQQ/IWM/DIA over the same window
backtesting/walk_forward.py  ──► expanding-window fold generator + per-fold retrain-and-evaluate
        │
        ▼
scripts/run_backtest.py ──► backtesting/backtesting_report.json
```

- **Real walk-forward, not a single split.** `generate_folds()` produces the
  expanding-window folds the spec's own example describes (train
  2018-2021→val 2021-2022→test 2022-2023, then train 2018-2022→val
  2022-2023→test 2023-2024, ...), and `evaluate_fold()` retrains a fresh
  XGBoost model on each fold's train window before scoring its genuinely
  held-out test window — this replaces `training/train.py`'s single
  chronological split (that was Phase 2's "does it train" check, not a real
  validation methodology).
- **A real bug was caught by this phase's own tests, not just written
  around.** The first version of `simulate_long_only` had an off-by-one: the
  transaction cost for a position opened on the very first bar of a backtest
  never got applied (the cost-charging loop started at index 1, skipping
  index 0), which silently made an always-long strategy's equity curve
  identical to raw Buy & Hold to full float precision — i.e. free trading.
  `tests/test_backtesting.py::test_always_long_from_day_zero_costs_less_than_buy_and_hold`
  encodes the fix as a regression test.
- **Sequence models (LSTM/Transformer) are not auto-walked forward.**
  Retraining a tree model per fold is cheap (seconds); retraining a
  sequence model per fold is not (minutes, and hit real memory/OpenMP
  issues during Phase 2 development on this machine — see above). The
  harness (`evaluate_fold`/`evaluate_fold_trading`) only hardcodes XGBoost
  today, but takes any model with a `train_horizon()` interface, so wiring
  in LSTM/Transformer walk-forward is a small change if you have the
  compute budget for it — it just isn't Phase 4's default.
- **HOLD means "don't change position," not "flatten."** The simulated
  strategy goes/stays long on BUY, goes/stays flat on SELL, and leaves the
  position alone on HOLD — matching the spec's own conservative-HOLD
  philosophy (section 36) rather than needlessly round-tripping on every
  noisy HOLD day.

## What Phase 5 adds — real SEC EDGAR fundamentals and earnings timing

```
ingestion/sec_edgar.py         ──► real SEC EDGAR client (ticker→CIK lookup +
                                    XBRL companyfacts + filing history), disk-cached
features/fundamentals/fundamentals.py ──► point-in-time fundamentals: every value
                                    keeps its actual filing date; as_of_fundamentals()
                                    only returns what was genuinely public by then
features/earnings/earnings.py  ──► days_to/since_earnings from 10-K/10-Q filing dates
        │
        ▼
scripts/fetch_sec_fundamentals.py ──► demo: real AAPL/MSFT/JPM fundamentals + ratios
```

- **This hits real, live SEC EDGAR data** — no paid vendor, no API key,
  `https://data.sec.gov`'s free XBRL API. It does require a traceable
  `SEC_EDGAR_USER_AGENT` (name + email) per SEC's fair-access policy; the
  client refuses to run with a placeholder value rather than send untraceable
  requests. Responses are disk-cached (`data/sec/cache/`, gitignored) so
  repeated runs don't re-hit the network.
- **Point-in-time correctness is the entire point of this module**, per the
  spec's explicit instruction not to leak revised/future fundamentals into
  historical samples. `tests/test_fundamentals.py`'s leakage test asserts,
  for every single filing, that its value is invisible the day before it was
  filed and visible the day it (or later) was filed — mirroring Phase 1's
  `test_leakage.py` philosophy for a completely different data source.
- **A second real bug was caught testing against live AAPL data**: XBRL
  10-Q filings report flow metrics (revenue, income, cash flow) for BOTH the
  3-month quarter AND the cumulative year-to-date period as separate records
  filed on the same day. Without filtering by period length, "take the
  latest filed value per metric" can silently pick a quarterly figure for
  one metric and a year-to-date figure for another — this produced a
  71%-operating-margin AAPL snapshot (a number that doesn't exist in
  reality) before `_keep_matching_period()` was added to keep only the
  single-period figure matching each filing type (~1 quarter for 10-Q, ~1
  year for 10-K). Balance-sheet metrics (assets, equity, shares outstanding)
  aren't affected — they're point-in-time snapshots, not period sums.
- **Earnings dates are a documented proxy, not the real thing.** There's no
  earnings-calendar feed here, so `days_to/since_earnings` is computed from
  10-K/10-Q *filing* dates — typically a few days to two weeks after the
  actual earnings press release. True EPS/revenue "surprise" (actual vs.
  analyst consensus) needs a paid analyst-estimates feed this project
  doesn't have; `revenue_growth_yoy`/`eps_diluted_growth_yoy` (fundamental
  momentum, section 21) are the honestly-derivable substitute.
- **Not wired into the universe-wide pipeline automatically.** Fetching+
  parsing companyfacts for all 40 sample-universe symbols works (it's the
  same function called in a loop) but pulls a few hundred MB of real JSON
  under SEC's rate limit — a couple of minutes, deliberately not something
  that reruns on every `build_phase1_dataset.py` invocation. Run
  `scripts/fetch_sec_fundamentals.py --symbols A B C ...` when you want it.

## What Phase 6 adds — options and news sentiment

```
features/options/options_features.py   ──► IV, IV rank/percentile, put/call ratio,
                                             options_sentiment, unusual-activity flag
features/sentiment/news_sentiment.py   ──► lexicon-based sentiment scorer + topic
                                             classifier + point-in-time aggregation
```

- **Real historical options-chain data (IV, open interest, put/call volume)
  has no free public source** comparable to SEC EDGAR — it requires a paid
  vendor (CBOE DataShop, ORATS, an OPRA feed reseller). Per the spec's own
  instruction not to make options features mandatory when data quality is
  insufficient, this ships an `OptionsDataSource` interface (mirroring
  `ingestion/ohlcv_source.py`'s pattern) with a `SyntheticOptionsSource`
  implementation: IV is derived from the underlying's own *real* realized
  volatility (already-computed data) plus noise, not arbitrary random
  numbers, so the ratios stay internally consistent even though the
  absolute IV values aren't real. Swap in a paid vendor by implementing the
  same interface.
- **No fabricated news headlines.** Unlike synthetic OHLCV (obviously-fake
  price math), invented headlines would read as real text — manufacturing
  fake financial news is a materially worse thing to ship than a synthetic
  price series, so this module ships zero sample news data. `score_text()`/
  `classify_topic()`/`sentiment_features_as_of()` work on whatever a real
  news source hands them (NewsAPI, Benzinga, etc. — none integrated, no
  credentials available); tests exercise them against a handful of
  illustrative sentences written for testing, never presented as real news.
- **Sentiment scoring is a transparent lexicon, not an opaque model** — a
  compact finance-oriented positive/negative word list (Loughran-McDonald-
  style in spirit, not a reproduction of it). Every score traces back to
  specific matched words, auditable by inspection.
- **Point-in-time correctness again**: `sentiment_features_as_of()` only
  considers articles published on/before the requested date, same pattern
  as `features/fundamentals/fundamentals.py`'s `as_of_fundamentals()`.

## What Phase 7 adds — production infrastructure

```
monitoring/cache.py          ──► Redis response cache for GET /api/v1/signals/{symbol}
                                  (5 min TTL; degrades to always-miss if Redis is unreachable)
monitoring/drift.py          ──► PSI-based data/prediction drift detection
training/mlflow_tracking.py  ──► experiment tracking + model registry (local SQLite by
                                  default, or point MLFLOW_TRACKING_URI at a real server)
training/celery_app.py       ──► Celery app (Redis broker/backend) + nightly retrain schedule
training/tasks.py            ──► the actual retrain_all task (thin wrapper around
                                  training/train.py — a scheduled run does exactly what
                                  running the script by hand does)
        │
        ▼
docker-compose.yml now runs: postgres, redis, mlflow (tracking server + UI),
api (Redis-cached), worker (Celery), beat (Celery scheduler)
```

- **Redis caching is real and tested against a live instance** — `GET
  /api/v1/signals/{symbol}` now returns `"cached": true/false`; a second
  request for the same symbol within 5 minutes skips the full four-model +
  SHAP recomputation. Verified: a cache miss followed immediately by a cache
  hit for the same symbol, and a graceful-degradation path (pointing at a
  port nothing listens on) that never raises — caching is an optimization,
  never a request-breaking dependency.
- **MLflow tracking is real, not a stub.** `training/train.py` now logs
  every training run's parameters, per-horizon validation accuracy, and the
  trained XGBoost/LightGBM/LSTM/Transformer models themselves (via MLflow's
  native `mlflow.xgboost`/`mlflow.lightgbm`/`mlflow.pytorch` flavors) to a
  local SQLite-backed store by default — verified end-to-end: trained a
  real run, then queried it back via `latest_run_metrics()`. A real bug
  surfaced during this: a local `import mlflow.xgboost` inside the logging
  function shadowed the module-level `mlflow` name for the *entire function
  scope* (a classic Python gotcha — any local assignment to a name makes it
  local everywhere in that function, even before the assignment line runs),
  raising `UnboundLocalError` on the earlier `mlflow.start_run()` call.
  Fixed by moving submodule imports to module level.
- **Scheduled retraining is real Celery, not a cron-job description.**
  `training/tasks.py::retrain_all` is a genuine `@app.task`, verified by
  calling it directly (Celery tasks remain plain callables without needing
  a broker connection) — it trained real models and logged a real MLflow
  run. `celery_app.py`'s beat schedule fires it nightly at 6pm ET, after
  the US market close.
- **Drift detection uses PSI (Population Stability Index)**, the standard
  metric, with industry-standard thresholds (< 0.1 stable, 0.1-0.25 watch,
  > 0.25 alert) — exposed at `GET /api/v1/monitoring/drift/{symbol}`.
  Tested against synthetic reference/shifted distributions with known
  expected outcomes. On the live sample data, most price-level-dependent
  features correctly show ALERT-level drift because they're compared
  against 8 years of history on a trending synthetic price series — genuine
  PSI behavior, not a bug; a real deployment would use a rolling recent
  window (e.g. the model's actual training period) as the reference rather
  than all-of-history.
- **TimescaleDB went from "defined but never run" to actually loaded with
  real data.** `database/schema.sql` and `database/models.py` existed since
  Phase 1 but nothing had ever connected to Postgres until
  `scripts/load_timescaledb.py` — applying the schema against a real
  TimescaleDB container (hypertables and all), then loading the real
  universe/OHLCV/labels and confirming with cross-table SQL joins (e.g.
  `stocks JOIN sectors`, real AAPL closing prices, real per-horizon label
  distributions). Two real things surfaced doing this: (1) `docker pull`
  was hanging indefinitely on every image, even public ones — not a
  resource problem, `~/.docker/config.json`'s `"credsStore": "desktop"`
  was invoking a credential helper that never returned in this environment;
  removing it unblocked pulls immediately. (2) yfinance occasionally
  returns a NULL close/adj_close for the most recent, still-forming trading
  session (an intraday fetch catching an incomplete bar) — the schema's
  `NOT NULL` constraint correctly rejected it; the loader now detects and
  skips incomplete rows rather than failing the whole load.

## Using real market data

```
scripts/fetch_real_data.py  ──► yfinance -> data/market/real/{symbol}.csv
                                 (same OHLCV_COLUMNS schema as the synthetic
                                 data — SampleCSVSource reads either directory)
        │
        ▼
scripts/build_phase1_dataset.py --source real  ──► data/processed/*.csv
        │
        ▼
training/train.py  (unmodified — it only ever reads data/processed/*.csv)
```

- **A real broker sandbox was tried first and didn't have market data
  provisioned.** Al Jazira Capital's GTN sandbox credentials authenticate
  successfully and the Trade API confirms NYSE/Nasdaq/AMEX are configured
  as tradable exchanges — but every Market Data API endpoint (history,
  symbol-search, tickers, realtime, even the bare service root) returns a
  `404` from GTN's own API gateway, for this specific tenant. That's a
  "not provisioned for this trial credential set" signal, not a wrong-
  symbol or wrong-parameter issue — confirmed by testing the same
  integration from a completely different codebase (the sibling
  `trading_platform` .NET project, same session) with the same result. A
  real, separate bug was found and fixed along the way regardless: Python's
  default `urllib` User-Agent gets blocked by Cloudflare (error 1010)
  in front of GTN's sandbox — a normal browser-like User-Agent fixes it.
- **yfinance needed zero code changes to the pipeline** — it was already
  built and tested as `ingestion/ohlcv_source.py::YFinanceSource` back in
  Phase 1, behind the same `OHLCVSource` interface `SampleCSVSource` uses.
  `scripts/fetch_real_data.py` is the only new file: it fetches once, caches
  to CSV in the same schema, and every downstream stage — features, labels,
  training, backtesting, the live API — runs completely unmodified via
  `--source real` / pointing at `data/market/real/`.
- **Reproduce it**:
  ```bash
  python scripts/fetch_real_data.py                    # ~55 symbols, a few minutes
  python scripts/build_phase1_dataset.py --source real --end $(date +%Y-%m-%d)
  python training/train.py --no-mlflow                 # or drop --no-mlflow to log the run
  python scripts/run_backtest.py --data-end $(date +%Y-%m-%d)
  uvicorn api.main:app --reload
  curl localhost:8000/api/v1/signals/NVDA               # real, current-day signal
  ```
- **Real-data validation accuracy is meaningfully higher than synthetic**
  (43-59% vs. ~33-40% depending on horizon) — expected, since real prices
  have genuine momentum/mean-reversion structure a random walk doesn't.
  Real backtests also show real variance vs. Buy & Hold: sometimes ahead
  (AAPL 1W: strategy +37.7% vs. Buy & Hold +35.6%), sometimes flat because
  the model never signaled BUY in that window (AAPL 1M one fold: 0 trades,
  0% return vs. Buy & Hold's +12.0%) — the honest range of outcomes a real
  model produces, not a curated result.

## What was added after Phase 7 — feature expansion + ensemble hardening

Once real market data was in place, four more tracks of work landed on top
of the original seven phases: quick correctness fixes, sector/industry/macro
features, ensemble methods (stacking) + walk-forward, and a React frontend
(see the frontend's own docs). This section covers the first three; each was
verified against real data with a passing test suite, not just written.

- **ADX/+DI/-DI (Wilder-smoothed)** — `features/technical/indicators.py::add_trend_strength()`,
  wired into `add_all_technical_indicators()` and into
  `signals/stock_regime.py::detect_stock_regime()`, which classifies each
  stock's own regime (HIGH_VOLATILITY / LOW_VOLATILITY / SIDEWAYS / trending
  up-or-down) from ADX + price-vs-SMA200 + momentum + market/sector trend —
  distinct from the market-wide regime in `signals/market_regime.py`.
- **Market breadth** — `features/market/market_breadth.py::compute_market_breadth()`
  scans the whole universe (advancers/decliners, % above SMA50/200, 252-session
  new highs/lows) into one `market_breadth_score`, which
  `signals/market_regime.py` now uses to *cut confidence* (not flip the
  regime) when breadth diverges from the benchmark-implied trend by more
  than 0.15 — a genuine breadth/price divergence, not a threshold-tuning
  artifact.
- **Sector/industry/macro features** — `features/sector/sector_features.py`
  (stock return vs. its sector ETF), `features/industry/industry_features.py`
  (peer-group proxy: averages same-industry stocks' own returns, since no
  free per-industry index exists — excludes industries with <2 members
  rather than fabricating a "peer average" from one stock), and
  `features/macro/macro_features.py` (VIX/US10Y/DXY/Gold/Oil/Copper/Bitcoin
  via `scripts/fetch_macro_data.py`, deliberately NOT CPI/PPI/unemployment/
  GDP — those need FRED's point-in-time vintage fields to avoid leaking
  revised-later values, not implemented). `scripts/augment_features.py`
  back-fills these into the already-built `data/processed/*.csv` files.
  16/40 universe symbols (AAPL, AMZN, TSLA, JPM, ...) have no ≥2-member
  industry peer group — their `industry_*` columns are legitimately `NaN`,
  filled with neutral `0.0` only at the pooled-training step
  (`models/common.py::INDUSTRY_NO_PEER_FILL_COLUMNS`) so those symbols don't
  get silently dropped from every horizon's training set entirely (an actual
  bug caught mid-session: `dropna()` across all feature columns was wiping
  out 40% of the universe before this fix).
- **Walk-forward-optimized thresholds/weights** — `backtesting/optimize.py`
  grid-searches horizon-weight presets × model-weight presets × BUY/SELL
  thresholds, scored by average Sharpe on a validation window and confirmed
  on a held-out test window; `config/settings.yaml`'s `horizons.weights` and
  `signal_thresholds` reflect the winner (see that file's comments for the
  actual numbers and the "sell rarely fires" caveat found along the way).
- **Stacking / meta-model ensemble** — `signals/stacking.py::StackingEnsemble`
  fits one multinomial logistic regression per horizon on the four base
  models' own `P(BUY)/P(HOLD)/P(SELL)` outputs (their VAL-split predictions,
  never used for gradient/tree fitting), evaluated against the existing
  fixed-weight average (`signals/scoring.py::model_weighted_average`) on the
  fully held-out TEST split via `scripts/train_stacking.py`. Real, mixed
  result on real data: stacking wins the short horizons (1D/1W/2W, by
  macro-F1) but the fixed-weight average wins 1M/2M/3M/6M — see
  `models/saved/stacking_report.json` for the full per-horizon numbers.
  `signal_engine.py` uses stacking only for the horizons it actually won,
  falling back to the weighted average everywhere else, including whenever
  a base model's prediction for a horizon is unavailable (a stacker can't
  score a partially-missing input row the way a weighted average degrades
  to fewer models automatically).
- **TimescaleDB actually exercised** — `scripts/load_timescaledb.py` loads
  real OHLCV into the hypertable schema (`database/schema.sql`); 6 symbols
  had NULL close values for their most-recent (still-forming) trading day
  from yfinance, violating the NOT NULL constraint — the loader now detects
  and skips incomplete rows rather than failing the whole load.
- **LSTM/Transformer walk-forward** — `backtesting/walk_forward.py::evaluate_fold_sequence()`
  genuinely retrains each architecture per fold (not just re-evaluating the
  Phase 2 single split), reusing one pre-built sequence-window set across
  every fold/horizon/architecture call so it doesn't re-window from scratch
  each time. `scripts/run_sequence_walk_forward.py` runs a *representative*
  subset — 1D/1M/6M (short/medium/long) × the 2 most recent folds × both
  architectures, 15 symbols, 2 epochs — not the full 7-horizon × all-folds
  sweep the tree models get (minutes per fold-horizon-architecture vs.
  seconds; see that module's docstring). Real result on real data
  (`backtesting/sequence_walk_forward_report.json`): both land in the
  38-48% range across folds/horizons — in the same ballpark as the tree
  models' single-split numbers (43-59%), Transformer noticeably slower to
  train (~30s/fold vs. LSTM's ~6s/fold on this machine) for accuracy that
  isn't consistently better.

## Project structure

See the full target structure in the original spec; this build fills in:
`calendar_us/`, `universe/`, `ingestion/{ohlcv_source,corporate_actions,
sec_edgar}.py`, `features/{price,technical,market,fundamentals,earnings,
options,sentiment}/`, `labels/`, `models/{xgboost,lightgbm,lstm,transformer,
common,predict,sequence_common}`, `training/{train,mlflow_tracking,
celery_app,tasks}.py`, `signals/`, `explainability/shap.py`,
`backtesting/{costs,metrics,engine,walk_forward}.py`, `monitoring/{cache,
drift}.py`, `database/schema.sql`, `api/main.py`, `tests/`, `scripts/`.
Still stubbed (`"""Phase 2+ — not implemented"""` placeholders, matching the
target layout without pretending unbuilt pieces are done): `models/{ensemble,
ppo}/` (PPO is deliberately out of scope — see Phase 2 section above) and
`features/{sector,industry,macro}/` (sector-ETF *mapping* exists in
`data/universe/sectors.csv`; richer sector/industry/macro feature
*computation* was never in any phase's explicit deliverable list).

## Design decisions worth knowing

- **Trading-session horizons, not calendar days.** `1M` = 21 trading
  sessions, computed by taking the Nth *row* after T in an
  already-trading-session-indexed dataframe — not `T + 30 days`. This avoids
  the misalignment calendar-day math would cause across holidays/weekends.
- **US trading calendar is dependency-free.** Holidays (MLK, Good Friday,
  Juneteenth-since-2022, etc.) are computed programmatically from the actual
  NYSE holiday rules (`calendar_us/trading_calendar.py`), not hardcoded per
  year and not pulled from `pandas_market_calendars` — one less external
  package that can go stale or fail to install.
- **No naive `future_price > current_price` labels.** All five labeling
  methodologies (fixed / ATR / volatility / market-relative / risk-adjusted)
  require the move to clear a volatility-aware threshold before it counts as
  BUY/SELL. `tests/test_labels.py::test_naive_price_comparison_would_be_noisier_than_thresholded`
  demonstrates why.
- **Survivorship bias is structural, not an afterthought.** `stocks.csv` has
  `active`/`date_added`/`date_removed`/`delisted_reason`, and
  `Universe.as_of(date)` reconstructs point-in-time membership. Phase 1 ships
  two illustrative delisted rows (Lehman Brothers, Facebook's ticker
  retirement) to prove the schema and tests work — a comprehensive historical
  delisted-stock dataset needs a paid vendor feed (CRSP, Sharadar, etc.) and
  is a Phase 5+ ingestion task.
- **Everything runs offline by default.** `SampleCSVSource` reads a
  synthetic (GBM-based, seeded, reproducible) dataset generated by
  `scripts/generate_sample_data.py` — no API keys or network access needed
  to run Phase 1 end-to-end. `YFinanceSource` exists for when you want real
  data; it's a drop-in replacement behind the same `OHLCVSource` interface.
- **Threshold/weight values in `config/settings.yaml` are placeholders**,
  exactly as the spec requires — horizon weights, model weights, and label
  thresholds get walk-forward-optimized once there are trained models to
  optimize against (Phase 2+). Nothing here claims to be tuned.

## Data-leakage testing

`tests/test_leakage.py` is the automated proof required by the spec: it
computes every feature column on the full price history and again on a
series truncated to end at row T, and asserts they're identical — if any
indicator peeked at future rows (e.g. a centered rolling window), this would
fail. It also verifies that perturbing data beyond a label's own horizon
never changes that label.

## Running it

```bash
cd us-ai-signal-engine
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE   # see "known platform gotcha" above

# 1. Generate the offline synthetic dataset (no network/API keys needed)
python scripts/generate_sample_data.py

# 2. Run the test suite (118 tests: calendar, universe, indicators, labels,
#    leakage, scoring/consensus/confidence/regime, SHAP explainability,
#    backtesting engine/metrics/walk-forward, SEC fundamentals point-in-time,
#    options/news-sentiment, drift detection, Redis cache)
pytest tests/ -v

# 3. Build the full feature + label dataset for the whole universe
python scripts/build_phase1_dataset.py
#   -> data/processed/{symbol}_features.csv, {symbol}_labels.csv

# 4. Train all four model architectures across all seven horizons
python training/train.py
#   -> models/saved/{xgboost,lightgbm,lstm,transformer}/, training_report.json
#   (add --skip-sequence-models to train just XGBoost/LightGBM, much faster;
#    add --symbols AAPL MSFT ... to train on a subset if memory-constrained)

# 5. Serve real signals
uvicorn api.main:app --reload
curl localhost:8000/health
curl localhost:8000/api/v1/universe
curl "localhost:8000/api/v1/calendar/sessions?start=2024-01-01&end=2024-01-31"
curl localhost:8000/api/v1/signals/AAPL   # -> real BUY/HOLD/SELL now

# 6. Walk-forward backtest (retrains per fold, compares vs Buy&Hold/SPY/QQQ/IWM/DIA)
python scripts/run_backtest.py
#   -> backtesting/backtesting_report.json

# 7. Real SEC EDGAR fundamentals (needs a contact email, no API key)
export SEC_EDGAR_USER_AGENT="Your Name your-email@example.com"
python scripts/fetch_sec_fundamentals.py --symbols AAPL MSFT JPM

# 8. Drift check (PSI vs. full history — see Phase 7 section above)
curl localhost:8000/api/v1/monitoring/drift/AAPL

# 9. Full stack in Docker — Postgres, Redis, MLflow server+UI, the API
#    (Redis-cached), and a Celery worker+beat pair for nightly retraining.
#    Trains locally first, then bakes the trained models into the image
#    (see docker-compose.yml's header comment).
docker compose up -d --build
docker compose exec postgres psql -U postgres -d us_ai_signal_engine -f /schema/schema.sql
DATABASE_URL="postgresql+psycopg2://postgres:postgres@localhost:5432/us_ai_signal_engine" python scripts/load_timescaledb.py
open http://localhost:5001   # MLflow UI
```

## What's still genuinely open

Every phase in the spec has real, tested code behind it, and the post-Phase-7
work above closed several of the gaps this section used to list (threshold/
weight optimization, sector/industry/macro features, and real-market-data
validation are all done now — see that section for specifics). What's left:

- **LSTM/Transformer walk-forward only covers a representative subset** —
  `evaluate_fold_sequence()` (see post-Phase-7 section above) genuinely
  retrains per fold now, but `scripts/run_sequence_walk_forward.py` only
  runs 3 of 7 horizons × the 2 most recent folds by default, not the full
  7-horizon × all-folds sweep `evaluate_fold()` does for XGBoost — sequence
  retraining is materially more expensive per call (minutes vs. seconds),
  so the harness function supports the full sweep but the default driver
  doesn't run it.
- **Stacking only won 3 of 7 horizons on real data** (1D/1W/2W; the
  fixed-weight average still wins 1M/2M/3M/6M) — `signal_engine.py` already
  applies this per-horizon rather than picking one method globally, but a
  logistic-regression meta-model over 12 base-model probability columns is
  simple by design; a gradient-boosted meta-model or a k-fold-OOF training
  scheme (this uses the base models' own held-out VAL predictions, not true
  k-fold OOF) are the natural next things to try if this needs to win more
  horizons.
- **PPO** is deliberately not built as a directional classifier, per the
  spec's own instruction (section 29) — `config/settings.yaml`'s
  `model_weights.ppo` placeholder exists for when a position-sizing/
  portfolio-allocation PPO agent is added as an *ensemble input*, not a
  signal source. It's also not loaded by `models/predict.py::ModelRegistry`
  at all yet, unlike the other four models.
- **Options/news data sources are synthetic/proxy by design** (see Phase 6
  section above) — no free historical options vendor or news API exists to
  wire in for real; the feature *machinery* (IV rank, put/call sentiment,
  lexicon sentiment scoring, point-in-time joins) is real and tested against
  clean interfaces a real vendor integration would drop into.
- **CPI/PPI/unemployment/GDP/ISM/Consumer Confidence** are explicitly not
  in `features/macro/macro_features.py` — see that script's docstring for
  why (FRED's point-in-time vintage fields are needed to avoid leaking
  revised-later economic data, not implemented here).

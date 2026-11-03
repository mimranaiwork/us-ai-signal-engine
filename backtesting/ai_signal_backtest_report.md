# AI signal backtest (walk-forward, XGBoost)

295 Shariah (SPUS) stocks, 2020-01-02 → 2026-11-02. Universe: **point-in-time SPUS holdings (SEC N-PORT, quarterly)**. The model is retrained every January on purged past data only, scores combine the 7 horizons with the repo's weights, and the portfolio holds the **top 10 stocks scoring ≥ +0.10** (the BUY threshold), rebalanced weekly, traded one day after the signal, 3 bps per side. Nothing was tuned on these results.

Stocks are picked, ranked and trained on only for dates they were in the Shariah index. XGBoost only (no LightGBM/LSTM/Transformer); no conservative-HOLD overrides; no sector/macro/fundamental features. Not financial advice.

Price data found for **94.8%** of index members across 26 quarterly snapshots (lowest 90%). Members with no price data (delisted or renamed), the bias that remains: ABMD, ALXN, ANSS, ATVI, AVB, CERN, CTLT, CTRA, CTXS, CXO, DAY, DRE, EQR, FDXF, HES, HOLX, JNPR, K, KLG, KSU, KVUE, MXIM, PKI, PXD, Q, QRVO, TIF, TWTR, VAR, WBA.

## 2020–2023 (2020-01-01 → 2023-12-31)

| Portfolio | CAGR | Max drawdown | Sharpe | Avg invested |
|---|---|---|---|---|
| AI top 10 (idle cash) | 8.3% | -43.3% | 0.41 | 100% |
| AI top 10 (idle cash in SPUS) | 8.2% | -43.3% | 0.41 | 100% |
| AI top 10, costs ×7 (20 bps per side) | 1.8% | -43.8% | 0.21 | 100% |
| Simple rule: top 10 by 6-month gain | 5.1% | -34.3% | 0.33 | 88% |
| Simple rule: top 10 most volatile | 35.2% | -38.8% | 0.98 | 94% |
| Equal-weight all Shariah stocks | 14.0% | -34.1% | 0.67 | 100% |
| Buy & hold SPUS | 14.9% | -30.8% | 0.72 | 100% |

Signal quality: average rank IC (score vs next-21-session return) **+0.007** (positive on 49% of days); top-minus-bottom decile 21-session return **+0.55%**.

## 2024–latest (2024-01-01 → 2026-11-02)

| Portfolio | CAGR | Max drawdown | Sharpe | Avg invested |
|---|---|---|---|---|
| AI top 10 (idle cash) | 26.7% | -24.8% | 1.08 | 100% |
| AI top 10 (idle cash in SPUS) | 26.7% | -24.8% | 1.08 | 100% |
| AI top 10, costs ×7 (20 bps per side) | 17.8% | -25.7% | 0.78 | 100% |
| Simple rule: top 10 by 6-month gain | 39.7% | -35.7% | 1.07 | 100% |
| Simple rule: top 10 most volatile | 32.8% | -36.6% | 0.88 | 100% |
| Equal-weight all Shariah stocks | 14.8% | -20.1% | 0.96 | 100% |
| Buy & hold SPUS | 21.6% | -22.8% | 1.17 | 100% |

Signal quality: average rank IC (score vs next-21-session return) **+0.020** (positive on 56% of days); top-minus-bottom decile 21-session return **+0.66%**.

## Whole period (2020-01-01 → 2026-11-02)

| Portfolio | CAGR | Max drawdown | Sharpe | Avg invested |
|---|---|---|---|---|
| AI top 10 (idle cash) | 15.4% | -43.3% | 0.64 | 100% |
| AI top 10 (idle cash in SPUS) | 15.3% | -43.3% | 0.64 | 100% |
| AI top 10, costs ×7 (20 bps per side) | 8.0% | -43.8% | 0.41 | 100% |
| Simple rule: top 10 by 6-month gain | 17.9% | -36.3% | 0.70 | 93% |
| Simple rule: top 10 most volatile | 34.4% | -38.8% | 0.94 | 97% |
| Equal-weight all Shariah stocks | 14.2% | -34.1% | 0.74 | 100% |
| Buy & hold SPUS | 17.4% | -30.8% | 0.87 | 100% |

Signal quality: average rank IC (score vs next-21-session return) **+0.012** (positive on 52% of days); top-minus-bottom decile 21-session return **+0.60%**.

Weeks with at least one stock above the BUY threshold: 100%. Average holdings when invested: 10.0 stocks.

Purging check: every model's latest training label date is before the year it trades (see `fit_log` in the JSON).

## What the model actually held

Average 60-day daily volatility of its picks: **2.62%** vs **2.10%** for the universe (1.2× as volatile). Share of its picks that were also in the 6-month-gain top 10: **4%**.

Most often held (% of weeks): CTVA 69%, NVDA 30%, ENPH 23%, GNRC 23%, ROL 22%, AOS 20%, ON 20%, SMCI 18%, TSLA 16%, TTD 16%


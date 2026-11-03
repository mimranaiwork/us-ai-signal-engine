# AI signal backtest (walk-forward, XGBoost)

212 Shariah (SPUS) stocks, 2020-01-02 → 2026-11-02. The model is retrained every January on purged past data only, scores combine the 7 horizons with the repo's weights, and the portfolio holds the **top 10 stocks scoring ≥ +0.10** (the BUY threshold), rebalanced weekly, traded one day after the signal, 3 bps per side. Nothing was tuned on these results.

**Read with care:** today's Shariah list applied to the past (survivorship bias); XGBoost only (no LightGBM/LSTM/Transformer); no conservative-HOLD overrides; no sector/macro/fundamental features. Not financial advice.

## 2020–2023 (2020-01-01 → 2023-12-31)

| Portfolio | CAGR | Max drawdown | Sharpe | Avg invested |
|---|---|---|---|---|
| AI top 10 (idle cash) | 31.7% | -45.7% | 0.91 | 100% |
| AI top 10 (idle cash in SPUS) | 31.6% | -45.7% | 0.91 | 100% |
| Equal-weight all Shariah stocks | 21.2% | -35.8% | 0.90 | 100% |
| Buy & hold SPUS | 14.9% | -30.8% | 0.72 | 100% |

Signal quality: average rank IC (score vs next-21-session return) **+0.016** (positive on 50% of days); top-minus-bottom decile 21-session return **+1.04%**.

## 2024–latest (2024-01-01 → 2026-11-02)

| Portfolio | CAGR | Max drawdown | Sharpe | Avg invested |
|---|---|---|---|---|
| AI top 10 (idle cash) | 45.4% | -41.3% | 1.40 | 100% |
| AI top 10 (idle cash in SPUS) | 45.4% | -41.3% | 1.40 | 100% |
| Equal-weight all Shariah stocks | 22.4% | -21.4% | 1.30 | 100% |
| Buy & hold SPUS | 21.6% | -22.8% | 1.17 | 100% |

Signal quality: average rank IC (score vs next-21-session return) **+0.049** (positive on 64% of days); top-minus-bottom decile 21-session return **+2.92%**.

## Whole period (2020-01-01 → 2026-11-02)

| Portfolio | CAGR | Max drawdown | Sharpe | Avg invested |
|---|---|---|---|---|
| AI top 10 (idle cash) | 36.9% | -45.7% | 1.07 | 100% |
| AI top 10 (idle cash in SPUS) | 36.8% | -45.7% | 1.07 | 100% |
| Equal-weight all Shariah stocks | 21.5% | -35.8% | 1.00 | 100% |
| Buy & hold SPUS | 17.4% | -30.8% | 0.87 | 100% |

Signal quality: average rank IC (score vs next-21-session return) **+0.029** (positive on 55% of days); top-minus-bottom decile 21-session return **+1.65%**.

Weeks with at least one stock above the BUY threshold: 100%. Average holdings when invested: 10.0 stocks.

Purging check: every model's latest training label date is before the year it trades (see `fit_log` in the JSON).


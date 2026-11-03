# AI signal backtest (walk-forward, XGBoost)

212 Shariah (SPUS) stocks, 2020-01-02 → 2026-11-02. The model is retrained every January on purged past data only, scores combine the 7 horizons with the repo's weights, and the portfolio holds the **top 10 stocks scoring ≥ +0.10** (the BUY threshold), rebalanced weekly, traded one day after the signal, 3 bps per side. Nothing was tuned on these results.

**Read with care:** today's Shariah list applied to the past (survivorship bias); XGBoost only (no LightGBM/LSTM/Transformer); no conservative-HOLD overrides; no sector/macro/fundamental features. Not financial advice.

## 2020–2023 (2020-01-01 → 2023-12-31)

| Portfolio | CAGR | Max drawdown | Sharpe | Avg invested |
|---|---|---|---|---|
| AI top 10 (idle cash) | 29.3% | -46.0% | 0.86 | 100% |
| AI top 10 (idle cash in SPUS) | 29.2% | -46.0% | 0.86 | 100% |
| AI top 10, costs ×7 (20 bps per side) | 21.8% | -46.3% | 0.71 | 100% |
| Simple rule: top 10 by 6-month gain | 28.0% | -20.4% | 1.07 | 88% |
| Simple rule: top 10 most volatile | 43.6% | -45.7% | 1.06 | 94% |
| Equal-weight all Shariah stocks | 21.2% | -35.8% | 0.90 | 100% |
| Buy & hold SPUS | 14.9% | -30.8% | 0.72 | 100% |

Signal quality: average rank IC (score vs next-21-session return) **+0.016** (positive on 49% of days); top-minus-bottom decile 21-session return **+1.03%**.

## 2024–latest (2024-01-01 → 2026-11-02)

| Portfolio | CAGR | Max drawdown | Sharpe | Avg invested |
|---|---|---|---|---|
| AI top 10 (idle cash) | 51.0% | -37.6% | 1.49 | 100% |
| AI top 10 (idle cash in SPUS) | 51.0% | -37.6% | 1.49 | 100% |
| AI top 10, costs ×7 (20 bps per side) | 42.1% | -38.1% | 1.30 | 100% |
| Simple rule: top 10 by 6-month gain | 75.9% | -38.4% | 1.47 | 100% |
| Simple rule: top 10 most volatile | 82.7% | -43.5% | 1.48 | 100% |
| Equal-weight all Shariah stocks | 22.4% | -21.4% | 1.30 | 100% |
| Buy & hold SPUS | 21.6% | -22.8% | 1.17 | 100% |

Signal quality: average rank IC (score vs next-21-session return) **+0.049** (positive on 64% of days); top-minus-bottom decile 21-session return **+2.77%**.

## Whole period (2020-01-01 → 2026-11-02)

| Portfolio | CAGR | Max drawdown | Sharpe | Avg invested |
|---|---|---|---|---|
| AI top 10 (idle cash) | 37.5% | -46.0% | 1.08 | 100% |
| AI top 10 (idle cash in SPUS) | 37.5% | -46.0% | 1.08 | 100% |
| AI top 10, costs ×7 (20 bps per side) | 29.5% | -46.3% | 0.91 | 100% |
| Simple rule: top 10 by 6-month gain | 45.5% | -38.4% | 1.23 | 93% |
| Simple rule: top 10 most volatile | 58.4% | -45.7% | 1.24 | 97% |
| Equal-weight all Shariah stocks | 21.5% | -35.8% | 1.00 | 100% |
| Buy & hold SPUS | 17.4% | -30.8% | 0.87 | 100% |

Signal quality: average rank IC (score vs next-21-session return) **+0.030** (positive on 55% of days); top-minus-bottom decile 21-session return **+1.63%**.

Weeks with at least one stock above the BUY threshold: 100%. Average holdings when invested: 10.0 stocks.

Purging check: every model's latest training label date is before the year it trades (see `fit_log` in the JSON).

## What the model actually held

Average 60-day daily volatility of its picks: **3.14%** vs **2.22%** for the universe (1.4× as volatile). Share of its picks that were also in the 6-month-gain top 10: **6%**.

Most often held (% of weeks): BE 65%, VRT 42%, COHR 39%, AOS 37%, P 35%, FLEX 33%, HAL 30%, MRVL 28%, LITE 27%, TWLO 26%


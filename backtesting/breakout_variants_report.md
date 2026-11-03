# Breakout strategy: variant study

Seven variants fixed in advance. Each was run on the **selection period (2020-01-01 → 2023-12-31)**; the best Sharpe ratio was chosen; that choice was then checked on the **test period (2024-01-01 → 2026-11-02)**, which played no part in choosing it. Same simulator, costs and caveats as `breakout_backtest_report.md` (survivorship bias flatters every variant equally; earnings guard and AI filter not simulated). Not financial advice.

## Variants

- **A. Baseline (live bot today)**: Current settings: sell half at +2R, trail 3×ATR, exit after 15 days under +0.5R, 1% risk, ≤20% per stock, 5 slots.
- **B. No time stop**: As A, but no 15-day time exit.
- **C. Let winners run**: No partial sale, wider 4×ATR trailing stop, no time exit — full position rides the trend.
- **D. Bigger positions**: As A, but 1.5% risk per trade, ≤33% per stock, 4 slots.
- **E. Let winners run + bigger positions**: C and D combined.
- **F. Baseline + idle cash in SPUS**: As A, but uninvested cash is held in SPUS instead of sitting idle.
- **G. E + idle cash in SPUS**: E with uninvested cash held in SPUS.

## Selection period (2020-01-01 → 2023-12-31)

| Variant | CAGR | Max drawdown | Sharpe | Trades | Win rate | Avg per trade |
|---|---|---|---|---|---|---|
| A. Baseline (live bot today) | 3.1% | -12.6% | 0.36 | 153 | 41.2% | +0.12R |
| B. No time stop | 3.1% | -13.8% | 0.34 | 123 | 36.6% | +0.17R |
| C. Let winners run | 1.3% | -22.0% | 0.16 | 115 | 33.0% | +0.11R |
| D. Bigger positions | 2.5% | -17.9% | 0.26 | 129 | 42.6% | +0.09R |
| E. Let winners run + bigger positions | 3.2% | -22.5% | 0.29 | 91 | 36.3% | +0.21R |
| F. Baseline + idle cash in SPUS | 7.5% | -30.8% | 0.43 | 153 | 41.2% | +0.12R |
| G. E + idle cash in SPUS | 6.6% | -30.8% | 0.39 | 91 | 36.3% | +0.21R |
| Buy & hold SPUS | 14.9% | -30.8% | 0.72 | — | — | — |

**Chosen on Sharpe: F. Baseline + idle cash in SPUS**

## Test period (2024-01-01 → 2026-11-02): unseen data

| Variant | CAGR | Max drawdown | Sharpe | Trades | Win rate | Avg per trade |
|---|---|---|---|---|---|---|
| A. Baseline (live bot today) | 7.7% | -10.0% | 0.76 | 130 | 48.5% | +0.19R |
| B. No time stop | 8.7% | -12.1% | 0.80 | 85 | 41.2% | +0.27R |
| C. Let winners run | 7.3% | -18.7% | 0.55 | 82 | 41.5% | +0.34R |
| D. Bigger positions | 3.2% | -16.9% | 0.32 | 110 | 46.4% | +0.10R |
| E. Let winners run + bigger positions | 0.8% | -26.8% | 0.13 | 71 | 38.0% | +0.13R |
| **F. Baseline + idle cash in SPUS ← chosen** | 17.9% | -24.2% | 0.99 | 130 | 48.5% | +0.19R |
| G. E + idle cash in SPUS | 9.1% | -19.6% | 0.53 | 71 | 38.0% | +0.13R |
| Buy & hold SPUS | 21.6% | -22.8% | 1.17 | — | — | — |

## Whole period (2020-01-01 → 2026-11-02)

| Variant | CAGR | Max drawdown | Sharpe | Trades | Win rate | Avg per trade |
|---|---|---|---|---|---|---|
| A. Baseline (live bot today) | 4.3% | -12.6% | 0.48 | 282 | 44.7% | +0.14R |
| F. Baseline + idle cash in SPUS | 11.0% | -30.8% | 0.60 | 282 | 44.7% | +0.14R |
| Buy & hold SPUS | 17.4% | -30.8% | 0.87 | — | — | — |


# Breakout strategy: variant study

Seven variants fixed in advance. Each was run on the **selection period (2020-01-01 → 2023-12-31)**; the best Sharpe ratio was chosen; that choice was then checked on the **test period (2024-01-01 → 2026-11-02)**, which played no part in choosing it. Same simulator, costs and caveats as `breakout_backtest_report.md`. Universe: **point-in-time SPUS holdings (SEC N-PORT, quarterly)**. Earnings guard and AI filter not simulated. Not financial advice.

Price data found for **94.8%** of index members across 26 quarterly snapshots (lowest 90%). Members with no price data (delisted or renamed), the bias that remains: ABMD, ALXN, ANSS, ATVI, AVB, CERN, CTLT, CTRA, CTXS, CXO, DAY, DRE, EQR, FDXF, HES, HOLX, JNPR, K, KLG, KSU, KVUE, MXIM, PKI, PXD, QRVO, TIF, TWTR, VAR, WBA.

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
| A. Baseline (live bot today) | 0.6% | -14.8% | 0.11 | 149 | 39.6% | +0.06R |
| B. No time stop | 1.0% | -19.4% | 0.15 | 105 | 35.2% | +0.08R |
| C. Let winners run | -0.5% | -22.1% | 0.01 | 99 | 32.3% | +0.06R |
| D. Bigger positions | -2.0% | -19.9% | -0.12 | 127 | 38.6% | +0.03R |
| E. Let winners run + bigger positions | -1.0% | -26.0% | -0.00 | 80 | 31.2% | +0.06R |
| F. Baseline + idle cash in SPUS | 7.5% | -30.8% | 0.44 | 149 | 39.6% | +0.06R |
| G. E + idle cash in SPUS | 1.9% | -30.8% | 0.20 | 80 | 31.2% | +0.06R |
| Buy & hold SPUS | 14.9% | -30.8% | 0.72 | — | — | — |

**Chosen on Sharpe: F. Baseline + idle cash in SPUS**

## Test period (2024-01-01 → 2026-11-02): unseen data

| Variant | CAGR | Max drawdown | Sharpe | Trades | Win rate | Avg per trade |
|---|---|---|---|---|---|---|
| A. Baseline (live bot today) | 5.9% | -9.6% | 0.62 | 126 | 47.6% | +0.20R |
| B. No time stop | 5.7% | -10.7% | 0.59 | 87 | 42.5% | +0.29R |
| C. Let winners run | 8.4% | -17.8% | 0.67 | 78 | 38.5% | +0.50R |
| D. Bigger positions | 5.9% | -11.6% | 0.55 | 101 | 49.5% | +0.25R |
| E. Let winners run + bigger positions | 1.2% | -27.3% | 0.15 | 64 | 35.9% | +0.32R |
| **F. Baseline + idle cash in SPUS ← chosen** | 15.4% | -22.1% | 0.89 | 126 | 47.6% | +0.20R |
| G. E + idle cash in SPUS | 10.2% | -18.9% | 0.58 | 64 | 35.9% | +0.32R |
| Buy & hold SPUS | 21.6% | -22.8% | 1.17 | — | — | — |

## Whole period (2020-01-01 → 2026-11-02)

| Variant | CAGR | Max drawdown | Sharpe | Trades | Win rate | Avg per trade |
|---|---|---|---|---|---|---|
| A. Baseline (live bot today) | 2.2% | -14.8% | 0.28 | 276 | 43.1% | +0.11R |
| F. Baseline + idle cash in SPUS | 10.2% | -30.8% | 0.57 | 276 | 43.1% | +0.11R |
| Buy & hold SPUS | 17.4% | -30.8% | 0.87 | — | — | — |


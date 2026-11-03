# Breakout strategy backtest

Period **2020-06-01 → 2026-11-02**, $100,000 starting capital, 296 of 325 Shariah (SPUS) stocks with data. Universe: **point-in-time SPUS holdings (SEC N-PORT, quarterly)**. Same rule functions as the live bot (`execution/breakout.py`), signals at the close, fills at the next open, stops checked against each day's low, 3 bps cost per side.

Price data found for **94.8%** of index members across 26 quarterly snapshots (lowest 90%). Members with no price data (delisted or renamed), the bias that remains: ABMD, ALXN, ANSS, ATVI, AVB, CERN, CTLT, CTRA, CTXS, CXO, DAY, DRE, EQR, FDXF, HES, HOLX, JNPR, K, KLG, KSU, KVUE, MXIM, PKI, PXD, QRVO, TIF, TWTR, VAR, WBA.

A stock can only be bought on dates it was actually in the Shariah index. Earnings guard, AI filter and daily-loss brake are not simulated. Past results don't predict future returns. Not financial advice.

### With SPUS market filter (live bot setting)

| | Strategy | Buy & hold SPUS |
|---|---|---|
| Total return | 16.3% | 200.0% |
| CAGR | 2.4% | 18.7% |
| Max drawdown | -14.8% | -28.2% |
| Sharpe | 0.29 | 1.00 |
| Annual volatility | 9.9% | 19.0% |

Trades: **276** · win rate **43.1%** · avg win **+1.36R** · avg loss **-0.84R** · expectancy **+0.11R per trade** · avg hold 18 sessions
Time with ≥1 position: 85.0% · avg positions held 3.0 · market filter on 72.8% of days

Exit reasons: stop hit 155, time stop 94, stop (gap down) 15, failed breakout 7, open at end of test 5

| Year | Strategy | SPUS |
|---|---|---|
| 2020 | -7.3% | 24.3% |
| 2021 | 9.3% | 37.6% |
| 2022 | -0.3% | -23.1% |
| 2023 | 2.0% | 35.5% |
| 2024 | 6.9% | 27.9% |
| 2025 | 4.1% | 19.9% |
| 2026 | 4.1% | 13.0% |

### Without market filter

| | Strategy | Buy & hold SPUS |
|---|---|---|
| Total return | 42.9% | 200.0% |
| CAGR | 5.7% | 18.7% |
| Max drawdown | -22.0% | -28.2% |
| Sharpe | 0.57 | 1.00 |
| Annual volatility | 10.8% | 19.0% |

Trades: **319** · win rate **43.9%** · avg win **+1.52R** · avg loss **-0.85R** · expectancy **+0.19R per trade** · avg hold 18 sessions
Time with ≥1 position: 92.1% · avg positions held 3.6 · market filter on 100.0% of days

Exit reasons: stop hit 184, time stop 98, stop (gap down) 21, failed breakout 11, open at end of test 5

| Year | Strategy | SPUS |
|---|---|---|
| 2020 | 4.7% | 24.3% |
| 2021 | 31.2% | 37.6% |
| 2022 | -2.0% | -23.1% |
| 2023 | -7.3% | 35.5% |
| 2024 | 12.5% | 27.9% |
| 2025 | 4.9% | 19.9% |
| 2026 | -0.8% | 13.0% |

No price data for: ABMD, ALXN, ANSS, ATVI, AVB, CERN, CTLT, CTRA, CTXS, CXO, DAY, DRE, EQR, FDXF, HES, HOLX, JNPR, K, KLG, KSU, KVUE, MXIM, PKI, PXD, QRVO, TIF, TWTR, VAR, WBA


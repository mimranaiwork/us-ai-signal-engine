# Breakout strategy backtest

Period **2019-12-18 → 2026-11-02**, $100,000 starting capital, 213 of 214 Shariah (SPUS) stocks with data. Same rule functions as the live bot (`execution/breakout.py`), signals at the close, fills at the next open, stops checked against each day's low, 3 bps cost per side.

**Read with care:** the universe is *today's* Shariah list applied to the past (survivorship bias: stocks that dropped out, often after falling, are missing), which flatters absolute returns. Earnings guard, AI filter and daily-loss brake are not simulated. Past results don't predict future returns. Not financial advice.

### With SPUS market filter (live bot setting)

| | Strategy | Buy & hold SPUS |
|---|---|---|
| Total return | 32.4% | 205.8% |
| CAGR | 4.2% | 17.7% |
| Max drawdown | -12.6% | -30.8% |
| Sharpe | 0.46 | 0.88 |
| Annual volatility | 10.0% | 21.0% |

Trades: **283** · win rate **45.2%** · avg win **+1.33R** · avg loss **-0.86R** · expectancy **+0.13R per trade** · avg hold 18 sessions
Time with ≥1 position: 76.7% · avg positions held 2.9 · market filter on 69.9% of days

Exit reasons: stop hit 144, time stop 96, stop (gap down) 26, failed breakout 12, open at end of test 5

| Year | Strategy | SPUS |
|---|---|---|
| 2019 | 0.0% | 0.8% |
| 2020 | 1.6% | 24.0% |
| 2021 | 13.4% | 37.6% |
| 2022 | -5.8% | -23.1% |
| 2023 | 2.3% | 35.5% |
| 2024 | 16.6% | 27.9% |
| 2025 | -3.9% | 19.9% |
| 2026 | 6.3% | 13.0% |

### Without market filter

| | Strategy | Buy & hold SPUS |
|---|---|---|
| Total return | 24.4% | 205.8% |
| CAGR | 3.2% | 17.7% |
| Max drawdown | -17.3% | -30.8% |
| Sharpe | 0.34 | 0.88 |
| Annual volatility | 11.1% | 21.0% |

Trades: **346** · win rate **42.8%** · avg win **+1.36R** · avg loss **-0.86R** · expectancy **+0.09R per trade** · avg hold 17 sessions
Time with ≥1 position: 89.7% · avg positions held 3.5 · market filter on 100.0% of days

Exit reasons: stop hit 171, time stop 118, stop (gap down) 36, failed breakout 16, open at end of test 5

| Year | Strategy | SPUS |
|---|---|---|
| 2019 | -0.5% | 0.8% |
| 2020 | -0.1% | 24.0% |
| 2021 | 21.2% | 37.6% |
| 2022 | -0.5% | -23.1% |
| 2023 | 2.0% | 35.5% |
| 2024 | 3.3% | 27.9% |
| 2025 | -0.8% | 19.9% |
| 2026 | -0.3% | 13.0% |

No price data for: FDXF


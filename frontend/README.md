# US AI Signal Engine — frontend

A TradingView-styled signal card UI (spec section 49) consuming the backend's
`GET /api/v1/signals/{symbol}` — real BUY/HOLD/SELL signals, not mocked data.
Plain React + Vite, no UI framework dependency beyond React itself.

## What it shows

- **Signal badge** — BUY/HOLD/SELL, colored like TradingView candles (teal/red),
  with a -1..+1 composite-score gauge next to it.
- **Tags** — market regime, this stock's own regime (`signals/stock_regime.py`),
  sector, industry, risk level, and a `cached` tag when the backend served a
  Redis-cached response instead of recomputing.
- **Earnings banner** — only rendered when `earnings.days_to_earnings` is set;
  says whether it's a real filing-calendar date or an estimate
  (`days_to_earnings_is_estimate`, see `features/earnings/earnings.py`).
- **Confidence / model consensus / horizon consensus meters.**
- **Per-horizon breakdown** — all seven horizons (1D through 6M), each with
  its own signal/score, tagged `stacked` when `signal_engine.py` used the
  stacking meta-model for that horizon instead of the fixed-weight average
  (see the main README's stacking section for which horizons that is on
  real data).
- **Reasons / risk factors** — the SHAP-derived `reasons`/`risks` lists from
  the API, or an honest "none available" message rather than an empty gap
  when a symbol has no XGBoost 1M model loaded.

## Running it

```bash
# 1. Backend must be running first (from the project root)
cd ..
uvicorn api.main:app --port 8000

# 2. Frontend
cd frontend
npm install
npm run dev              # http://localhost:5173
```

`VITE_API_BASE` (see `.env.example`) points at the backend; defaults to
`http://localhost:8000`, matching the uvicorn command above.

## Notes

- The watchlist is a plain array in `localStorage` (`signal-engine-watchlist`) —
  no backend persistence, no accounts. Seeded with `NVDA/AAPL/MSFT/TSLA/JPM`
  on first load.
- Symbol search only autocompletes against `GET /api/v1/universe` (the
  configured stock universe) — searching a real ticker outside that universe
  correctly surfaces the backend's actual 404 message rather than pretending
  it doesn't exist as a distinct "not found" state.
- No routing library — a single view, symbol switch is client-state only, no
  URL deep-linking to a specific symbol yet.

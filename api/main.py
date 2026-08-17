"""
Phase 1 API skeleton. Only what Phase 1 can honestly serve:

  GET /health                     liveness check
  GET /api/v1/universe            the configured stock universe
  GET /api/v1/universe/{symbol}   one stock's reference data
  GET /api/v1/calendar/sessions   trading sessions in a date range

GET /api/v1/signals/{symbol} — the final BUY/HOLD/SELL endpoint from the
spec — is intentionally a 501 stub here: it needs trained models (Phase 2),
an ensemble (Phase 3), and calibrated confidence (Phase 3), none of which
exist yet. Returning a fabricated signal would be worse than admitting it
isn't built.
"""

from __future__ import annotations

import os
import sys
from datetime import date

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from calendar_us.trading_calendar import trading_sessions  # noqa: E402
from models.common import PROCESSED_DIR, feature_columns  # noqa: E402
from models.predict import ModelRegistry  # noqa: E402
from monitoring.cache import cache_get, cache_set  # noqa: E402
from monitoring.drift import detect_feature_drift, summarize_drift  # noqa: E402
from signals.signal_engine import generate_signal  # noqa: E402
from universe.universe_loader import Universe  # noqa: E402

app = FastAPI(title="US AI Signal Engine", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_universe: Universe | None = None
_registry: ModelRegistry | None = None


def get_universe() -> Universe:
    global _universe
    if _universe is None:
        _universe = Universe()
    return _universe


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


@app.get("/health")
def health():
    try:
        models_loaded = get_registry().available_models()
    except Exception:
        models_loaded = []
    return {"status": "ok", "models_loaded": models_loaded}


@app.get("/api/v1/universe")
def list_universe(active_only: bool = True):
    symbols = get_universe().all_symbols(active_only=active_only)
    return {"count": len(symbols), "symbols": symbols}


@app.get("/api/v1/universe/{symbol}")
def get_stock(symbol: str):
    try:
        stock = get_universe().get(symbol.upper())
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown symbol '{symbol}'")
    return {
        "symbol": stock.symbol,
        "company_name": stock.company_name,
        "exchange": stock.exchange,
        "sector": stock.sector_code,
        "industry": stock.industry_code,
        "market_cap": stock.market_cap,
        "country": stock.country,
        "active": stock.active,
        "date_added": stock.date_added.isoformat() if stock.date_added else None,
        "date_removed": stock.date_removed.isoformat() if stock.date_removed else None,
        "delisted_reason": stock.delisted_reason,
    }


@app.get("/api/v1/quotes")
def get_quotes(symbols: str):
    """Lightweight last-price/change/%change for the watchlist sidebar —
    just the last two rows of each symbol's processed features CSV, no ML
    inference. Deliberately NOT /api/v1/signals/{symbol}: that runs all four
    models + SHAP per call, wasteful to do once per watchlist row just for a
    price ticker. `symbols`: comma-separated, e.g. `?symbols=AAPL,MSFT`."""
    import math

    import pandas as pd

    requested = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    quotes = []
    for symbol in requested:
        path = os.path.join(PROCESSED_DIR, f"{symbol}_features.csv")
        if not os.path.exists(path):
            quotes.append({"symbol": symbol, "available": False})
            continue

        df = pd.read_csv(path, usecols=["date", "adj_close", "volume"], parse_dates=["date"])
        df = df.dropna(subset=["adj_close"]).tail(2)
        if df.empty:
            quotes.append({"symbol": symbol, "available": False})
            continue

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) == 2 else None
        price = float(last["adj_close"])
        change = float(price - prev["adj_close"]) if prev is not None else None
        change_pct = round(change / float(prev["adj_close"]) * 100, 2) if change and prev is not None and prev["adj_close"] else None
        volume = last.get("volume")

        quotes.append({
            "symbol": symbol,
            "available": True,
            "price": round(price, 2),
            "change": round(change, 2) if change is not None else None,
            "change_pct": change_pct,
            "volume": None if volume is None or math.isnan(volume) else float(volume),
            "date": last["date"].strftime("%Y-%m-%d"),
        })

    return {"count": len(quotes), "quotes": quotes}


@app.get("/api/v1/calendar/sessions")
def get_sessions(start: date, end: date):
    sessions = trading_sessions(start, end)
    return {"count": len(sessions), "sessions": [d.isoformat() for d in sessions]}


PIVOT_COLUMNS = ["pivot", "resistance_1", "resistance_2", "resistance_3", "support_1", "support_2", "support_3"]
INDICATOR_COLUMNS = ["rsi_14", "bollinger_upper", "bollinger_mid", "bollinger_lower", "volume_sma_20", "atr_14"]


@app.get("/api/v1/prices/{symbol}")
def get_prices(symbol: str, days: int = 180):
    """Recent OHLCV + pivot levels (features/technical/pivot_points.py) +
    RSI/Bollinger Bands for the frontend's price chart — reads straight off
    the already-computed data/processed/{symbol}_features.csv (same file
    generate_signal() reads the latest row of), no recomputation."""
    import math

    import pandas as pd

    symbol = symbol.upper()
    try:
        get_universe().get(symbol)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown symbol '{symbol}'")

    path = os.path.join(PROCESSED_DIR, f"{symbol}_features.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No processed price history for '{symbol}'")

    cols = ["date", "open", "high", "low", "close", "adj_close", "volume"] + PIVOT_COLUMNS + INDICATOR_COLUMNS
    df = pd.read_csv(path, usecols=lambda c: c in cols, parse_dates=["date"])
    # Today's still-forming session can come through with a NULL close/adj_close
    # (same yfinance quirk scripts/load_timescaledb.py already works around) —
    # drop it rather than serving a null price the chart can't plot.
    df = df.dropna(subset=["close", "adj_close"]).tail(max(1, days))

    def _num(v):
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 4)

    bars = [
        {
            "date": row["date"].strftime("%Y-%m-%d"),
            **{
                c: _num(row.get(c))
                for c in ["open", "high", "low", "close", "adj_close", "volume"] + PIVOT_COLUMNS + INDICATOR_COLUMNS
            },
        }
        for _, row in df.iterrows()
    ]
    return {"symbol": symbol, "count": len(bars), "bars": bars}


@app.get("/api/v1/signals/{symbol}")
def get_signal(symbol: str):
    symbol = symbol.upper()
    try:
        get_universe().get(symbol)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown symbol '{symbol}'")

    cache_key = f"signal:{symbol}"
    cached = cache_get(cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    registry = get_registry()
    if not registry.available_models():
        raise HTTPException(
            status_code=503,
            detail="No trained models found — run training/train.py first.",
        )

    try:
        result = generate_signal(symbol, registry)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    cache_set(cache_key, result)
    return {**result, "cached": False}


SCREENER_TTL_SECONDS = 600  # more expensive than a single signal (runs the whole universe) — cache a bit longer
SHARIA_REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "saved", "sharia_screen_report.json")


def _load_sharia_lookup() -> dict:
    """{symbol: {compliant, fully_determined, note}} from the last
    scripts/run_sharia_screen.py run — not computed live (needs one SEC
    EDGAR call per symbol; fundamentals change quarterly at most, see that
    script's docstring). Missing report file -> empty dict, never a 500."""
    if not os.path.exists(SHARIA_REPORT_PATH):
        return {}
    import json

    with open(SHARIA_REPORT_PATH) as f:
        report = json.load(f)
    return {
        r["symbol"]: {"compliant": r["compliant"], "fully_determined": r["fully_determined"], "note": r["note"]}
        for r in report.get("results", [])
    }


@app.get("/api/v1/compliance/sharia")
def get_sharia_compliance():
    """The full DIY Sharia-compliance screen report — see
    features/compliance/sharia_screen.py's module docstring for methodology
    and the "not a certified determination" caveat."""
    if not os.path.exists(SHARIA_REPORT_PATH):
        raise HTTPException(status_code=404, detail="No compliance report found — run scripts/run_sharia_screen.py first.")
    import json

    with open(SHARIA_REPORT_PATH) as f:
        return json.load(f)


@app.get("/api/v1/screener")
def get_screener(refresh: bool = False):
    """Runs generate_signal() across the whole tracked universe and returns
    each symbol's signal + breakout levels (features/technical/pivot_points.py)
    in one response — what the frontend's Screener view sorts/filters
    client-side. Whole-universe run, so cached longer than a single signal
    (SCREENER_TTL_SECONDS vs. monitoring/cache.py's DEFAULT_TTL_SECONDS);
    `?refresh=true` bypasses the cache for an on-demand recompute."""
    cache_key = "screener:all"
    if not refresh:
        cached = cache_get(cache_key)
        if cached is not None:
            return {**cached, "cached": True}

    registry = get_registry()
    if not registry.available_models():
        raise HTTPException(
            status_code=503,
            detail="No trained models found — run training/train.py first.",
        )

    universe = get_universe()
    sharia_lookup = _load_sharia_lookup()
    results = []
    for symbol in universe.all_symbols(active_only=True):
        try:
            r = generate_signal(symbol, registry)
        except Exception:
            continue  # one bad symbol shouldn't fail the whole screener
        stock = universe.get(symbol)
        levels = r.get("levels", {})
        sharia = sharia_lookup.get(symbol)
        results.append({
            "symbol": symbol,
            "company": stock.company_name,
            "sector": stock.sector_code,
            "industry": stock.industry_code,
            "signal": r["signal"],
            "score": r["score"],
            "confidence": r["confidence"],
            "risk": r["risk"],
            "price": levels.get("current_price"),
            "target": levels.get("target"),
            "expected_gain_pct": levels.get("expected_profit_pct"),
            "near_breakout": levels.get("near_breakout"),
            "sharia_compliant": sharia["compliant"] if sharia else None,
            "sharia_note": sharia["note"] if sharia else "No compliance screen run for this symbol yet.",
        })

    payload = {"count": len(results), "results": results}
    cache_set(cache_key, payload, ttl_seconds=SCREENER_TTL_SECONDS)
    return {**payload, "cached": False}


@app.get("/api/v1/monitoring/drift/{symbol}")
def get_drift(symbol: str, recent_sessions: int = 20):
    """Data drift (spec Phase 7): compares the most recent `recent_sessions`
    trading sessions' feature distributions against the full historical
    distribution for the same symbol. Flags features whose PSI indicates a
    meaningful shift — see monitoring/drift.py for the methodology."""
    import pandas as pd

    symbol = symbol.upper()
    path = os.path.join(PROCESSED_DIR, f"{symbol}_features.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No processed features for '{symbol}'")

    df = pd.read_csv(path)
    if len(df) < recent_sessions + 60:
        raise HTTPException(status_code=422, detail="Insufficient history to compute drift")

    reference_df = df.iloc[: -recent_sessions]
    current_df = df.iloc[-recent_sessions:]
    cols = feature_columns(df)

    results = detect_feature_drift(reference_df, current_df, cols)
    summary = summarize_drift(results)
    summary["details"] = [{"feature": r.feature, "psi": round(r.psi, 4), "status": r.status} for r in results]
    return summary


BACKTESTING_DIR = os.path.join(os.path.dirname(__file__), "..", "backtesting")
MODELS_SAVED_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")


def _load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    import json

    with open(path) as f:
        return json.load(f)


@app.get("/api/v1/track-record")
def get_track_record():
    """Surfaces the walk-forward validation results that already exist as
    report files on disk (scripts/run_backtest.py, scripts/train_stacking.py,
    scripts/run_sequence_walk_forward.py) — real, historical, out-of-sample
    numbers, not live signals. Any report file that hasn't been generated
    yet is simply omitted (`available: false`), never faked."""
    backtest = _load_json(os.path.join(BACKTESTING_DIR, "backtesting_report.json"))
    stacking = _load_json(os.path.join(MODELS_SAVED_DIR, "stacking_report.json"))
    sequence = _load_json(os.path.join(BACKTESTING_DIR, "sequence_walk_forward_report.json"))

    classification_by_horizon: dict[str, list[dict]] = {}
    if backtest:
        for row in backtest["classification"]:
            if "error" in row:
                continue
            classification_by_horizon.setdefault(row["horizon"], []).append({
                "test_start": row["fold"]["val_end"],
                "test_end": row["fold"]["test_end"],
                "accuracy": row["test_classification"]["accuracy"],
                "f1_macro": row["test_classification"]["f1_macro"],
                "n_test": row["n_test"],
            })

    trading = []
    if backtest:
        for row in backtest["trading"]:
            trading.append({
                "symbol": row["symbol"],
                "horizon": row["horizon"],
                "strategy_sharpe": row["strategy"].get("sharpe"),
                "strategy_return": row["strategy"].get("total_return"),
                "buy_and_hold_return": row["buy_and_hold"].get("total_return"),
                "max_drawdown": row["strategy"].get("max_drawdown"),
                "win_rate": row["strategy"].get("win_rate"),
                "n_trades": row.get("n_trades"),
            })

    return {
        "tree_model_walk_forward": {
            "available": backtest is not None,
            "by_horizon": classification_by_horizon,
        },
        "trading_simulation": {
            "available": backtest is not None,
            "results": trading,
        },
        "stacking": {
            "available": stacking is not None,
            "summary": stacking.get("summary") if stacking else None,
            "by_horizon": stacking.get("horizons") if stacking else None,
        },
        "sequence_walk_forward": {
            "available": sequence is not None,
            "note": sequence.get("note") if sequence else None,
            "results": sequence.get("results") if sequence else None,
        },
    }

"""
Alpaca adapter (alpaca-py). Paper trading unless ALPACA_LIVE=yes.

Kept deliberately thin: everything that decides *what* to do lives in
execution/breakout.py (pure, tested without a broker); this file only
reads bars/positions and places market + GTC stop orders.

Bars come from Alpaca's free IEX feed: prices are fine, volume is IEX's
share of consolidated volume only — fine for volume-vs-own-average ratios,
not for absolute dollar-volume filters (hence none here; the SPUS universe
is all S&P 500 large caps anyway).
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

NY = ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)


class AlpacaBroker:
    def __init__(self):
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.trading.client import TradingClient

        key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY.")
        self.live = os.environ.get("ALPACA_LIVE", "").lower() == "yes"
        self.trading = TradingClient(key, secret, paper=not self.live)
        self.data = StockHistoricalDataClient(key, secret)

    def market_open(self) -> bool:
        return bool(self.trading.get_clock().is_open)

    def bars(self, symbols: list[str], days: int = 420) -> dict[str, pd.DataFrame]:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        start = datetime.now(timezone.utc) - timedelta(days=days)
        today = datetime.now(NY).date()
        out: dict[str, pd.DataFrame] = {}
        for i in range(0, len(symbols), 100):
            chunk = symbols[i:i + 100]
            try:
                df = self.data.get_stock_bars(StockBarsRequest(
                    symbol_or_symbols=chunk, timeframe=TimeFrame.Day, start=start, feed=DataFeed.IEX)).df
            except Exception as exc:
                logger.warning("bars failed for chunk %d: %s", i, exc)
                continue
            if not isinstance(df.index, pd.MultiIndex):
                continue
            present = set(df.index.get_level_values(0))
            for sym in chunk:
                if sym in present:
                    d = df.xs(sym, level=0)[["open", "high", "low", "close", "volume"]].sort_index()
                    out[sym] = d[d.index.tz_convert(NY).date < today]  # completed sessions only
        return out

    def account(self) -> dict:
        a = self.trading.get_account()
        return {"equity": float(a.equity), "cash": float(a.cash), "last_equity": float(a.last_equity)}

    def positions(self) -> dict[str, tuple[int, float]]:
        return {p.symbol: (int(float(p.qty)), float(p.avg_entry_price)) for p in self.trading.get_all_positions()}

    def _open_orders(self, symbol: str):
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        return self.trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol]))

    def cancel(self, symbol: str) -> None:
        for o in self._open_orders(symbol):
            self.trading.cancel_order_by_id(o.id)
        time.sleep(1.5)

    def market(self, symbol: str, side: str, qty: int) -> float:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        o = self.trading.submit_order(MarketOrderRequest(
            symbol=symbol, qty=qty, time_in_force=TimeInForce.DAY,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL))
        for _ in range(30):
            o = self.trading.get_order_by_id(o.id)
            st = str(o.status).lower()
            if st.endswith("filled") and "partially" not in st:
                return float(o.filled_avg_price)
            if any(s in st for s in ("rejected", "canceled", "expired")):
                raise RuntimeError(f"order {o.status}")
            time.sleep(2)
        raise RuntimeError("order not filled within 60s")

    def set_stop(self, symbol: str, qty: int, price: float) -> bool:
        """Ensures exactly one GTC stop for the full position. Returns True if changed."""
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import StopOrderRequest
        for o in self._open_orders(symbol):
            if ("stop" in str(o.order_type).lower() and int(float(o.qty)) == qty
                    and abs(float(o.stop_price) - price) < 0.01):
                return False
        self.cancel(symbol)
        if qty > 0:
            self.trading.submit_order(StopOrderRequest(
                symbol=symbol, qty=qty, side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC, stop_price=round(price, 2)))
        return True

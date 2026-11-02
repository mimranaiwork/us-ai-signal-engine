"""
Optional confirmation from this project's own ensemble.

If SIGNAL_API_URL points at a running instance of api/main.py (with
trained models), each breakout candidate is checked against
GET /api/v1/signals/{symbol} before buying:

  mode "off"          — ignore the engine entirely (pure technical breakout)
  mode "veto_sell"    — default: skip a breakout the engine currently calls SELL
  mode "require_buy"  — only buy breakouts the engine also calls BUY

Why veto_sell is the default rather than require_buy: per
config/settings.yaml's own walk-forward notes, the ensemble's final signal
is HOLD most of the time (conservative overrides) and SELL rarely fires,
so require_buy would block almost every trade. Treat the engine as a
second opinion, not the trigger.

If the API is unreachable or the symbol is outside the engine's universe
(404), "veto_sell" lets the trade through with a note; "require_buy"
blocks it (strict means strict).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

MODES = ("off", "veto_sell", "require_buy")


def fetch_signal(symbol: str, base_url: str | None, timeout: int = 30, getter=None) -> dict | None:
    if not base_url:
        return None
    url = f"{base_url.rstrip('/')}/api/v1/signals/{symbol}"
    try:
        if getter is not None:
            return getter(url)
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, TimeoutError, OSError):
        return None


def decide(signal: dict | None, mode: str) -> tuple[bool, str]:
    if mode not in MODES:
        raise ValueError(f"ai_filter mode must be one of {MODES}, got {mode!r}")
    if mode == "off":
        return True, "AI filter off"
    if signal is None:
        if mode == "require_buy":
            return False, "AI signal unavailable (require_buy blocks)"
        return True, "AI signal unavailable"
    s = signal.get("signal")
    conf = signal.get("confidence")
    tag = f"AI {s}" + (f" ({conf:.0%} conf)" if isinstance(conf, (int, float)) else "")
    if mode == "veto_sell" and s == "SELL":
        return False, f"{tag}: vetoed"
    if mode == "require_buy" and s != "BUY":
        return False, f"{tag}: not BUY"
    return True, tag

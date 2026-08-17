"""
Redis-backed response cache for the API (spec Phase 7: Redis). Generating a
signal means loading a feature row, running four models across seven
horizons, computing SHAP contributions, and rebuilding benchmark market-
regime features — worth caching for a short TTL rather than redoing on every
request for the same symbol on the same trading day.

Degrades gracefully: if Redis isn't reachable (not started, wrong host in a
non-Docker dev run, etc.), every call is a cache miss and the API just does
the full computation each time — never a hard failure.
"""

from __future__ import annotations

import json
import os

DEFAULT_TTL_SECONDS = 300  # signals are only meaningful per trading session; 5 min is a safe re-check window

_client = None
_client_attempted = False


def _get_client():
    global _client, _client_attempted
    if _client_attempted:
        return _client
    _client_attempted = True
    try:
        import redis
        host = os.environ.get("REDIS_HOST", "localhost")
        port = int(os.environ.get("REDIS_PORT", "6379"))
        client = redis.Redis(host=host, port=port, socket_connect_timeout=1, socket_timeout=1)
        client.ping()
        _client = client
    except Exception:
        _client = None
    return _client


def cache_get(key: str) -> dict | None:
    client = _get_client()
    if client is None:
        return None
    try:
        raw = client.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


def cache_set(key: str, value: dict, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
    client = _get_client()
    if client is None:
        return
    try:
        client.set(key, json.dumps(value, default=str), ex=ttl_seconds)
    except Exception:
        pass  # caching is an optimization, never a request-breaking dependency


def clear_pattern(pattern: str) -> int:
    """Deletes every key matching `pattern` (e.g. "signal:*") — used after a
    data refresh (training/tasks.py::refresh_data) so the next request
    recomputes against the new data instead of serving a stale cached
    response for up to the remaining TTL. Returns the number of keys
    deleted; 0 (not an error) if Redis is unreachable, same
    degrade-gracefully contract as cache_get/cache_set."""
    client = _get_client()
    if client is None:
        return 0
    try:
        keys = list(client.scan_iter(match=pattern))
        if not keys:
            return 0
        return client.delete(*keys)
    except Exception:
        return 0


def reset_client_for_testing() -> None:
    """Forces the next cache_get/cache_set to re-attempt a Redis connection —
    used by tests that need to simulate Redis being unavailable/available."""
    global _client, _client_attempted
    _client, _client_attempted = None, False

"""
Tests the Redis cache against a REAL Redis instance when one is reachable
(same graceful-detection logic the module itself uses), and always tests the
graceful-degradation path (pointing at a port nothing listens on) — the
cache must never raise, only silently miss.
"""

import pytest

from monitoring import cache


@pytest.fixture(autouse=True)
def _reset_cache_client():
    cache.reset_client_for_testing()
    yield
    cache.reset_client_for_testing()


def _redis_reachable() -> bool:
    cache.reset_client_for_testing()
    return cache._get_client() is not None


@pytest.mark.skipif(not _redis_reachable(), reason="no local Redis instance reachable")
def test_cache_set_then_get_roundtrip():
    cache.cache_set("test:roundtrip", {"signal": "BUY", "score": 0.5}, ttl_seconds=30)
    result = cache.cache_get("test:roundtrip")
    assert result == {"signal": "BUY", "score": 0.5}


@pytest.mark.skipif(not _redis_reachable(), reason="no local Redis instance reachable")
def test_cache_miss_for_unknown_key():
    assert cache.cache_get("test:definitely_not_set") is None


def test_cache_degrades_gracefully_when_redis_unreachable(monkeypatch):
    monkeypatch.setenv("REDIS_HOST", "127.0.0.1")
    monkeypatch.setenv("REDIS_PORT", "1")  # nothing listens on port 1
    cache.reset_client_for_testing()

    # Neither call should raise, even though there's no Redis to talk to.
    cache.cache_set("test:key", {"a": 1})
    result = cache.cache_get("test:key")
    assert result is None


@pytest.mark.skipif(not _redis_reachable(), reason="no local Redis instance reachable")
def test_clear_pattern_deletes_matching_keys_only():
    cache.cache_set("test:clear:NVDA", {"a": 1}, ttl_seconds=30)
    cache.cache_set("test:clear:AAPL", {"a": 2}, ttl_seconds=30)
    cache.cache_set("test:unrelated", {"a": 3}, ttl_seconds=30)

    deleted = cache.clear_pattern("test:clear:*")

    assert deleted == 2
    assert cache.cache_get("test:clear:NVDA") is None
    assert cache.cache_get("test:clear:AAPL") is None
    assert cache.cache_get("test:unrelated") == {"a": 3}
    cache.cache_get("test:unrelated")  # sanity: unrelated key untouched
    cache._get_client().delete("test:unrelated")  # test cleanup


@pytest.mark.skipif(not _redis_reachable(), reason="no local Redis instance reachable")
def test_clear_pattern_returns_zero_when_nothing_matches():
    assert cache.clear_pattern("test:nothing_matches_this:*") == 0


def test_clear_pattern_degrades_gracefully_when_redis_unreachable(monkeypatch):
    monkeypatch.setenv("REDIS_HOST", "127.0.0.1")
    monkeypatch.setenv("REDIS_PORT", "1")
    cache.reset_client_for_testing()
    assert cache.clear_pattern("signal:*") == 0

"""Tests for in-memory TTL cache module."""

from __future__ import annotations

import time

import pytest

from app.cache import (
    clear_cache,
    evict_expired,
    get_cache_stats,
    ttl_cache,
)


@pytest.fixture(autouse=True)
def clean_cache():
    clear_cache()
    yield
    clear_cache()


class TestTTLCacheDecorator:
    def test_basic_caching(self):
        call_count = [0]

        @ttl_cache(ttl_seconds=10)
        def expensive(x: int) -> int:
            call_count[0] += 1
            return x * 2

        result1 = expensive(5)
        result2 = expensive(5)
        assert result1 == result2 == 10
        assert call_count[0] == 1

    def test_different_args_not_shared(self):
        call_count = [0]

        @ttl_cache(ttl_seconds=10)
        def f(x: int) -> int:
            call_count[0] += 1
            return x

        f(1)
        f(2)
        assert call_count[0] == 2

    def test_cache_miss_after_expiry(self):
        call_count = [0]

        @ttl_cache(ttl_seconds=0)
        def g(x: int) -> int:
            call_count[0] += 1
            return x

        g(1)
        time.sleep(0.01)
        g(1)
        assert call_count[0] == 2


class TestCacheStats:
    def test_empty_cache_stats(self):
        stats = get_cache_stats()
        assert stats["live_entries"] == 0

    def test_stats_after_cache_write(self):
        @ttl_cache(ttl_seconds=60)
        def h(x: int) -> int:
            return x

        h(42)
        stats = get_cache_stats()
        assert stats["live_entries"] >= 1


class TestClearCache:
    def test_clear_removes_all_entries(self):
        @ttl_cache(ttl_seconds=60)
        def k(x: int) -> int:
            return x

        k(1)
        k(2)
        removed = clear_cache()
        assert removed >= 2
        assert get_cache_stats()["total_entries"] == 0


class TestEvictExpired:
    def test_evict_expired_entries(self):
        @ttl_cache(ttl_seconds=0)
        def m(x: int) -> int:
            return x

        m(1)
        time.sleep(0.01)
        removed = evict_expired()
        assert removed >= 1

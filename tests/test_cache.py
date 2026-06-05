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

    def test_evict_keeps_live_entries(self):
        @ttl_cache(ttl_seconds=60)
        def n(x: int) -> int:
            return x

        n(1)
        n(2)
        removed = evict_expired()
        assert removed == 0
        assert get_cache_stats()["live_entries"] >= 2


class TestCacheKeyIsolation:
    def test_kwargs_cached_separately(self):
        calls = [0]

        @ttl_cache(ttl_seconds=60)
        def func(x: int, y: int = 0) -> int:
            calls[0] += 1
            return x + y

        func(1, y=1)
        func(1, y=2)
        assert calls[0] == 2

    def test_cache_returns_correct_value(self):
        @ttl_cache(ttl_seconds=60)
        def square(n: int) -> int:
            return n * n

        for i in range(1, 6):
            assert square(i) == i * i

    def test_cache_stats_keys_are_truncated(self):
        @ttl_cache(ttl_seconds=60)
        def longfunc(x: str) -> str:
            return x

        longfunc("a" * 200)
        stats = get_cache_stats()
        for key in stats["keys"]:
            assert len(key) <= 80

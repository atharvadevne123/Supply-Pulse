"""In-memory TTL cache for expensive computations."""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)

_cache: dict[str, tuple[Any, float]] = {}
_MAX_CACHE_ENTRIES = 1_000


def _evict_one_entry() -> None:
    """Remove the oldest entry from the cache when capacity is exceeded."""
    if _cache:
        oldest_key = next(iter(_cache))
        del _cache[oldest_key]


def ttl_cache(ttl_seconds: int = 60) -> Callable:
    """Decorator that caches function results for ttl_seconds seconds.

    Args:
        ttl_seconds: Number of seconds before cached entry expires.

    Returns:
        Decorated function with caching applied.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache_key = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
            now = time.monotonic()
            if cache_key in _cache:
                value, expires_at = _cache[cache_key]
                if now < expires_at:
                    logger.debug("Cache hit for '%s'", func.__name__)
                    return value
                del _cache[cache_key]
            if len(_cache) >= _MAX_CACHE_ENTRIES:
                _evict_one_entry()
            result = func(*args, **kwargs)
            _cache[cache_key] = (result, now + ttl_seconds)
            logger.debug("Cache miss for '%s' - stored with TTL=%ds", func.__name__, ttl_seconds)
            return result

        return wrapper

    return decorator


def get_cache_stats() -> dict[str, Any]:
    """Return current cache state summary.

    Returns:
        Dictionary with entry count, keys, and expiry times.
    """
    now = time.monotonic()
    live_entries = [(k, v[1] - now) for k, v in _cache.items() if v[1] > now]
    return {
        "total_entries": len(_cache),
        "live_entries": len(live_entries),
        "keys": [e[0][:80] for e in live_entries],
    }


def clear_cache() -> int:
    """Evict all entries from cache.

    Returns:
        Number of entries removed.
    """
    n = len(_cache)
    _cache.clear()
    logger.info("Cache cleared: %d entries removed", n)
    return n


def evict_expired() -> int:
    """Remove expired entries from cache.

    Returns:
        Number of expired entries removed.
    """
    now = time.monotonic()
    expired_keys = [k for k, (_, exp) in _cache.items() if now >= exp]
    for k in expired_keys:
        del _cache[k]
    if expired_keys:
        logger.debug("Evicted %d expired cache entries", len(expired_keys))
    return len(expired_keys)

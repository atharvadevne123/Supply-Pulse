"""Prometheus-compatible metrics counters for Supply-Pulse API."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

_counters: dict[str, int] = defaultdict(int)
_histograms: dict[str, list[float]] = defaultdict(list)
_start_time: float = time.time()


def increment(name: str, value: int = 1) -> None:
    """Increment a named counter by value."""
    _counters[name] += value


def record_latency(name: str, latency_ms: float) -> None:
    """Append a latency observation to a named histogram."""
    _histograms[name].append(latency_ms)
    if len(_histograms[name]) > 10000:
        _histograms[name] = _histograms[name][-5000:]


def get_counter(name: str) -> int:
    """Return the current value of a named counter."""
    return _counters.get(name, 0)


def get_histogram_stats(name: str) -> dict[str, float]:
    """Return summary statistics for a named latency histogram.

    Args:
        name: Histogram name.

    Returns:
        Dict with count, mean, p50, p95, p99, max latency values.
    """
    import numpy as np

    values = _histograms.get(name, [])
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    arr = np.array(values)
    return {
        "count": len(arr),
        "mean": round(float(arr.mean()), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "p99": round(float(np.percentile(arr, 99)), 2),
        "max": round(float(arr.max()), 2),
    }


def get_all_metrics() -> dict[str, Any]:
    """Return all counters and histogram summaries.

    Returns:
        Dictionary with uptime_seconds, counters, and latency histograms.
    """
    uptime = round(time.time() - _start_time, 2)
    latency_stats = {name: get_histogram_stats(name) for name in _histograms}
    return {
        "uptime_seconds": uptime,
        "counters": dict(_counters),
        "latency_histograms": latency_stats,
    }


def reset_metrics() -> None:
    """Clear all counters and histograms — intended for testing only."""
    global _start_time
    _counters.clear()
    _histograms.clear()
    _start_time = time.time()

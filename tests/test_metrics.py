"""Tests for metrics module."""

from __future__ import annotations

import pytest

from app.metrics import (
    get_all_metrics,
    get_counter,
    get_histogram_stats,
    increment,
    record_latency,
    reset_metrics,
)


@pytest.fixture(autouse=True)
def reset():
    reset_metrics()
    yield
    reset_metrics()


class TestCounters:
    def test_increment_default(self):
        increment("test_counter")
        assert get_counter("test_counter") == 1

    def test_increment_by_value(self):
        increment("test_counter", 5)
        assert get_counter("test_counter") == 5

    def test_multiple_increments(self):
        increment("test_counter")
        increment("test_counter")
        increment("test_counter")
        assert get_counter("test_counter") == 3

    def test_unknown_counter_returns_zero(self):
        assert get_counter("nonexistent") == 0


class TestHistograms:
    def test_record_latency(self):
        record_latency("api_latency", 10.5)
        stats = get_histogram_stats("api_latency")
        assert stats["count"] == 1

    def test_empty_histogram(self):
        stats = get_histogram_stats("empty_hist")
        assert stats["count"] == 0
        assert stats["mean"] == 0.0

    def test_histogram_percentiles(self):
        for v in range(1, 101):
            record_latency("p_test", float(v))
        stats = get_histogram_stats("p_test")
        assert stats["p50"] > 0
        assert stats["p95"] > stats["p50"]
        assert stats["p99"] >= stats["p95"]

    def test_histogram_max(self):
        for v in [10.0, 20.0, 50.0, 100.0]:
            record_latency("max_test", v)
        stats = get_histogram_stats("max_test")
        assert stats["max"] == 100.0


class TestGetAllMetrics:
    def test_returns_uptime(self):
        metrics = get_all_metrics()
        assert "uptime_seconds" in metrics
        assert metrics["uptime_seconds"] >= 0

    def test_returns_counters(self):
        increment("test_c")
        metrics = get_all_metrics()
        assert "counters" in metrics
        assert "test_c" in metrics["counters"]

    def test_returns_latency_histograms(self):
        record_latency("req_latency", 15.0)
        metrics = get_all_metrics()
        assert "latency_histograms" in metrics
        assert "req_latency" in metrics["latency_histograms"]

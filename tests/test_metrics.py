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


class TestMetricsEdgeCases:
    def test_increment_multiple_counters_independent(self):
        increment("counter_a", 3)
        increment("counter_b", 7)
        assert get_counter("counter_a") == 3
        assert get_counter("counter_b") == 7

    def test_histogram_trims_at_10000(self):
        for i in range(10_005):
            record_latency("big_hist", float(i))
        stats = get_histogram_stats("big_hist")
        assert stats["count"] <= 10_000

    def test_histogram_mean_is_reasonable(self):
        for v in [10.0, 20.0, 30.0]:
            record_latency("mean_test", v)
        stats = get_histogram_stats("mean_test")
        assert stats["mean"] == pytest.approx(20.0, rel=0.01)

    @pytest.mark.parametrize("value", [0.0, 0.001, 1000.0, 99999.9])
    def test_record_extreme_latencies(self, value):
        record_latency("extreme", value)
        stats = get_histogram_stats("extreme")
        assert stats["count"] >= 1

    def test_reset_clears_counters_and_histograms(self):
        increment("to_reset", 5)
        record_latency("to_reset_hist", 10.0)
        reset_metrics()
        assert get_counter("to_reset") == 0
        assert get_histogram_stats("to_reset_hist")["count"] == 0

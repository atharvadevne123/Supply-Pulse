"""Tests for demand forecasting module."""

from __future__ import annotations

import pytest

from app.demand_forecast import compute_demand_statistics, forecast_demand


class TestForecastDemand:
    def test_basic_forecast_returns_horizon_values(self):
        history = [100.0] * 24
        result = forecast_demand(history, horizon=6)
        assert len(result["forecast"]) == 6

    def test_forecast_with_trend(self):
        history = list(range(1, 25))
        result = forecast_demand(history, horizon=3)
        assert result["forecast"][0] > 0

    def test_forecast_returns_confidence_intervals(self):
        history = [100.0 + i for i in range(24)]
        result = forecast_demand(history, horizon=4)
        assert result["lower_bound"] is not None
        assert result["upper_bound"] is not None
        assert len(result["lower_bound"]) == 4

    def test_lower_bound_lte_forecast(self):
        history = [50.0 + i * 2 for i in range(24)]
        result = forecast_demand(history, horizon=6)
        for lo, fc in zip(result["lower_bound"], result["forecast"]):
            assert lo <= fc + 0.01

    def test_upper_bound_gte_forecast(self):
        history = [50.0 + i * 2 for i in range(24)]
        result = forecast_demand(history, horizon=6)
        for hi, fc in zip(result["upper_bound"], result["forecast"]):
            assert hi >= fc - 0.01

    def test_short_history_returns_insufficient(self):
        result = forecast_demand([100.0], horizon=3)
        assert "reason" in result
        assert result["reason"] == "insufficient_history"

    def test_empty_history(self):
        result = forecast_demand([], horizon=3)
        assert len(result["forecast"]) == 3

    @pytest.mark.parametrize("horizon", [1, 3, 6, 12])
    def test_various_horizons(self, horizon):
        history = [100.0] * 24
        result = forecast_demand(history, horizon=horizon)
        assert len(result["forecast"]) == horizon

    def test_forecast_non_negative(self):
        history = [50.0] * 24
        result = forecast_demand(history, horizon=6)
        assert all(f >= 0 for f in result["forecast"])

    def test_returns_trend_slope(self):
        history = list(range(1, 25))
        result = forecast_demand(history, horizon=3)
        assert "trend_slope" in result

    def test_returns_history_length(self):
        history = [100.0] * 20
        result = forecast_demand(history, horizon=3)
        assert result["history_length"] == 20


class TestComputeDemandStatistics:
    def test_empty_history(self):
        stats = compute_demand_statistics([])
        assert stats["mean"] == 0.0

    def test_constant_demand(self):
        stats = compute_demand_statistics([100.0] * 10)
        assert stats["mean"] == 100.0
        assert stats["std"] == 0.0
        assert stats["cv"] == 0.0

    def test_cv_calculation(self):
        stats = compute_demand_statistics([80.0, 90.0, 100.0, 110.0, 120.0])
        assert stats["cv"] > 0

    @pytest.mark.parametrize(
        "values,expected_min,expected_max",
        [
            ([10, 20, 30], 10.0, 30.0),
            ([5], 5.0, 5.0),
        ],
    )
    def test_min_max_values(self, values, expected_min, expected_max):
        stats = compute_demand_statistics(values)
        assert stats["min"] == expected_min
        assert stats["max"] == expected_max

    def test_increasing_trend(self):
        stats = compute_demand_statistics(list(range(1, 21)))
        assert stats["trend"] > 0

    def test_decreasing_trend(self):
        stats = compute_demand_statistics(list(range(20, 0, -1)))
        assert stats["trend"] < 0

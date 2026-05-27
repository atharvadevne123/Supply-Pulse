"""Extended reorder point and inventory calculation tests."""

from __future__ import annotations

import pytest

from app.model import compute_reorder_point


class TestReorderPointExtended:
    @pytest.mark.parametrize(
        "mean,std,lead,z",
        [
            (50.0, 10.0, 7, 1.28),
            (200.0, 50.0, 21, 1.645),
            (500.0, 100.0, 30, 2.326),
            (10.0, 2.0, 3, 1.645),
        ],
    )
    def test_reorder_formula_components(self, mean, std, lead, z):
        result = compute_reorder_point(mean, std, lead, z)
        expected_lead_demand = mean * lead
        expected_safety_stock = z * std * (lead**0.5)
        assert abs(result["lead_demand"] - round(expected_lead_demand, 2)) < 0.01
        assert abs(result["safety_stock"] - round(expected_safety_stock, 2)) < 0.01

    def test_higher_service_level_more_safety_stock(self):
        low_z = compute_reorder_point(100.0, 20.0, 14, 1.28)
        high_z = compute_reorder_point(100.0, 20.0, 14, 2.326)
        assert high_z["safety_stock"] > low_z["safety_stock"]

    def test_longer_lead_time_higher_reorder_point(self):
        short = compute_reorder_point(100.0, 20.0, 7, 1.645)
        long = compute_reorder_point(100.0, 20.0, 30, 1.645)
        assert long["reorder_point"] > short["reorder_point"]

    def test_higher_demand_higher_reorder_point(self):
        low = compute_reorder_point(50.0, 10.0, 14, 1.645)
        high = compute_reorder_point(200.0, 40.0, 14, 1.645)
        assert high["reorder_point"] > low["reorder_point"]

    def test_result_is_sum_of_components(self):
        result = compute_reorder_point(100.0, 20.0, 14, 1.645)
        expected = result["lead_demand"] + result["safety_stock"]
        assert abs(result["reorder_point"] - round(expected, 2)) < 0.01

    def test_returns_float_values(self):
        result = compute_reorder_point(100.0, 20.0, 14, 1.645)
        assert isinstance(result["reorder_point"], float)
        assert isinstance(result["safety_stock"], float)

    @pytest.mark.parametrize("lead", [1, 7, 14, 30, 90])
    def test_various_lead_times(self, lead):
        result = compute_reorder_point(100.0, 10.0, lead, 1.645)
        assert result["reorder_point"] > 0

    def test_very_high_demand(self):
        result = compute_reorder_point(10000.0, 500.0, 30, 1.645)
        assert result["reorder_point"] > 100000

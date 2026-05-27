"""Tests for input validation utilities."""

from __future__ import annotations

import pytest

from app.validators import (
    sanitize_supplier_fields,
    validate_demand_history,
    validate_supplier_input,
)


class TestValidateSupplierInput:
    def test_valid_input_returns_no_errors(self):
        data = {
            "on_time_rate": 0.9, "defect_rate": 0.02, "financial_score": 0.8,
            "geopolitical_risk": 0.3, "capacity_utilization": 0.6,
            "lead_time_days": 30, "years_active": 10,
        }
        errors = validate_supplier_input(data)
        assert errors == []

    @pytest.mark.parametrize("field,value", [
        ("on_time_rate", 1.5),
        ("defect_rate", -0.1),
        ("financial_score", 2.0),
        ("geopolitical_risk", -0.5),
    ])
    def test_out_of_range_returns_error(self, field, value):
        data = {field: value}
        errors = validate_supplier_input(data)
        assert len(errors) > 0

    def test_negative_lead_time(self):
        errors = validate_supplier_input({"lead_time_days": 0})
        assert any("lead_time_days" in e for e in errors)

    def test_negative_years_active(self):
        errors = validate_supplier_input({"years_active": -1})
        assert any("years_active" in e for e in errors)


class TestValidateDemandHistory:
    def test_valid_history(self):
        errors = validate_demand_history([100.0, 110.0, 120.0])
        assert errors == []

    def test_empty_history(self):
        errors = validate_demand_history([])
        assert len(errors) > 0

    def test_negative_values(self):
        errors = validate_demand_history([100.0, -5.0, 80.0])
        assert len(errors) > 0

    def test_too_long_history(self):
        errors = validate_demand_history([1.0] * 10001)
        assert len(errors) > 0


class TestSanitizeSupplierFields:
    def test_none_values_get_defaults(self):
        safe = sanitize_supplier_fields({})
        assert safe["on_time_rate"] == 0.9
        assert safe["country"] == "US"

    def test_out_of_range_clipped(self):
        safe = sanitize_supplier_fields({"on_time_rate": 2.0, "defect_rate": -0.5})
        assert safe["on_time_rate"] == 1.0
        assert safe["defect_rate"] == 0.0

    def test_is_sole_source_cast_to_int(self):
        safe = sanitize_supplier_fields({"is_sole_source": True})
        assert safe["is_sole_source"] == 1

    def test_country_truncated(self):
        safe = sanitize_supplier_fields({"country": "X" * 100})
        assert len(safe["country"]) <= 50

    @pytest.mark.parametrize("lead_time,expected", [(0, 1), (30, 30), (-5, 1)])
    def test_lead_time_minimum(self, lead_time, expected):
        safe = sanitize_supplier_fields({"lead_time_days": lead_time})
        assert safe["lead_time_days"] == expected

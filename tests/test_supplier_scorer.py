"""Tests for supplier scorecard module."""

from __future__ import annotations

import pytest

from app.supplier_scorer import (
    _grade,
    compute_scorecard,
    score_capacity,
    score_delivery,
    score_financial,
    score_geopolitical,
    score_quality,
)

GOOD_SUPPLIER = {
    "on_time_rate": 0.98, "lead_time_days": 10, "defect_rate": 0.005,
    "years_active": 20, "financial_score": 0.95, "geopolitical_risk": 0.10,
    "is_sole_source": 0, "capacity_utilization": 0.40,
}

BAD_SUPPLIER = {
    "on_time_rate": 0.55, "lead_time_days": 100, "defect_rate": 0.14,
    "years_active": 1, "financial_score": 0.30, "geopolitical_risk": 0.90,
    "is_sole_source": 1, "capacity_utilization": 0.98,
}


class TestDeliveryScore:
    @pytest.mark.parametrize("on_time,lead_time,expected_min", [
        (0.99, 7, 0.7),
        (0.55, 90, 0.0),
    ])
    def test_delivery_score_range(self, on_time, lead_time, expected_min):
        score = score_delivery(on_time, lead_time)
        assert score >= expected_min
        assert score <= 1.0

    def test_high_on_time_high_score(self):
        assert score_delivery(0.99, 7) > score_delivery(0.60, 90)


class TestQualityScore:
    def test_zero_defect_gives_high_score(self):
        score = score_quality(0.0, 15)
        assert score >= 0.9

    def test_high_defect_gives_low_score(self):
        score = score_quality(0.14, 1)
        assert score < 0.5

    def test_maturity_bonus_applied(self):
        young = score_quality(0.02, 1)
        old = score_quality(0.02, 20)
        assert old >= young


class TestFinancialScore:
    @pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
    def test_passthrough_values(self, score):
        assert score_financial(score) == score

    def test_clipped_above_one(self):
        assert score_financial(1.5) == 1.0

    def test_clipped_below_zero(self):
        assert score_financial(-0.5) == 0.0


class TestGeopoliticalScore:
    def test_low_risk_high_score(self):
        assert score_geopolitical(0.1, 0) > 0.8

    def test_high_risk_low_score(self):
        assert score_geopolitical(0.9, 0) < 0.2

    def test_sole_source_penalty(self):
        multi = score_geopolitical(0.4, 0)
        sole = score_geopolitical(0.4, 1)
        assert multi > sole


class TestCapacityScore:
    @pytest.mark.parametrize("util,expected_max", [
        (0.0, 1.0),
        (1.0, 0.0),
    ])
    def test_capacity_extremes(self, util, expected_max):
        score = score_capacity(util)
        assert abs(score - expected_max) < 0.01


class TestComputeScorecard:
    def test_good_supplier_high_score(self):
        card = compute_scorecard(GOOD_SUPPLIER)
        assert card["total_score"] > 0.6

    def test_bad_supplier_low_score(self):
        card = compute_scorecard(BAD_SUPPLIER)
        assert card["total_score"] < 0.5

    def test_scorecard_contains_all_components(self):
        card = compute_scorecard(GOOD_SUPPLIER)
        for key in ["delivery", "quality", "financial", "geopolitical", "capacity"]:
            assert key in card["components"]

    def test_scorecard_grade_in_valid_set(self):
        card = compute_scorecard(GOOD_SUPPLIER)
        assert card["grade"] in ("A", "B", "C", "D", "F")

    def test_good_supplier_gets_higher_grade(self):
        good_card = compute_scorecard(GOOD_SUPPLIER)
        bad_card = compute_scorecard(BAD_SUPPLIER)
        assert good_card["total_score"] > bad_card["total_score"]


class TestGrade:
    @pytest.mark.parametrize("score,expected", [
        (0.90, "A"), (0.75, "B"), (0.60, "C"), (0.45, "D"), (0.30, "F"),
    ])
    def test_grade_boundaries(self, score, expected):
        assert _grade(score) == expected

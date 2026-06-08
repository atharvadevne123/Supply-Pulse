"""Tests for risk report generation module."""

from __future__ import annotations

import pytest

from app.risk_report import (
    build_risk_report,
    classify_risk,
    generate_recommendations,
)

MOCK_DISRUPTION_HIGH = {"disruption_risk": 0.85, "disruption_label": "HIGH"}
MOCK_DISRUPTION_LOW = {"disruption_risk": 0.10, "disruption_label": "LOW"}
MOCK_SCORECARD_GOOD = {
    "grade": "A",
    "total_score": 0.88,
    "components": {
        "delivery": 0.95,
        "quality": 0.92,
        "financial": 0.90,
        "geopolitical": 0.85,
        "capacity": 0.80,
    },
}
MOCK_SCORECARD_BAD = {
    "grade": "F",
    "total_score": 0.25,
    "components": {
        "delivery": 0.30,
        "quality": 0.20,
        "financial": 0.25,
        "geopolitical": 0.15,
        "capacity": 0.10,
    },
}


class TestClassifyRisk:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (0.90, "CRITICAL"),
            (0.65, "HIGH"),
            (0.50, "MEDIUM"),
            (0.20, "LOW"),
            (0.0, "LOW"),
        ],
    )
    def test_risk_classification(self, score, expected):
        assert classify_risk(score) == expected

    def test_boundary_at_80(self):
        assert classify_risk(0.80) == "CRITICAL"

    def test_just_below_60(self):
        assert classify_risk(0.59) == "MEDIUM"


class TestGenerateRecommendations:
    def test_high_risk_recommends_alternative(self):
        recs = generate_recommendations(0.80, MOCK_SCORECARD_BAD)
        assert any("alternative" in r.lower() for r in recs)

    def test_medium_risk_recommends_safety_stock(self):
        recs = generate_recommendations(0.45, MOCK_SCORECARD_BAD)
        assert any("safety stock" in r.lower() for r in recs)

    def test_low_risk_good_supplier_minimal_recs(self):
        recs = generate_recommendations(0.05, MOCK_SCORECARD_GOOD)
        assert len(recs) >= 1

    def test_bad_quality_recommends_inspection(self):
        recs = generate_recommendations(0.30, MOCK_SCORECARD_BAD)
        assert any("quality" in r.lower() or "inspection" in r.lower() for r in recs)

    def test_returns_list_of_strings(self):
        recs = generate_recommendations(0.50, MOCK_SCORECARD_BAD)
        assert isinstance(recs, list)
        assert all(isinstance(r, str) for r in recs)


class TestBuildRiskReport:
    def test_report_contains_required_fields(self):
        report = build_risk_report("S-001", "Test Corp", MOCK_DISRUPTION_HIGH, MOCK_SCORECARD_BAD)
        for field in [
            "supplier_id",
            "supplier_name",
            "disruption_risk",
            "severity",
            "grade",
            "recommendations",
        ]:
            assert field in report

    def test_report_severity_matches_risk(self):
        report = build_risk_report("S-001", "Corp", MOCK_DISRUPTION_HIGH, MOCK_SCORECARD_GOOD)
        assert report["severity"] == "CRITICAL"

    def test_report_low_risk(self):
        report = build_risk_report("S-002", "Safe Corp", MOCK_DISRUPTION_LOW, MOCK_SCORECARD_GOOD)
        assert report["severity"] == "LOW"

    def test_report_has_generated_at(self):
        report = build_risk_report("S-003", "Corp", MOCK_DISRUPTION_LOW, MOCK_SCORECARD_GOOD)
        assert "generated_at" in report
        assert isinstance(report["generated_at"], str)

    def test_report_has_n_recommendations(self):
        report = build_risk_report("S-004", "Corp", MOCK_DISRUPTION_HIGH, MOCK_SCORECARD_BAD)
        assert report["n_recommendations"] == len(report["recommendations"])

    def test_report_generated_at_is_iso_format(self):
        import datetime

        report = build_risk_report("S-005", "Corp", MOCK_DISRUPTION_LOW, MOCK_SCORECARD_GOOD)
        dt = datetime.datetime.fromisoformat(report["generated_at"])
        assert dt is not None

    def test_report_total_score_matches_scorecard(self):
        report = build_risk_report("S-006", "Corp", MOCK_DISRUPTION_LOW, MOCK_SCORECARD_GOOD)
        assert report["total_score"] == MOCK_SCORECARD_GOOD["total_score"]

    def test_report_components_matches_scorecard(self):
        report = build_risk_report("S-007", "Corp", MOCK_DISRUPTION_HIGH, MOCK_SCORECARD_BAD)
        assert report["components"] == MOCK_SCORECARD_BAD["components"]

    @pytest.mark.parametrize(
        "risk_score,expected_severity",
        [
            (0.81, "CRITICAL"),
            (0.61, "HIGH"),
            (0.41, "MEDIUM"),
            (0.10, "LOW"),
        ],
    )
    def test_severity_thresholds(self, risk_score, expected_severity):
        disruption = {"disruption_risk": risk_score, "disruption_label": "TEST"}
        report = build_risk_report("S-P", "Corp", disruption, MOCK_SCORECARD_GOOD)
        assert report["severity"] == expected_severity

    def test_geopolitical_risk_triggers_recommendation(self):
        scorecard_geo = {
            **MOCK_SCORECARD_GOOD,
            "components": {**MOCK_SCORECARD_GOOD["components"], "geopolitical": 0.3},
        }
        recs = generate_recommendations(0.50, scorecard_geo)
        assert any("geopolit" in r.lower() or "dual-source" in r.lower() for r in recs)

    def test_financial_risk_triggers_recommendation(self):
        scorecard_fin = {
            **MOCK_SCORECARD_GOOD,
            "components": {**MOCK_SCORECARD_GOOD["components"], "financial": 0.4},
        }
        recs = generate_recommendations(0.30, scorecard_fin)
        assert any("financial" in r.lower() for r in recs)

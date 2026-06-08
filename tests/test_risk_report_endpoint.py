"""Tests for the /api/v1/suppliers/risk-report endpoint."""

from __future__ import annotations

import pytest


class TestRiskReportEndpoint:
    def test_risk_report_returns_200(self, client, risk_report_payload):
        resp = client.post("/api/v1/suppliers/risk-report", json=risk_report_payload)
        assert resp.status_code == 200

    def test_risk_report_contains_required_fields(self, client, risk_report_payload):
        resp = client.post("/api/v1/suppliers/risk-report", json=risk_report_payload)
        data = resp.json()
        assert "supplier_id" in data
        assert "supplier_name" in data
        assert "disruption_risk" in data
        assert "severity" in data
        assert "grade" in data
        assert "recommendations" in data

    def test_risk_report_supplier_id_matches_input(self, client, risk_report_payload):
        resp = client.post("/api/v1/suppliers/risk-report", json=risk_report_payload)
        assert resp.json()["supplier_id"] == risk_report_payload["supplier_id"]

    def test_risk_report_supplier_name_matches_input(self, client, risk_report_payload):
        resp = client.post("/api/v1/suppliers/risk-report", json=risk_report_payload)
        assert resp.json()["supplier_name"] == risk_report_payload["supplier_name"]

    def test_risk_report_disruption_risk_in_range(self, client, risk_report_payload):
        resp = client.post("/api/v1/suppliers/risk-report", json=risk_report_payload)
        risk = resp.json()["disruption_risk"]
        assert 0.0 <= risk <= 1.0

    def test_risk_report_severity_valid(self, client, risk_report_payload):
        resp = client.post("/api/v1/suppliers/risk-report", json=risk_report_payload)
        assert resp.json()["severity"] in ("CRITICAL", "HIGH", "MEDIUM", "LOW")

    def test_risk_report_grade_valid(self, client, risk_report_payload):
        resp = client.post("/api/v1/suppliers/risk-report", json=risk_report_payload)
        assert resp.json()["grade"] in ("A", "B", "C", "D", "F")

    def test_risk_report_recommendations_is_list(self, client, risk_report_payload):
        resp = client.post("/api/v1/suppliers/risk-report", json=risk_report_payload)
        assert isinstance(resp.json()["recommendations"], list)

    def test_risk_report_has_generated_at(self, client, risk_report_payload):
        resp = client.post("/api/v1/suppliers/risk-report", json=risk_report_payload)
        assert "generated_at" in resp.json()

    def test_risk_report_high_risk_supplier_has_recommendations(
        self, client, high_risk_supplier_payload
    ):
        payload = {
            "supplier_id": "SUPP-HIGH",
            "supplier_name": "High Risk Corp",
            "supplier": high_risk_supplier_payload,
        }
        resp = client.post("/api/v1/suppliers/risk-report", json=payload)
        assert resp.status_code == 200
        assert len(resp.json()["recommendations"]) >= 1

    def test_risk_report_missing_supplier_id_returns_422(self, client, supplier_payload):
        payload = {"supplier_name": "Test", "supplier": supplier_payload}
        resp = client.post("/api/v1/suppliers/risk-report", json=payload)
        assert resp.status_code == 422

    @pytest.mark.parametrize(
        "country,category",
        [
            ("US", "electronics"),
            ("CN", "semiconductor"),
            ("IN", "textile"),
        ],
    )
    def test_risk_report_various_suppliers(self, client, supplier_payload, country, category):
        payload = {
            "supplier_id": f"SUPP-{country}",
            "supplier_name": f"Supplier {country}",
            "supplier": {**supplier_payload, "country": country, "category": category},
        }
        resp = client.post("/api/v1/suppliers/risk-report", json=payload)
        assert resp.status_code == 200
        assert 0.0 <= resp.json()["disruption_risk"] <= 1.0

"""Tests for the supplier scorecard API endpoint."""

from __future__ import annotations

import pytest


class TestScorecardEndpoint:
    def test_scorecard_basic(self, client, supplier_payload):
        resp = client.post("/api/v1/suppliers/scorecard", json=supplier_payload)
        assert resp.status_code == 200

    def test_scorecard_has_components(self, client, supplier_payload):
        resp = client.post("/api/v1/suppliers/scorecard", json=supplier_payload)
        data = resp.json()
        assert "components" in data
        for key in ["delivery", "quality", "financial", "geopolitical", "capacity"]:
            assert key in data["components"]

    def test_scorecard_has_total_score(self, client, supplier_payload):
        resp = client.post("/api/v1/suppliers/scorecard", json=supplier_payload)
        data = resp.json()
        assert "total_score" in data
        assert 0.0 <= data["total_score"] <= 1.0

    def test_scorecard_has_grade(self, client, supplier_payload):
        resp = client.post("/api/v1/suppliers/scorecard", json=supplier_payload)
        assert resp.json()["grade"] in ("A", "B", "C", "D", "F")

    def test_good_supplier_high_grade(self, client):
        good_supplier = {
            "lead_time_days": 7,
            "on_time_rate": 0.99,
            "defect_rate": 0.001,
            "financial_score": 0.99,
            "geopolitical_risk": 0.05,
            "capacity_utilization": 0.30,
            "years_active": 25,
            "is_sole_source": 0,
            "country": "US",
            "category": "logistics",
        }
        resp = client.post("/api/v1/suppliers/scorecard", json=good_supplier)
        assert resp.status_code == 200
        assert resp.json()["grade"] in ("A", "B")

    def test_bad_supplier_low_grade(self, client, high_risk_supplier_payload):
        resp = client.post("/api/v1/suppliers/scorecard", json=high_risk_supplier_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["grade"] in ("C", "D", "F")

    def test_scorecard_invalid_payload(self, client):
        resp = client.post("/api/v1/suppliers/scorecard", json={"on_time_rate": 2.0})
        assert resp.status_code == 422

    @pytest.mark.parametrize("country", ["US", "CN", "DE"])
    def test_scorecard_various_countries(self, client, supplier_payload, country):
        payload = dict(supplier_payload)
        payload["country"] = country
        resp = client.post("/api/v1/suppliers/scorecard", json=payload)
        assert resp.status_code == 200

    def test_scorecard_has_weights(self, client, supplier_payload):
        resp = client.post("/api/v1/suppliers/scorecard", json=supplier_payload)
        data = resp.json()
        assert "weights" in data
        weights = data["weights"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001

    def test_scorecard_sole_source_lowers_score(self, client, supplier_payload):
        multi = dict(supplier_payload)
        multi["is_sole_source"] = 0
        sole = dict(supplier_payload)
        sole["is_sole_source"] = 1

        multi_resp = client.post("/api/v1/suppliers/scorecard", json=multi)
        sole_resp = client.post("/api/v1/suppliers/scorecard", json=sole)
        assert sole_resp.json()["total_score"] <= multi_resp.json()["total_score"]

    @pytest.mark.parametrize("category", ["electronics", "logistics", "pharmaceutical"])
    def test_scorecard_various_categories(self, client, supplier_payload, category):
        payload = {**supplier_payload, "category": category}
        resp = client.post("/api/v1/suppliers/scorecard", json=payload)
        assert resp.status_code == 200
        assert 0.0 <= resp.json()["total_score"] <= 1.0

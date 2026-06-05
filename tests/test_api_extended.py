"""Extended API tests covering edge cases and demand/anomaly endpoints."""

from __future__ import annotations

import pytest


class TestDemandForecastEndpoint:
    def test_forecast_basic(self, client):
        payload = {"history": [100.0] * 24, "horizon": 6}
        resp = client.post("/api/v1/demand/forecast", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["forecast"]) == 6

    def test_forecast_returns_confidence_intervals(self, client):
        payload = {"history": list(range(1, 25)), "horizon": 4}
        resp = client.post("/api/v1/demand/forecast", json=payload)
        assert resp.status_code == 200
        assert "lower_bound" in resp.json()

    def test_forecast_includes_stats(self, client):
        payload = {"history": [100.0, 110.0, 120.0] * 8, "horizon": 3}
        resp = client.post("/api/v1/demand/forecast", json=payload)
        assert resp.status_code == 200
        assert "stats" in resp.json()

    def test_forecast_invalid_horizon(self, client):
        payload = {"history": [100.0] * 12, "horizon": 100}
        resp = client.post("/api/v1/demand/forecast", json=payload)
        assert resp.status_code == 422

    def test_forecast_empty_history(self, client):
        payload = {"history": [], "horizon": 3}
        resp = client.post("/api/v1/demand/forecast", json=payload)
        assert resp.status_code == 422

    @pytest.mark.parametrize("horizon", [1, 6, 12, 24])
    def test_forecast_various_horizons(self, client, horizon):
        payload = {"history": [100.0] * 24, "horizon": horizon}
        resp = client.post("/api/v1/demand/forecast", json=payload)
        assert resp.status_code == 200
        assert len(resp.json()["forecast"]) == horizon


class TestSimilarSuppliersEndpoint:
    def test_similar_suppliers_basic(self, client, supplier_payload):
        payload = {"supplier": supplier_payload, "top_k": 3}
        resp = client.post("/api/v1/suppliers/similar", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "similar_suppliers" in data
        assert "count" in data

    def test_similar_suppliers_top_k_respected(self, client, supplier_payload):
        payload = {"supplier": supplier_payload, "top_k": 2}
        resp = client.post("/api/v1/suppliers/similar", json=payload)
        assert resp.status_code == 200
        assert resp.json()["count"] <= 2

    def test_similar_suppliers_invalid_top_k(self, client, supplier_payload):
        payload = {"supplier": supplier_payload, "top_k": 0}
        resp = client.post("/api/v1/suppliers/similar", json=payload)
        assert resp.status_code == 422


class TestOpsEndpoints:
    def test_health_has_model_version(self, client):
        resp = client.get("/health")
        assert "model_version" in resp.json()

    def test_readyz_has_ready_field(self, client):
        resp = client.get("/readyz")
        assert "ready" in resp.json()

    def test_version_has_both_versions(self, client):
        resp = client.get("/version")
        data = resp.json()
        assert "app_version" in data
        assert "model_version" in data

    def test_metrics_total_predictions(self, client):
        resp = client.get("/metrics")
        data = resp.json()
        assert "total_predictions" in data


class TestInputBoundaryValidation:
    @pytest.mark.parametrize(
        "field,value",
        [
            ("on_time_rate", 0.0),
            ("on_time_rate", 1.0),
            ("defect_rate", 0.0),
            ("defect_rate", 1.0),
            ("financial_score", 0.0),
            ("geopolitical_risk", 1.0),
        ],
    )
    def test_boundary_values_accepted(self, client, supplier_payload, field, value):
        payload = dict(supplier_payload)
        payload[field] = value
        resp = client.post("/api/v1/predict/disruption", json=payload)
        assert resp.status_code == 200

    @pytest.mark.parametrize("lead_time", [1, 100, 365])
    def test_lead_time_boundary_values(self, client, supplier_payload, lead_time):
        payload = dict(supplier_payload)
        payload["lead_time_days"] = lead_time
        resp = client.post("/api/v1/predict/disruption", json=payload)
        assert resp.status_code == 200


class TestResponseSchemas:
    def test_predict_response_has_latency_ms(self, client, supplier_payload):
        resp = client.post("/api/v1/predict/disruption", json=supplier_payload)
        assert "latency_ms" in resp.json()

    def test_predict_response_latency_positive(self, client, supplier_payload):
        resp = client.post("/api/v1/predict/disruption", json=supplier_payload)
        assert resp.json()["latency_ms"] >= 0.0

    def test_reorder_response_has_all_fields(self, client, reorder_payload):
        resp = client.post("/api/v1/inventory/reorder-point", json=reorder_payload)
        data = resp.json()
        for field in ("reorder_point", "safety_stock", "lead_demand", "service_level", "z_score"):
            assert field in data, f"Missing field: {field}"

    def test_drift_scan_response_drifted_features_is_list(self, client):
        payload = {"features": {"lead_time_days": list(range(1, 51))}}
        resp = client.post("/api/v1/monitoring/drift", json=payload)
        assert isinstance(resp.json()["drifted_features"], list)

    def test_forecast_response_has_trend_slope(self, client):
        payload = {"history": [float(i) for i in range(1, 25)], "horizon": 3}
        resp = client.post("/api/v1/demand/forecast", json=payload)
        assert "trend_slope" in resp.json()

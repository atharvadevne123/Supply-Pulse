"""API endpoint tests for Supply-Pulse FastAPI app."""

from __future__ import annotations

import pytest


class TestHealthEndpoints:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_has_version(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "version" in data
        assert "model_version" in data

    def test_version_endpoint(self, client):
        resp = client.get("/version")
        assert resp.status_code == 200
        assert "app_version" in resp.json()

    def test_readyz_returns_ready(self, client):
        resp = client.get("/readyz")
        assert resp.status_code == 200

    def test_metrics_endpoint(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200


class TestDisruptionPredictionEndpoint:
    def test_predict_low_risk_supplier(self, client, supplier_payload):
        resp = client.post("/api/v1/predict/disruption", json=supplier_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "disruption_risk" in data
        assert 0.0 <= data["disruption_risk"] <= 1.0

    def test_predict_high_risk_supplier(self, client, high_risk_supplier_payload):
        resp = client.post("/api/v1/predict/disruption", json=high_risk_supplier_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["disruption_risk"] >= 0.0

    def test_predict_returns_label(self, client, supplier_payload):
        resp = client.post("/api/v1/predict/disruption", json=supplier_payload)
        assert resp.json()["disruption_label"] in ("LOW", "MEDIUM", "HIGH")

    def test_predict_returns_confidence(self, client, supplier_payload):
        resp = client.post("/api/v1/predict/disruption", json=supplier_payload)
        assert "confidence" in resp.json()

    def test_predict_invalid_on_time_rate(self, client, supplier_payload):
        payload = dict(supplier_payload)
        payload["on_time_rate"] = 1.5
        resp = client.post("/api/v1/predict/disruption", json=payload)
        assert resp.status_code == 422

    def test_predict_missing_required_field(self, client):
        resp = client.post("/api/v1/predict/disruption", json={"lead_time_days": 30})
        assert resp.status_code == 422

    def test_predict_negative_lead_time(self, client, supplier_payload):
        payload = dict(supplier_payload)
        payload["lead_time_days"] = -1
        resp = client.post("/api/v1/predict/disruption", json=payload)
        assert resp.status_code == 422

    @pytest.mark.parametrize("country", ["US", "CN", "DE", "IN", "RU"])
    def test_predict_various_countries(self, client, supplier_payload, country):
        payload = dict(supplier_payload)
        payload["country"] = country
        resp = client.post("/api/v1/predict/disruption", json=payload)
        assert resp.status_code == 200

    @pytest.mark.parametrize("category", ["electronics", "textile", "semiconductor", "logistics"])
    def test_predict_various_categories(self, client, supplier_payload, category):
        payload = dict(supplier_payload)
        payload["category"] = category
        resp = client.post("/api/v1/predict/disruption", json=payload)
        assert resp.status_code == 200

    def test_predict_sole_source_supplier(self, client, supplier_payload):
        payload = dict(supplier_payload)
        payload["is_sole_source"] = 1
        resp = client.post("/api/v1/predict/disruption", json=payload)
        assert resp.status_code == 200

    def test_predict_high_defect_rate(self, client, supplier_payload):
        payload = dict(supplier_payload)
        payload["defect_rate"] = 0.15
        resp = client.post("/api/v1/predict/disruption", json=payload)
        assert resp.status_code == 200


class TestReorderPointEndpoint:
    def test_reorder_basic(self, client, reorder_payload):
        resp = client.post("/api/v1/inventory/reorder-point", json=reorder_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "reorder_point" in data
        assert "safety_stock" in data

    def test_reorder_point_positive(self, client, reorder_payload):
        resp = client.post("/api/v1/inventory/reorder-point", json=reorder_payload)
        assert resp.json()["reorder_point"] > 0

    def test_reorder_returns_z_score(self, client, reorder_payload):
        resp = client.post("/api/v1/inventory/reorder-point", json=reorder_payload)
        assert "z_score" in resp.json()

    def test_reorder_zero_std_demand(self, client, reorder_payload):
        payload = dict(reorder_payload)
        payload["std_daily_demand"] = 0.0
        resp = client.post("/api/v1/inventory/reorder-point", json=payload)
        assert resp.status_code == 200
        assert resp.json()["safety_stock"] == 0.0

    @pytest.mark.parametrize(
        "service_level,expected_z",
        [
            (0.90, 1.28),
            (0.95, 1.645),
            (0.99, 2.326),
        ],
    )
    def test_reorder_service_levels(self, client, reorder_payload, service_level, expected_z):
        payload = dict(reorder_payload)
        payload["service_level"] = service_level
        resp = client.post("/api/v1/inventory/reorder-point", json=payload)
        assert resp.status_code == 200
        assert abs(resp.json()["z_score"] - expected_z) < 0.01

    def test_reorder_invalid_demand(self, client, reorder_payload):
        payload = dict(reorder_payload)
        payload["mean_daily_demand"] = -10
        resp = client.post("/api/v1/inventory/reorder-point", json=payload)
        assert resp.status_code == 422


class TestDriftEndpoint:
    def test_drift_scan_no_reference(self, client):
        payload = {"features": {"geopolitical_risk": [0.3, 0.4, 0.5, 0.6] * 10}}
        resp = client.post("/api/v1/monitoring/drift", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "drift_detected" in data

    def test_drift_scan_multiple_features(self, client):
        payload = {
            "features": {
                "geopolitical_risk": [0.3] * 50,
                "defect_rate": [0.05] * 50,
            }
        }
        resp = client.post("/api/v1/monitoring/drift", json=payload)
        assert resp.status_code == 200
        assert resp.json()["total_features_scanned"] == 2

    def test_monitoring_stats(self, client):
        resp = client.get("/api/v1/monitoring/stats")
        assert resp.status_code == 200


class TestCorrelationIDMiddleware:
    def test_response_has_correlation_id_header(self, client):
        resp = client.get("/health")
        assert "x-correlation-id" in resp.headers

    def test_supplied_correlation_id_is_echoed(self, client):
        cid = "test-corr-id-12345"
        resp = client.get("/health", headers={"X-Correlation-ID": cid})
        assert resp.headers.get("x-correlation-id") == cid

    def test_generated_correlation_id_is_uuid_format(self, client):
        import re
        resp = client.get("/health")
        cid = resp.headers.get("x-correlation-id", "")
        uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        assert re.match(uuid_pattern, cid)


class TestRateLimitMiddleware:
    def test_readyz_endpoint(self, client):
        resp = client.get("/readyz")
        assert resp.status_code == 200
        assert "ready" in resp.json()


class TestSimilarSuppliersEndpoint:
    def test_similar_suppliers_returns_200(self, client, supplier_payload):
        payload = {"supplier": supplier_payload, "top_k": 3}
        resp = client.post("/api/v1/suppliers/similar", json=payload)
        assert resp.status_code == 200

    def test_similar_suppliers_has_count(self, client, supplier_payload):
        payload = {"supplier": supplier_payload, "top_k": 3}
        resp = client.post("/api/v1/suppliers/similar", json=payload)
        data = resp.json()
        assert "count" in data
        assert "similar_suppliers" in data

    def test_demand_forecast_endpoint(self, client):
        payload = {"history": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0], "horizon": 3}
        resp = client.post("/api/v1/demand/forecast", json=payload)
        assert resp.status_code == 200
        assert "forecast" in resp.json()

    def test_demand_forecast_has_stats(self, client):
        payload = {"history": [100.0] * 12, "horizon": 6}
        resp = client.post("/api/v1/demand/forecast", json=payload)
        assert "stats" in resp.json()

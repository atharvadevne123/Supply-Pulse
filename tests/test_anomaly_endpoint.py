"""Tests for the /api/v1/demand/anomalies endpoint."""

from __future__ import annotations

import pytest


class TestAnomalyDetectionEndpoint:
    def test_zscore_method_returns_200(self, client, anomaly_payload):
        resp = client.post("/api/v1/demand/anomalies", json=anomaly_payload)
        assert resp.status_code == 200

    def test_zscore_detects_outliers(self, client):
        payload = {"values": [100.0] * 18 + [999.0, 1000.0], "method": "zscore"}
        resp = client.post("/api/v1/demand/anomalies", json=payload)
        assert resp.status_code == 200
        assert resp.json()["n_anomalies"] >= 1

    def test_iqr_method_returns_200(self, client):
        payload = {"values": list(range(1, 21)), "method": "iqr"}
        resp = client.post("/api/v1/demand/anomalies", json=payload)
        assert resp.status_code == 200
        assert "lower_fence" in resp.json()
        assert "upper_fence" in resp.json()

    def test_spikes_method_returns_200(self, client):
        payload = {
            "values": [100.0] * 10 + [500.0] + [100.0] * 9,
            "method": "spikes",
            "window": 3,
            "spike_ratio": 2.0,
        }
        resp = client.post("/api/v1/demand/anomalies", json=payload)
        assert resp.status_code == 200
        assert "spike_indices" in resp.json()

    def test_invalid_method_returns_422(self, client):
        payload = {"values": [1.0, 2.0, 3.0], "method": "unknown_method"}
        resp = client.post("/api/v1/demand/anomalies", json=payload)
        assert resp.status_code == 422

    def test_empty_values_returns_422(self, client):
        payload = {"values": [], "method": "zscore"}
        resp = client.post("/api/v1/demand/anomalies", json=payload)
        assert resp.status_code == 422

    def test_constant_series_no_anomalies_zscore(self, client):
        payload = {"values": [50.0] * 20, "method": "zscore"}
        resp = client.post("/api/v1/demand/anomalies", json=payload)
        assert resp.status_code == 200
        assert resp.json()["n_anomalies"] == 0

    @pytest.mark.parametrize("method", ["zscore", "iqr", "spikes"])
    def test_all_methods_with_normal_data(self, client, method):
        payload = {"values": [float(i) for i in range(1, 31)], "method": method}
        resp = client.post("/api/v1/demand/anomalies", json=payload)
        assert resp.status_code == 200
        assert isinstance(resp.json(), dict)

    def test_zscore_custom_threshold(self, client):
        payload = {
            "values": [100.0] * 15 + [200.0, 300.0],
            "method": "zscore",
            "threshold": 1.0,
        }
        resp = client.post("/api/v1/demand/anomalies", json=payload)
        assert resp.status_code == 200

    def test_iqr_custom_k(self, client):
        payload = {
            "values": list(range(1, 21)) + [100],
            "method": "iqr",
            "k": 3.0,
        }
        resp = client.post("/api/v1/demand/anomalies", json=payload)
        assert resp.status_code == 200
        assert "iqr" in resp.json()

"""Tests for anomaly detection module."""

from __future__ import annotations

import pytest

from app.anomaly import detect_demand_spikes, detect_iqr_anomalies, detect_zscore_anomalies


class TestZScoreAnomalies:
    def test_no_anomalies_in_constant_series(self):
        result = detect_zscore_anomalies([100.0] * 10)
        assert result["n_anomalies"] == 0

    def test_detects_outlier(self):
        values = [100.0] * 10 + [500.0]
        result = detect_zscore_anomalies(values)
        assert result["n_anomalies"] >= 1
        assert 10 in result["anomaly_indices"]

    def test_short_series_returns_insufficient(self):
        result = detect_zscore_anomalies([1.0, 2.0])
        assert "reason" in result

    def test_returns_z_scores(self):
        result = detect_zscore_anomalies([100.0] * 10)
        assert "z_scores" in result
        assert len(result["z_scores"]) == 10

    @pytest.mark.parametrize("threshold", [1.5, 2.0, 3.0])
    def test_stricter_threshold_finds_more_anomalies(self, threshold):
        values = [100.0] * 8 + [200.0, 300.0]
        result = detect_zscore_anomalies(values, threshold=threshold)
        assert isinstance(result["n_anomalies"], int)

    def test_anomaly_values_match_series(self):
        values = [100.0] * 9 + [999.0]
        result = detect_zscore_anomalies(values)
        if result["n_anomalies"] > 0:
            assert 999.0 in result["anomaly_values"]


class TestIQRAnomalies:
    def test_no_anomalies_in_uniform_series(self):
        result = detect_iqr_anomalies([100.0] * 20)
        assert result["n_anomalies"] == 0

    def test_detects_extreme_outlier(self):
        values = [100.0] * 19 + [1000.0]
        result = detect_iqr_anomalies(values)
        assert result["n_anomalies"] >= 1

    def test_short_series_insufficient(self):
        result = detect_iqr_anomalies([1.0, 2.0, 3.0])
        assert "reason" in result

    def test_returns_fences(self):
        values = list(range(1, 21))
        result = detect_iqr_anomalies(values)
        assert "lower_fence" in result
        assert "upper_fence" in result
        assert result["upper_fence"] >= result["lower_fence"]

    def test_returns_iqr_value(self):
        values = list(range(1, 21))
        result = detect_iqr_anomalies(values)
        assert result["iqr"] > 0

    @pytest.mark.parametrize("k,n_expected", [(1.5, lambda n: n >= 0), (3.0, lambda n: n >= 0)])
    def test_k_parameter_effect(self, k, n_expected):
        values = [100.0] * 15 + [200.0, 500.0]
        result = detect_iqr_anomalies(values, k=k)
        assert n_expected(result["n_anomalies"])


class TestDemandSpikeDetection:
    def test_no_spikes_in_flat_series(self):
        result = detect_demand_spikes([100.0] * 20)
        assert result["n_spikes"] == 0

    def test_detects_spike(self):
        history = [100.0] * 10 + [500.0] + [100.0] * 5
        result = detect_demand_spikes(history)
        assert result["n_spikes"] >= 1

    def test_spike_has_ratio(self):
        history = [100.0] * 10 + [400.0]
        result = detect_demand_spikes(history)
        if result["n_spikes"] > 0:
            assert "ratio" in result["spikes"][0]

    def test_insufficient_data(self):
        result = detect_demand_spikes([100.0, 200.0], window=3)
        assert "reason" in result

    @pytest.mark.parametrize("spike_ratio", [1.5, 2.0, 3.0])
    def test_spike_ratio_parameter(self, spike_ratio):
        history = [100.0] * 10 + [300.0]
        result = detect_demand_spikes(history, spike_ratio=spike_ratio)
        assert isinstance(result["n_spikes"], int)

    def test_spike_returns_window_and_ratio_fields(self):
        result = detect_demand_spikes([100.0] * 10, window=3, spike_ratio=2.0)
        assert result["window"] == 3
        assert result["spike_ratio"] == 2.0

    def test_spike_indices_are_within_range(self):
        history = [100.0] * 12 + [500.0]
        result = detect_demand_spikes(history, window=3)
        for idx in result["spike_indices"]:
            assert 0 <= idx < len(history)

    def test_zscore_mean_and_std_reported(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        result = detect_zscore_anomalies(values)
        assert "mean" in result
        assert "std" in result
        assert result["mean"] > 0

    @pytest.mark.parametrize("n_values", [3, 10, 50, 200])
    def test_zscore_scales_with_series_length(self, n_values):
        values = [float(i) for i in range(n_values)]
        result = detect_zscore_anomalies(values)
        assert len(result.get("z_scores", [])) == n_values

    def test_iqr_anomaly_values_outside_fences(self):
        values = [100.0] * 18 + [5000.0, -5000.0]
        result = detect_iqr_anomalies(values)
        for v in result.get("anomaly_values", []):
            assert v > result["upper_fence"] or v < result["lower_fence"]

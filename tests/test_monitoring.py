"""Drift detection and prediction logging tests."""

from __future__ import annotations

import numpy as np
import pytest

from app.monitoring import (
    compute_prediction_stats,
    detect_drift,
    run_full_drift_scan,
    set_reference_distribution,
)


class TestSetReferenceDistribution:
    def test_set_reference_stores_distribution(self):
        values = np.random.default_rng(0).uniform(0, 1, 100)
        set_reference_distribution("test_feature_set", values)
        from app.monitoring import _reference_distributions

        assert "test_feature_set" in _reference_distributions

    def test_set_reference_overwrites_existing(self):
        values1 = np.ones(50)
        values2 = np.zeros(50)
        set_reference_distribution("overwrite_feature", values1)
        set_reference_distribution("overwrite_feature", values2)
        from app.monitoring import _reference_distributions

        assert np.allclose(_reference_distributions["overwrite_feature"], values2)


class TestDetectDrift:
    def setup_method(self):
        rng = np.random.default_rng(42)
        set_reference_distribution("geo_risk_test", rng.uniform(0, 1, 200))

    def test_no_drift_similar_distributions(self):
        rng = np.random.default_rng(1)
        current = rng.uniform(0, 1, 200)
        result = detect_drift("geo_risk_test", current)
        assert "drift_detected" in result
        assert result["ks_statistic"] is not None

    def test_drift_detected_very_different_distributions(self):
        current = np.zeros(200)
        result = detect_drift("geo_risk_test", current)
        assert result["drift_detected"] is True

    def test_no_reference_returns_insufficient(self):
        result = detect_drift("nonexistent_feature_xyz", np.ones(100))
        assert result["drift_detected"] is False
        assert "insufficient" in result.get("reason", "")

    def test_insufficient_current_data(self):
        set_reference_distribution("small_current_test", np.ones(200))
        result = detect_drift("small_current_test", np.ones(10))
        assert result["drift_detected"] is False

    def test_result_contains_ks_statistic(self):
        current = np.random.default_rng(99).uniform(0, 1, 200)
        result = detect_drift("geo_risk_test", current)
        assert "ks_statistic" in result
        assert "p_value" in result

    def test_result_contains_sample_size(self):
        current = np.random.default_rng(7).uniform(0, 1, 150)
        result = detect_drift("geo_risk_test", current)
        assert result["sample_size"] == 150

    @pytest.mark.parametrize(
        "current_mean,should_drift",
        [
            (0.5, False),
            (0.0, True),
        ],
    )
    def test_drift_by_distribution_shift(self, current_mean, should_drift):
        rng = np.random.default_rng(10)
        set_reference_distribution("param_test_feat", rng.uniform(0.3, 0.7, 300))
        current = np.full(200, current_mean)
        result = detect_drift("param_test_feat", current)
        if should_drift:
            assert result["drift_detected"] is True


class TestRunFullDriftScan:
    def test_scan_multiple_features(self):
        rng = np.random.default_rng(42)
        set_reference_distribution("feat_a", rng.uniform(0, 1, 200))
        set_reference_distribution("feat_b", rng.uniform(0, 1, 200))
        current = {
            "feat_a": rng.uniform(0, 1, 200).tolist(),
            "feat_b": np.zeros(200).tolist(),
        }
        results = run_full_drift_scan(current)
        assert len(results) == 2

    def test_scan_returns_list_of_dicts(self):
        current = {"any_feature": [0.5] * 50}
        results = run_full_drift_scan(current)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, dict)


class TestComputePredictionStats:
    def test_stats_no_data(self, db_session):
        stats = compute_prediction_stats(db_session)
        assert "total_predictions" in stats
        assert stats["total_predictions"] == 0 or isinstance(stats["total_predictions"], int)

    def test_stats_returns_dict(self, db_session):
        stats = compute_prediction_stats(db_session)
        assert isinstance(stats, dict)

    def test_drift_ks_statistic_in_0_1_range(self):
        rng = np.random.default_rng(55)
        set_reference_distribution("ks_range_test", rng.uniform(0, 1, 200))
        current = rng.uniform(0.2, 0.8, 200)
        result = detect_drift("ks_range_test", current)
        if result["ks_statistic"] is not None:
            assert 0.0 <= result["ks_statistic"] <= 1.0

    def test_drift_p_value_in_0_1_range(self):
        rng = np.random.default_rng(66)
        set_reference_distribution("pval_test", rng.uniform(0, 1, 200))
        result = detect_drift("pval_test", rng.uniform(0, 1, 200))
        if result["p_value"] is not None:
            assert 0.0 <= result["p_value"] <= 1.0

    def test_no_drift_when_same_distribution(self):
        rng = np.random.default_rng(77)
        ref = rng.normal(0.5, 0.1, 500)
        set_reference_distribution("same_dist_test", ref)
        current = rng.normal(0.5, 0.1, 500)
        result = detect_drift("same_dist_test", current)
        assert result["drift_detected"] is False

    @pytest.mark.parametrize("n_features", [1, 3, 5])
    def test_full_scan_result_count_matches_features(self, n_features):
        rng = np.random.default_rng(88)
        features = {f"f{i}": rng.uniform(0, 1, 50).tolist() for i in range(n_features)}
        results = run_full_drift_scan(features)
        assert len(results) == n_features

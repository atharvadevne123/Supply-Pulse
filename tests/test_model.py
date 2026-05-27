"""Model training, prediction, and reorder point tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.model import (
    _risk_label,
    _train_default_model,
    build_model,
    compute_reorder_point,
    predict,
)


def _make_dataframe(n: int = 100, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "lead_time_days": rng.integers(5, 120, n),
            "on_time_rate": rng.uniform(0.5, 1.0, n),
            "defect_rate": rng.uniform(0.0, 0.15, n),
            "financial_score": rng.uniform(0.3, 1.0, n),
            "geopolitical_risk": rng.uniform(0.0, 1.0, n),
            "capacity_utilization": rng.uniform(0.3, 1.0, n),
            "years_active": rng.integers(1, 30, n),
            "is_sole_source": rng.integers(0, 2, n),
            "country": rng.choice(["US", "CN", "DE", "IN", "MX"], n),
            "category": rng.choice(["electronics", "textile", "logistics"], n),
        }
    )
    y = pd.Series(
        ((df["geopolitical_risk"] > 0.6).astype(int) | (df["defect_rate"] > 0.10).astype(int)).clip(
            0, 1
        )
    )
    return df, y


class TestModelBuilding:
    def test_build_model_returns_voting_classifier(self):
        from sklearn.ensemble import VotingClassifier

        model = build_model()
        assert isinstance(model, VotingClassifier)

    def test_build_model_has_rf_estimator(self):
        model = build_model()
        names = [name for name, _ in model.estimators]
        assert "rf" in names

    def test_default_model_trains_without_error(self):
        pipeline = _train_default_model()
        assert pipeline is not None

    def test_default_model_can_predict(self):
        pipeline = _train_default_model()
        sample = {
            "lead_time_days": 30,
            "on_time_rate": 0.9,
            "defect_rate": 0.02,
            "financial_score": 0.8,
            "geopolitical_risk": 0.3,
            "capacity_utilization": 0.6,
            "years_active": 5,
            "is_sole_source": 0,
            "country": "US",
            "category": "electronics",
        }
        result = predict(sample, pipeline)
        assert 0.0 <= result["disruption_risk"] <= 1.0


class TestPrediction:
    @pytest.fixture(scope="class")
    def trained_pipeline(self):
        return _train_default_model()

    def test_predict_returns_risk_score(self, trained_pipeline):
        sample = {
            "lead_time_days": 30,
            "on_time_rate": 0.9,
            "defect_rate": 0.01,
            "financial_score": 0.9,
            "geopolitical_risk": 0.2,
            "capacity_utilization": 0.5,
            "years_active": 10,
            "is_sole_source": 0,
            "country": "US",
            "category": "logistics",
        }
        result = predict(sample, trained_pipeline)
        assert "disruption_risk" in result
        assert 0.0 <= result["disruption_risk"] <= 1.0

    def test_predict_returns_label(self, trained_pipeline):
        sample = {
            "lead_time_days": 30,
            "on_time_rate": 0.9,
            "defect_rate": 0.01,
            "financial_score": 0.9,
            "geopolitical_risk": 0.2,
            "capacity_utilization": 0.5,
            "years_active": 10,
            "is_sole_source": 0,
            "country": "US",
            "category": "logistics",
        }
        result = predict(sample, trained_pipeline)
        assert result["disruption_label"] in ("LOW", "MEDIUM", "HIGH")

    def test_high_risk_supplier_gets_higher_score(self, trained_pipeline):
        low_risk = {
            "lead_time_days": 14,
            "on_time_rate": 0.98,
            "defect_rate": 0.005,
            "financial_score": 0.95,
            "geopolitical_risk": 0.1,
            "capacity_utilization": 0.4,
            "years_active": 20,
            "is_sole_source": 0,
            "country": "US",
            "category": "logistics",
        }
        high_risk = {
            "lead_time_days": 90,
            "on_time_rate": 0.55,
            "defect_rate": 0.14,
            "financial_score": 0.35,
            "geopolitical_risk": 0.92,
            "capacity_utilization": 0.98,
            "years_active": 1,
            "is_sole_source": 1,
            "country": "CN",
            "category": "semiconductor",
        }
        low_result = predict(low_risk, trained_pipeline)
        high_result = predict(high_risk, trained_pipeline)
        assert high_result["disruption_risk"] > low_result["disruption_risk"]

    @pytest.mark.parametrize("country", ["US", "CN", "DE", "IN", "RU"])
    def test_predict_country_variants(self, trained_pipeline, country):
        sample = {
            "lead_time_days": 30,
            "on_time_rate": 0.8,
            "defect_rate": 0.05,
            "financial_score": 0.7,
            "geopolitical_risk": 0.4,
            "capacity_utilization": 0.6,
            "years_active": 5,
            "is_sole_source": 0,
            "country": country,
            "category": "electronics",
        }
        result = predict(sample, trained_pipeline)
        assert 0.0 <= result["disruption_risk"] <= 1.0


class TestRiskLabel:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (0.0, "LOW"),
            (0.39, "LOW"),
            (0.40, "MEDIUM"),
            (0.69, "MEDIUM"),
            (0.70, "HIGH"),
            (1.0, "HIGH"),
        ],
    )
    def test_risk_label_boundaries(self, score, expected):
        assert _risk_label(score) == expected


class TestReorderPoint:
    @pytest.mark.parametrize(
        "mean,std,lead,z,expected_min",
        [
            (100.0, 20.0, 14, 1.645, 1400),
            (50.0, 5.0, 7, 1.28, 350),
            (200.0, 0.0, 30, 1.645, 6000),
        ],
    )
    def test_reorder_point_calculation(self, mean, std, lead, z, expected_min):
        result = compute_reorder_point(mean, std, lead, z)
        assert result["reorder_point"] >= expected_min

    def test_reorder_zero_std(self):
        result = compute_reorder_point(100.0, 0.0, 14, 1.645)
        assert result["safety_stock"] == 0.0

    def test_reorder_positive_values(self):
        result = compute_reorder_point(100.0, 20.0, 14, 1.645)
        assert result["reorder_point"] > 0
        assert result["safety_stock"] > 0
        assert result["lead_demand"] > 0

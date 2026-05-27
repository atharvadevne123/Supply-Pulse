"""Integration tests for the full ML pipeline end-to-end."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import build_feature_pipeline
from app.model import _train_default_model, predict, train


def _make_df(n: int = 50, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "lead_time_days": rng.integers(5, 90, n),
        "on_time_rate": rng.uniform(0.6, 1.0, n),
        "defect_rate": rng.uniform(0.0, 0.12, n),
        "financial_score": rng.uniform(0.4, 1.0, n),
        "geopolitical_risk": rng.uniform(0.0, 1.0, n),
        "capacity_utilization": rng.uniform(0.3, 0.9, n),
        "years_active": rng.integers(1, 25, n),
        "is_sole_source": rng.integers(0, 2, n),
        "country": rng.choice(["US", "CN", "DE"], n),
        "category": rng.choice(["electronics", "textile", "logistics"], n),
    })
    y = pd.Series(((df["geopolitical_risk"] > 0.6) | (df["defect_rate"] > 0.08)).astype(int))
    return df, y


class TestEndToEndPipeline:
    @pytest.fixture(scope="class")
    def pipeline(self):
        df, y = _make_df(n=200, seed=42)
        pipe, _ = train(df, y, cv_folds=2)
        return pipe

    def test_pipeline_predict_proba_shape(self, pipeline):
        df, _ = _make_df(n=10)
        result = pipeline.predict_proba(df)
        assert result.shape == (10, 2)

    def test_pipeline_predict_proba_sums_to_one(self, pipeline):
        df, _ = _make_df(n=5)
        probas = pipeline.predict_proba(df)
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_function_wrapper(self, pipeline):
        sample = {
            "lead_time_days": 30, "on_time_rate": 0.9, "defect_rate": 0.02,
            "financial_score": 0.8, "geopolitical_risk": 0.3,
            "capacity_utilization": 0.6, "years_active": 5,
            "is_sole_source": 0, "country": "US", "category": "electronics",
        }
        result = predict(sample, pipeline)
        assert 0.0 <= result["disruption_risk"] <= 1.0
        assert result["model_version"] == "1.0.0"

    def test_feature_pipeline_transforms_consistently(self):
        pipe = build_feature_pipeline()
        df1 = pd.DataFrame([{
            "lead_time_days": 30, "on_time_rate": 0.9, "defect_rate": 0.02,
            "financial_score": 0.8, "geopolitical_risk": 0.3,
            "capacity_utilization": 0.6, "years_active": 5,
            "is_sole_source": 0, "country": "US", "category": "electronics",
        }] * 3)
        out = pipe.fit_transform(df1)
        assert np.allclose(out[0], out[1]) and np.allclose(out[1], out[2])

    def test_high_and_low_risk_separated(self, pipeline):
        low = {"lead_time_days": 7, "on_time_rate": 0.99, "defect_rate": 0.001,
               "financial_score": 0.99, "geopolitical_risk": 0.05,
               "capacity_utilization": 0.30, "years_active": 20,
               "is_sole_source": 0, "country": "US", "category": "logistics"}
        high = {"lead_time_days": 120, "on_time_rate": 0.50, "defect_rate": 0.14,
                "financial_score": 0.30, "geopolitical_risk": 0.95,
                "capacity_utilization": 0.98, "years_active": 1,
                "is_sole_source": 1, "country": "CN", "category": "semiconductor"}
        low_risk = predict(low, pipeline)
        high_risk = predict(high, pipeline)
        assert high_risk["disruption_risk"] > low_risk["disruption_risk"]


class TestCVMetrics:
    def test_cv_auc_in_valid_range(self):
        df, y = _make_df(n=150, seed=5)
        _, metrics = train(df, y, cv_folds=3)
        assert 0.0 <= metrics["cv_auc_mean"] <= 1.0

    def test_cv_metrics_contain_expected_keys(self):
        df, y = _make_df(n=100, seed=7)
        _, metrics = train(df, y, cv_folds=2)
        assert "cv_auc_mean" in metrics
        assert "cv_auc_std" in metrics
        assert "n_samples" in metrics
        assert "n_features" in metrics

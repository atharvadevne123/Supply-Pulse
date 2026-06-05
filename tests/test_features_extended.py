"""Extended feature pipeline tests with edge cases and multi-row data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import (
    DropCategoricalTransformer,
    GeopoliticalRiskEncoder,
    build_feature_pipeline,
    compute_demand_features,
)


class TestDropCategoricalTransformer:
    def test_drops_country_and_category(self):
        df = pd.DataFrame({"country": ["US"], "category": ["electronics"], "value": [1.0]})
        t = DropCategoricalTransformer()
        out = t.fit_transform(df)
        assert "country" not in out.columns
        assert "category" not in out.columns

    def test_keeps_numeric_columns(self):
        df = pd.DataFrame({"country": ["US"], "category": ["electronics"], "value": [1.0]})
        t = DropCategoricalTransformer()
        out = t.fit_transform(df)
        assert "value" in out.columns

    def test_custom_cols_to_drop(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        t = DropCategoricalTransformer(cols=["a"])
        out = t.fit_transform(df)
        assert "a" not in out.columns
        assert "b" in out.columns


class TestGeopoliticalRiskEncoderEdgeCases:
    def test_handles_unknown_country(self):
        df = pd.DataFrame({"country": ["ZZ"]})
        enc = GeopoliticalRiskEncoder()
        out = enc.fit_transform(df)
        assert out["country_risk_score"].iloc[0] == 0.2

    def test_empty_dataframe(self):
        df = pd.DataFrame({"country": []})
        enc = GeopoliticalRiskEncoder()
        out = enc.fit_transform(df)
        assert len(out) == 0

    def test_multiple_rows(self):
        df = pd.DataFrame({"country": ["US", "CN", "DE", "RU"]})
        enc = GeopoliticalRiskEncoder()
        out = enc.fit_transform(df)
        assert len(out) == 4
        assert out["country_risk_score"].iloc[1] == 0.9


class TestPipelineConsistency:
    def test_pipeline_same_output_for_same_input(self):
        pipe = build_feature_pipeline()
        row = {
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
        df = pd.DataFrame([row] * 2)
        out = pipe.fit_transform(df)
        assert np.allclose(out[0], out[1])

    def test_pipeline_output_all_finite(self):
        pipe = build_feature_pipeline()
        rng = np.random.default_rng(0)
        n = 50
        df = pd.DataFrame(
            {
                "lead_time_days": rng.integers(5, 90, n),
                "on_time_rate": rng.uniform(0.5, 1.0, n),
                "defect_rate": rng.uniform(0.0, 0.15, n),
                "financial_score": rng.uniform(0.3, 1.0, n),
                "geopolitical_risk": rng.uniform(0.0, 1.0, n),
                "capacity_utilization": rng.uniform(0.3, 1.0, n),
                "years_active": rng.integers(1, 30, n),
                "is_sole_source": rng.integers(0, 2, n),
                "country": rng.choice(["US", "CN", "DE"], n),
                "category": rng.choice(["electronics", "logistics"], n),
            }
        )
        out = pipe.fit_transform(df)
        assert np.all(np.isfinite(out))


class TestComputeDemandFeaturesEdgeCases:
    def test_single_value(self):
        result = compute_demand_features([100.0])
        assert result["mean"] == 100.0
        assert result["std"] == 0.0

    def test_large_dataset(self):
        values = list(range(1, 1001))
        result = compute_demand_features(values)
        assert result["mean"] > 0

    @pytest.mark.parametrize(
        "values",
        [
            [0.0] * 10,
            [1e6] * 10,
            [0.001] * 10,
        ],
    )
    def test_extreme_values(self, values):
        result = compute_demand_features(values)
        assert result["mean"] >= 0


class TestGeopoliticalRiskCountries:
    @pytest.mark.parametrize("country,expected", [
        ("CN", 0.9),
        ("RU", 0.9),
        ("IR", 0.9),
        ("IN", 0.5),
        ("MX", 0.5),
        ("US", 0.2),
        ("DE", 0.2),
        ("UNKNOWN", 0.2),
    ])
    def test_country_risk_scores(self, country, expected):
        df = pd.DataFrame({"country": [country]})
        enc = GeopoliticalRiskEncoder()
        out = enc.fit_transform(df)
        assert out["country_risk_score"].iloc[0] == expected

    def test_high_risk_countries_above_low_risk(self):
        enc = GeopoliticalRiskEncoder()
        df_high = pd.DataFrame({"country": ["CN"]})
        df_low = pd.DataFrame({"country": ["US"]})
        high = enc.fit_transform(df_high)["country_risk_score"].iloc[0]
        low = enc.fit_transform(df_low)["country_risk_score"].iloc[0]
        assert high > low

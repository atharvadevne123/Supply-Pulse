"""Feature engineering pipeline tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.features import (
    CategoryRiskEncoder,
    CompositeRiskTransformer,
    GeopoliticalRiskEncoder,
    ReliabilityIndexTransformer,
    SupplyConcentrationTransformer,
    build_feature_pipeline,
    build_raw_dataframe,
    compute_demand_features,
)


def _base_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "lead_time_days": [30] * n,
        "on_time_rate": [0.9] * n,
        "defect_rate": [0.02] * n,
        "financial_score": [0.8] * n,
        "geopolitical_risk": [0.3] * n,
        "capacity_utilization": [0.6] * n,
        "years_active": [10] * n,
        "is_sole_source": [0] * n,
        "country": ["US"] * n,
        "category": ["electronics"] * n,
    })


class TestGeopoliticalRiskEncoder:
    def test_high_risk_country(self):
        df = pd.DataFrame({"country": ["CN"]})
        enc = GeopoliticalRiskEncoder()
        out = enc.fit_transform(df)
        assert out["country_risk_score"].iloc[0] == 0.9

    def test_medium_risk_country(self):
        df = pd.DataFrame({"country": ["IN"]})
        enc = GeopoliticalRiskEncoder()
        out = enc.fit_transform(df)
        assert out["country_risk_score"].iloc[0] == 0.5

    def test_low_risk_country(self):
        df = pd.DataFrame({"country": ["US"]})
        enc = GeopoliticalRiskEncoder()
        out = enc.fit_transform(df)
        assert out["country_risk_score"].iloc[0] == 0.2

    @pytest.mark.parametrize("country,expected", [
        ("RU", 0.9), ("KP", 0.9), ("MX", 0.5), ("DE", 0.2), ("GB", 0.2),
    ])
    def test_country_risk_scores(self, country, expected):
        df = pd.DataFrame({"country": [country]})
        enc = GeopoliticalRiskEncoder()
        out = enc.fit_transform(df)
        assert out["country_risk_score"].iloc[0] == expected


class TestCategoryRiskEncoder:
    @pytest.mark.parametrize("category,expected", [
        ("semiconductor", 0.85),
        ("rare_earth", 0.90),
        ("textile", 0.40),
        ("logistics", 0.45),
    ])
    def test_category_scores(self, category, expected):
        df = pd.DataFrame({"category": [category]})
        enc = CategoryRiskEncoder()
        out = enc.fit_transform(df)
        assert out["category_risk_score"].iloc[0] == expected

    def test_unknown_category_defaults_to_0_5(self):
        df = pd.DataFrame({"category": ["unknown_widget"]})
        enc = CategoryRiskEncoder()
        out = enc.fit_transform(df)
        assert out["category_risk_score"].iloc[0] == 0.50


class TestCompositeRiskTransformer:
    def test_composite_risk_in_range(self):
        df = _base_df()
        df = GeopoliticalRiskEncoder().fit_transform(df)
        df = CategoryRiskEncoder().fit_transform(df)
        df = CompositeRiskTransformer().fit_transform(df)
        assert (df["composite_risk"].between(0, 1)).all()

    def test_high_risk_inputs_give_higher_composite(self):
        low_df = pd.DataFrame({
            "geopolitical_risk": [0.1], "country_risk_score": [0.2],
            "category_risk_score": [0.4], "financial_score": [0.95], "defect_rate": [0.01],
        })
        high_df = pd.DataFrame({
            "geopolitical_risk": [0.9], "country_risk_score": [0.9],
            "category_risk_score": [0.9], "financial_score": [0.3], "defect_rate": [0.15],
        })
        t = CompositeRiskTransformer()
        low_risk = t.fit_transform(low_df)["composite_risk"].iloc[0]
        high_risk = t.fit_transform(high_df)["composite_risk"].iloc[0]
        assert high_risk > low_risk


class TestReliabilityIndexTransformer:
    def test_reliability_in_range(self):
        df = _base_df(10)
        out = ReliabilityIndexTransformer().fit_transform(df)
        assert (out["reliability_index"].between(0, 1)).all()

    def test_high_on_time_yields_higher_reliability(self):
        df_high = _base_df(1)
        df_high["on_time_rate"] = 0.99
        df_low = _base_df(1)
        df_low["on_time_rate"] = 0.55
        t = ReliabilityIndexTransformer()
        high = t.fit_transform(df_high)["reliability_index"].iloc[0]
        low = t.fit_transform(df_low)["reliability_index"].iloc[0]
        assert high > low


class TestSupplyConcentrationTransformer:
    def test_sole_source_increases_concentration(self):
        df_sole = _base_df(1)
        df_sole["is_sole_source"] = 1
        df_multi = _base_df(1)
        df_multi["is_sole_source"] = 0
        t = SupplyConcentrationTransformer()
        sole = t.fit_transform(df_sole)["supply_concentration"].iloc[0]
        multi = t.fit_transform(df_multi)["supply_concentration"].iloc[0]
        assert sole >= multi


class TestBuildFeaturePipeline:
    def test_pipeline_transforms_dataframe(self):
        pipe = build_feature_pipeline()
        df = _base_df(10)
        out = pipe.fit_transform(df)
        assert out.shape[0] == 10

    def test_pipeline_output_is_numeric(self):
        pipe = build_feature_pipeline()
        df = _base_df(5)
        out = pipe.fit_transform(df)
        assert np.issubdtype(type(out[0, 0]), np.floating)

    def test_build_raw_dataframe(self):
        data = {"lead_time_days": 30, "on_time_rate": 0.9}
        df = build_raw_dataframe(data)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1


class TestComputeDemandFeatures:
    def test_empty_demands(self):
        result = compute_demand_features([])
        assert result["mean"] == 0.0

    def test_constant_demand(self):
        result = compute_demand_features([100] * 10)
        assert result["mean"] == 100.0
        assert result["std"] == 0.0

    def test_trend_detection(self):
        increasing = list(range(1, 21))
        result = compute_demand_features(increasing)
        assert result["trend"] > 0

    @pytest.mark.parametrize("demands,expected_max", [
        ([10, 20, 30], 30),
        ([5, 5, 5], 5),
        ([1, 100, 50], 100),
    ])
    def test_max_demand(self, demands, expected_max):
        result = compute_demand_features(demands)
        assert result["max"] == float(expected_max)

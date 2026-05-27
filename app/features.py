"""Feature engineering pipeline for supply chain disruption prediction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "lead_time_days",
    "on_time_rate",
    "defect_rate",
    "financial_score",
    "geopolitical_risk",
    "capacity_utilization",
    "years_active",
    "is_sole_source",
]

CATEGORICAL_FEATURES = ["country", "category"]

ALL_FEATURES = NUMERIC_FEATURES + [
    "country_risk_score",
    "category_risk_score",
    "composite_risk",
    "reliability_index",
    "supply_concentration",
]


class GeopoliticalRiskEncoder(BaseEstimator, TransformerMixin):
    """Map country to a continuous geopolitical risk score."""

    HIGH_RISK_COUNTRIES = {"CN", "RU", "IR", "KP", "MM", "BY"}
    MED_RISK_COUNTRIES = {"IN", "MX", "TR", "EG", "NG", "PK", "BD", "VN"}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "GeopoliticalRiskEncoder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        country_col = X.get("country", pd.Series(["US"] * len(X)))
        X["country_risk_score"] = country_col.apply(self._score)
        return X

    def _score(self, country: str) -> float:
        if country in self.HIGH_RISK_COUNTRIES:
            return 0.9
        if country in self.MED_RISK_COUNTRIES:
            return 0.5
        return 0.2


class CategoryRiskEncoder(BaseEstimator, TransformerMixin):
    """Map supplier category to a risk weight."""

    RISK_MAP = {
        "semiconductor": 0.85,
        "rare_earth": 0.90,
        "pharmaceutical": 0.75,
        "electronics": 0.70,
        "automotive": 0.60,
        "textile": 0.40,
        "food": 0.35,
        "logistics": 0.45,
    }

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CategoryRiskEncoder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cat_col = X.get("category", pd.Series(["logistics"] * len(X)))
        X["category_risk_score"] = cat_col.apply(lambda c: self.RISK_MAP.get(str(c).lower(), 0.50))
        return X


class CompositeRiskTransformer(BaseEstimator, TransformerMixin):
    """Combine individual risk signals into a composite score."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CompositeRiskTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["composite_risk"] = (
            0.30 * X.get("geopolitical_risk", 0)
            + 0.25 * X.get("country_risk_score", 0)
            + 0.20 * X.get("category_risk_score", 0)
            + 0.15 * (1 - X.get("financial_score", 1))
            + 0.10 * X.get("defect_rate", 0)
        ).clip(0, 1)
        return X


class ReliabilityIndexTransformer(BaseEstimator, TransformerMixin):
    """Compute a supplier reliability index from performance metrics."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ReliabilityIndexTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        on_time = X.get("on_time_rate", 1.0)
        defect = X.get("defect_rate", 0.0)
        years = np.log1p(X.get("years_active", 1))
        capacity = X.get("capacity_utilization", 0.5)
        X["reliability_index"] = (
            0.40 * on_time
            + 0.30 * (1 - defect)
            + 0.15 * (years / years.max() if years.max() > 0 else years)
            + 0.15 * (1 - capacity.clip(0, 1))
        ).clip(0, 1)
        return X


class SupplyConcentrationTransformer(BaseEstimator, TransformerMixin):
    """Flag high supply concentration risk for sole-source suppliers."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "SupplyConcentrationTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        sole = X.get("is_sole_source", pd.Series([0] * len(X))).astype(float)
        lead = X.get("lead_time_days", pd.Series([30] * len(X)))
        X["supply_concentration"] = sole * (lead / lead.clip(lower=1).max())
        return X


class DropCategoricalTransformer(BaseEstimator, TransformerMixin):
    """Remove string columns before scaling."""

    def __init__(self, cols: list[str] | None = None) -> None:
        self.cols = cols or CATEGORICAL_FEATURES

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DropCategoricalTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.cols if c in X.columns], errors="ignore")


def build_feature_pipeline() -> Pipeline:
    """Return a 7-stage sklearn Pipeline that produces numeric feature matrix."""
    return Pipeline(
        [
            ("geo_risk", GeopoliticalRiskEncoder()),
            ("cat_risk", CategoryRiskEncoder()),
            ("composite", CompositeRiskTransformer()),
            ("reliability", ReliabilityIndexTransformer()),
            ("concentration", SupplyConcentrationTransformer()),
            ("drop_cat", DropCategoricalTransformer()),
            ("scaler", StandardScaler()),
        ]
    )


def build_raw_dataframe(supplier_data: dict[str, Any]) -> pd.DataFrame:
    """Convert a single supplier dict into a one-row DataFrame."""
    return pd.DataFrame([supplier_data])


def compute_demand_features(demands: list[float]) -> dict[str, float]:
    """Extract statistical demand features from a history sequence."""
    arr = np.array(demands, dtype=float)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "cv": 0.0, "trend": 0.0, "max": 0.0}
    mean = float(arr.mean())
    std = float(arr.std()) if len(arr) > 1 else 0.0
    cv = std / mean if mean > 0 else 0.0
    trend = float(np.polyfit(np.arange(len(arr)), arr, 1)[0]) if len(arr) > 1 else 0.0
    return {"mean": mean, "std": std, "cv": cv, "trend": trend, "max": float(arr.max())}

"""XGBoost + LightGBM + RandomForest ensemble with 5-fold CV for supply chain disruption."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from app.features import build_feature_pipeline

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / "model_artifacts" / "ensemble.pkl"
MODEL_VERSION = "1.0.0"

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("xgboost not installed - using RF-only ensemble")

try:
    import lightgbm as lgb

    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    logger.warning("lightgbm not installed - using RF-only ensemble")


def _build_estimators() -> list[tuple[str, Any]]:
    """Return list of (name, estimator) tuples for VotingClassifier."""
    estimators: list[tuple[str, Any]] = []
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    estimators.append(("rf", rf))
    if XGB_AVAILABLE:
        xgb_clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=3,
            eval_metric="auc", random_state=42, n_jobs=-1,
        )
        estimators.append(("xgb", xgb_clf))
    if LGB_AVAILABLE:
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1,
        )
        estimators.append(("lgb", lgb_clf))
    return estimators


def build_model() -> VotingClassifier:
    """Instantiate the soft-voting ensemble."""
    return VotingClassifier(estimators=_build_estimators(), voting="soft", n_jobs=-1)


def train(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
) -> tuple[Pipeline, dict[str, float]]:
    """Train feature pipeline + ensemble; return fitted pipeline and CV metrics."""
    feature_pipe = build_feature_pipeline()
    X_transformed = feature_pipe.fit_transform(X)

    model = build_model()
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    aucs: list[float] = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_transformed, y)):
        X_tr, X_val = X_transformed[train_idx], X_transformed[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        fold_model = build_model()
        fold_model.fit(X_tr, y_tr)
        proba = fold_model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, proba)
        aucs.append(fold_auc)
        logger.info("Fold %d AUC: %.4f", fold + 1, fold_auc)

    model.fit(X_transformed, y)

    full_pipeline = Pipeline([("features", feature_pipe), ("model", model)])
    metrics = {
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std": float(np.std(aucs)),
        "cv_folds": cv_folds,
        "n_samples": len(y),
        "n_features": X_transformed.shape[1],
        "disruption_rate": float(y.mean()),
    }
    logger.info("Training complete. CV AUC: %.4f ± %.4f", metrics["cv_auc_mean"], metrics["cv_auc_std"])
    return full_pipeline, metrics


def save_model(pipeline: Pipeline) -> None:
    """Persist trained pipeline to disk."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info("Model saved to %s", MODEL_PATH)


def load_model() -> Pipeline:
    """Load trained pipeline from disk, training a default if missing."""
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    logger.warning("No saved model found - training default model on synthetic data")
    return _train_default_model()


def _train_default_model() -> Pipeline:
    """Train a minimal model on synthetic data for cold-start scenarios."""
    rng = np.random.default_rng(42)
    n = 800
    df = pd.DataFrame({
        "lead_time_days": rng.integers(5, 120, n),
        "on_time_rate": rng.uniform(0.5, 1.0, n),
        "defect_rate": rng.uniform(0.0, 0.15, n),
        "financial_score": rng.uniform(0.3, 1.0, n),
        "geopolitical_risk": rng.uniform(0.0, 1.0, n),
        "capacity_utilization": rng.uniform(0.3, 1.0, n),
        "years_active": rng.integers(1, 30, n),
        "is_sole_source": rng.integers(0, 2, n),
        "country": rng.choice(["US", "CN", "DE", "IN", "MX"], n),
        "category": rng.choice(["electronics", "textile", "logistics", "semiconductor"], n),
    })
    y = pd.Series((
        (df["geopolitical_risk"] > 0.6).astype(int)
        | (df["defect_rate"] > 0.10).astype(int)
        | (df["on_time_rate"] < 0.70).astype(int)
    ).clip(0, 1))
    pipeline, _ = train(df, y)
    save_model(pipeline)
    return pipeline


def predict(supplier_data: dict[str, Any], pipeline: Pipeline | None = None) -> dict[str, Any]:
    """Return disruption risk score for a single supplier record."""
    if pipeline is None:
        pipeline = load_model()
    df = pd.DataFrame([supplier_data])
    proba = pipeline.predict_proba(df)[0]
    risk_score = float(proba[1])
    return {
        "disruption_risk": round(risk_score, 4),
        "disruption_label": _risk_label(risk_score),
        "confidence": round(float(max(proba)), 4),
        "model_version": MODEL_VERSION,
    }


def _risk_label(score: float) -> str:
    if score >= 0.70:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    return "LOW"


def compute_reorder_point(
    mean_daily_demand: float,
    std_daily_demand: float,
    lead_time_days: int,
    service_level_z: float = 1.645,
) -> dict[str, float]:
    """Calculate EOQ-based reorder point with safety stock."""
    lead_demand = mean_daily_demand * lead_time_days
    safety_stock = service_level_z * std_daily_demand * (lead_time_days ** 0.5)
    reorder_point = lead_demand + safety_stock
    return {
        "reorder_point": round(reorder_point, 2),
        "safety_stock": round(safety_stock, 2),
        "lead_demand": round(lead_demand, 2),
    }

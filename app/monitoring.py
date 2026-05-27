"""KS-test drift detection and prediction logging for Supply-Pulse."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from scipy import stats
from sqlalchemy.orm import Session

from app.database import DriftLog, PredictionLog

logger = logging.getLogger(__name__)

DRIFT_P_VALUE_THRESHOLD = 0.05
DRIFT_MIN_SAMPLE_SIZE = 30

_reference_distributions: dict[str, np.ndarray] = {}


def set_reference_distribution(feature_name: str, values: np.ndarray) -> None:
    """Store reference distribution for a feature column."""
    _reference_distributions[feature_name] = np.array(values, dtype=float)
    logger.info("Reference distribution set for '%s' (%d samples)", feature_name, len(values))


def detect_drift(
    feature_name: str,
    current_values: np.ndarray,
    db: Session | None = None,
) -> dict[str, Any]:
    """Run KS-test between reference and current distributions."""
    current = np.array(current_values, dtype=float)
    reference = _reference_distributions.get(feature_name)

    if reference is None or len(reference) < DRIFT_MIN_SAMPLE_SIZE:
        return {
            "feature": feature_name,
            "drift_detected": False,
            "ks_statistic": None,
            "p_value": None,
            "reason": "insufficient_reference_data",
        }

    if len(current) < DRIFT_MIN_SAMPLE_SIZE:
        return {
            "feature": feature_name,
            "drift_detected": False,
            "ks_statistic": None,
            "p_value": None,
            "reason": "insufficient_current_data",
        }

    ks_stat, p_value = stats.ks_2samp(reference, current)
    drift_detected = bool(p_value < DRIFT_P_VALUE_THRESHOLD)

    if db is not None:
        try:
            log_entry = DriftLog(
                feature_name=feature_name,
                ks_statistic=float(ks_stat),
                p_value=float(p_value),
                drift_detected=drift_detected,
                sample_size=len(current),
                notes=f"Reference n={len(reference)}, current n={len(current)}",
            )
            db.add(log_entry)
            db.commit()
        except Exception:
            logger.exception("Failed to persist drift log for '%s'", feature_name)
            db.rollback()

    if drift_detected:
        logger.warning("Drift detected on '%s': KS=%.4f, p=%.4f", feature_name, ks_stat, p_value)

    return {
        "feature": feature_name,
        "drift_detected": drift_detected,
        "ks_statistic": round(float(ks_stat), 6),
        "p_value": round(float(p_value), 6),
        "sample_size": len(current),
    }


def run_full_drift_scan(
    current_data: dict[str, list[float]],
    db: Session | None = None,
) -> list[dict[str, Any]]:
    """Scan all supplied feature columns for distribution drift."""
    results = []
    for feature_name, values in current_data.items():
        result = detect_drift(feature_name, np.array(values), db=db)
        results.append(result)
    drifted = [r for r in results if r.get("drift_detected")]
    logger.info("Drift scan complete: %d/%d features drifted", len(drifted), len(results))
    return results


def log_prediction(
    prediction_type: str,
    input_data: dict[str, Any],
    prediction: float,
    confidence: float | None,
    model_version: str,
    latency_ms: float | None,
    db: Session | None = None,
) -> None:
    """Persist a prediction record to the database."""
    if db is None:
        return
    try:
        entry = PredictionLog(
            prediction_type=prediction_type,
            input_data=input_data,
            prediction=prediction,
            confidence=confidence,
            model_version=model_version,
            latency_ms=latency_ms,
        )
        db.add(entry)
        db.commit()
    except Exception:
        logger.exception("Failed to log prediction")
        db.rollback()


def compute_prediction_stats(db: Session) -> dict[str, Any]:
    """Aggregate prediction statistics from the log table."""
    try:
        total = db.query(PredictionLog).count()
        if total == 0:
            return {"total_predictions": 0, "avg_prediction": None, "high_risk_count": 0}
        rows = db.query(PredictionLog.prediction).all()
        predictions = [r.prediction for r in rows]
        high_risk = sum(1 for p in predictions if p >= 0.70)
        return {
            "total_predictions": total,
            "avg_prediction": round(float(np.mean(predictions)), 4),
            "std_prediction": round(float(np.std(predictions)), 4),
            "high_risk_count": high_risk,
            "high_risk_pct": round(high_risk / total * 100, 2),
        }
    except Exception:
        logger.exception("Error computing prediction stats")
        return {"total_predictions": 0, "error": "stats_unavailable"}


def timed(func: Any) -> Any:
    """Decorator that injects latency_ms into the wrapped function result."""
    import functools

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if isinstance(result, dict):
            result["latency_ms"] = round(elapsed_ms, 2)
        return result

    return wrapper

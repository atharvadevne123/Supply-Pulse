"""Generate structured risk report summaries for supplier assessments."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "RISK_THRESHOLDS",
    "classify_risk",
    "generate_recommendations",
    "build_risk_report",
]

RISK_THRESHOLDS = {
    "CRITICAL": 0.80,
    "HIGH": 0.60,
    "MEDIUM": 0.40,
    "LOW": 0.0,
}


def classify_risk(score: float) -> str:
    """Classify a risk score into severity bucket.

    Args:
        score: Disruption risk score in [0, 1].

    Returns:
        Risk severity label.
    """
    for label, threshold in RISK_THRESHOLDS.items():
        if score >= threshold:
            return label
    return "LOW"


def generate_recommendations(
    disruption_risk: float,
    scorecard: dict[str, Any],
) -> list[str]:
    """Generate actionable recommendations based on risk profile.

    Args:
        disruption_risk: ML-predicted disruption probability.
        scorecard: Component scorecard from supplier_scorer.

    Returns:
        List of recommendation strings.
    """
    recs: list[str] = []
    components = scorecard.get("components", {})

    if disruption_risk >= 0.70:
        recs.append("Identify and qualify alternative suppliers immediately")
    if disruption_risk >= 0.40:
        recs.append("Increase safety stock by 20-30% for affected SKUs")

    if components.get("delivery", 1.0) < 0.6:
        recs.append("Negotiate delivery SLA with penalty clauses")
    if components.get("quality", 1.0) < 0.6:
        recs.append("Implement incoming quality inspection programme")
    if components.get("financial", 1.0) < 0.5:
        recs.append("Request supplier financial statements and monitor quarterly")
    if components.get("geopolitical", 1.0) < 0.5:
        recs.append("Develop dual-source strategy for geopolitically sensitive suppliers")
    if components.get("capacity", 1.0) < 0.3:
        recs.append("Reserve capacity with advance purchase orders or capacity reservations")

    if not recs:
        recs.append("Continue standard monitoring cadence")

    return recs


def build_risk_report(
    supplier_id: str,
    supplier_name: str,
    disruption_result: dict[str, Any],
    scorecard: dict[str, Any],
) -> dict[str, Any]:
    """Assemble a comprehensive supplier risk report.

    Args:
        supplier_id: Unique identifier for the supplier.
        supplier_name: Human-readable supplier name.
        disruption_result: Output from model.predict().
        scorecard: Output from supplier_scorer.compute_scorecard().

    Returns:
        Structured risk report dictionary.
    """
    disruption_risk = disruption_result.get("disruption_risk", 0.0)
    severity = classify_risk(disruption_risk)
    recommendations = generate_recommendations(disruption_risk, scorecard)

    report = {
        "supplier_id": supplier_id,
        "supplier_name": supplier_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "disruption_risk": disruption_risk,
        "disruption_label": disruption_result.get("disruption_label", "UNKNOWN"),
        "severity": severity,
        "grade": scorecard.get("grade", "N/A"),
        "total_score": scorecard.get("total_score", 0.0),
        "components": scorecard.get("components", {}),
        "recommendations": recommendations,
        "n_recommendations": len(recommendations),
    }
    logger.info(
        "Risk report for '%s': severity=%s, risk=%.4f, grade=%s",
        supplier_name,
        severity,
        disruption_risk,
        scorecard.get("grade", "N/A"),
    )
    return report

"""Multi-dimensional supplier risk scorecard with weighted component scores."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

SCORE_WEIGHTS = {
    "delivery": 0.30,
    "quality": 0.25,
    "financial": 0.20,
    "geopolitical": 0.15,
    "capacity": 0.10,
}


def score_delivery(on_time_rate: float, lead_time_days: int) -> float:
    """Score supplier delivery reliability (0=worst, 1=best).

    Args:
        on_time_rate: Historical on-time delivery rate (0-1).
        lead_time_days: Typical lead time in calendar days.

    Returns:
        Delivery score in [0, 1].
    """
    lead_penalty = max(0.0, (lead_time_days - 14) / 100.0)
    return float(max(0.0, min(1.0, on_time_rate - lead_penalty)))


def score_quality(defect_rate: float, years_active: int) -> float:
    """Score supplier quality based on defect rate and track record.

    Args:
        defect_rate: Product defect rate (0-1).
        years_active: Years the supplier has been operating.

    Returns:
        Quality score in [0, 1].
    """
    maturity_bonus = min(0.1, years_active / 100.0)
    return float(max(0.0, min(1.0, 1.0 - defect_rate * 8 + maturity_bonus)))


def score_financial(financial_score: float) -> float:
    """Pass-through financial health score.

    Args:
        financial_score: Pre-computed financial health index (0-1).

    Returns:
        Financial score unchanged.
    """
    return float(max(0.0, min(1.0, financial_score)))


def score_geopolitical(geopolitical_risk: float, is_sole_source: int) -> float:
    """Score geopolitical exposure with sole-source penalty.

    Args:
        geopolitical_risk: Geopolitical risk index (0-1, higher=riskier).
        is_sole_source: 1 if this is the only supplier for this category.

    Returns:
        Geopolitical resilience score in [0, 1].
    """
    base = 1.0 - geopolitical_risk
    sole_penalty = 0.15 if is_sole_source else 0.0
    return float(max(0.0, min(1.0, base - sole_penalty)))


def score_capacity(capacity_utilization: float) -> float:
    """Score capacity slack — higher utilization reduces flexibility.

    Args:
        capacity_utilization: Current capacity utilization ratio (0-1).

    Returns:
        Capacity score in [0, 1].
    """
    return float(max(0.0, 1.0 - capacity_utilization))


def compute_scorecard(supplier: dict[str, Any]) -> dict[str, Any]:
    """Compute a full multi-dimensional scorecard for a supplier.

    Args:
        supplier: Dictionary of supplier attributes.

    Returns:
        Dictionary with component scores, weighted total, and grade.
    """
    delivery = score_delivery(
        supplier.get("on_time_rate", 0.9),
        supplier.get("lead_time_days", 30),
    )
    quality = score_quality(
        supplier.get("defect_rate", 0.02),
        supplier.get("years_active", 5),
    )
    financial = score_financial(supplier.get("financial_score", 0.8))
    geopolitical = score_geopolitical(
        supplier.get("geopolitical_risk", 0.3),
        supplier.get("is_sole_source", 0),
    )
    capacity = score_capacity(supplier.get("capacity_utilization", 0.6))

    components = {
        "delivery": round(delivery, 4),
        "quality": round(quality, 4),
        "financial": round(financial, 4),
        "geopolitical": round(geopolitical, 4),
        "capacity": round(capacity, 4),
    }

    total = sum(SCORE_WEIGHTS[k] * v for k, v in components.items())
    grade = _grade(total)

    logger.debug("Supplier scorecard: total=%.4f grade=%s", total, grade)

    return {
        "components": components,
        "total_score": round(total, 4),
        "grade": grade,
        "weights": SCORE_WEIGHTS,
    }


def _grade(score: float) -> str:
    """Convert a total score to a letter grade."""
    if score >= 0.85:
        return "A"
    if score >= 0.70:
        return "B"
    if score >= 0.55:
        return "C"
    if score >= 0.40:
        return "D"
    return "F"

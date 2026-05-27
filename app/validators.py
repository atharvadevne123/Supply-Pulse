"""Input validation utilities for Supply-Pulse API endpoints."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

VALID_COUNTRIES = {
    "US",
    "CN",
    "DE",
    "IN",
    "MX",
    "JP",
    "GB",
    "KR",
    "VN",
    "BR",
    "FR",
    "IT",
    "ES",
    "CA",
    "AU",
    "RU",
    "TR",
    "IR",
    "KP",
    "MM",
    "BY",
    "PK",
    "BD",
    "EG",
    "NG",
}

VALID_CATEGORIES = {
    "electronics",
    "semiconductor",
    "rare_earth",
    "pharmaceutical",
    "automotive",
    "textile",
    "food",
    "logistics",
}


def validate_supplier_input(data: dict[str, Any]) -> list[str]:
    """Return list of validation error messages for a supplier dict.

    Args:
        data: Raw supplier attribute dictionary.

    Returns:
        List of error strings (empty if valid).
    """
    errors: list[str] = []
    if not (0.0 <= data.get("on_time_rate", 0.5) <= 1.0):
        errors.append("on_time_rate must be between 0.0 and 1.0")
    if not (0.0 <= data.get("defect_rate", 0.0) <= 1.0):
        errors.append("defect_rate must be between 0.0 and 1.0")
    if not (0.0 <= data.get("financial_score", 0.5) <= 1.0):
        errors.append("financial_score must be between 0.0 and 1.0")
    if not (0.0 <= data.get("geopolitical_risk", 0.3) <= 1.0):
        errors.append("geopolitical_risk must be between 0.0 and 1.0")
    if not (0.0 <= data.get("capacity_utilization", 0.5) <= 1.0):
        errors.append("capacity_utilization must be between 0.0 and 1.0")
    if data.get("lead_time_days", 30) < 1:
        errors.append("lead_time_days must be >= 1")
    if data.get("years_active", 1) < 0:
        errors.append("years_active must be >= 0")
    return errors


def validate_demand_history(history: list[float]) -> list[str]:
    """Validate demand history list for forecasting.

    Args:
        history: Historical demand values.

    Returns:
        List of error strings (empty if valid).
    """
    errors: list[str] = []
    if not history:
        errors.append("history must not be empty")
        return errors
    if any(v < 0 for v in history):
        errors.append("All demand values must be non-negative")
    if len(history) > 10000:
        errors.append("history length must not exceed 10000")
    return errors


def sanitize_supplier_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize and sanitize supplier field values.

    Args:
        data: Raw supplier dict (may contain None or out-of-range values).

    Returns:
        Sanitized copy with defaults applied.
    """
    safe = dict(data)
    safe["on_time_rate"] = float(max(0.0, min(1.0, safe.get("on_time_rate") or 0.9)))
    safe["defect_rate"] = float(max(0.0, min(1.0, safe.get("defect_rate") or 0.02)))
    safe["financial_score"] = float(max(0.0, min(1.0, safe.get("financial_score") or 0.8)))
    safe["geopolitical_risk"] = float(max(0.0, min(1.0, safe.get("geopolitical_risk") or 0.3)))
    safe["capacity_utilization"] = float(
        max(0.0, min(1.0, safe.get("capacity_utilization") or 0.6))
    )
    safe["lead_time_days"] = max(1, int(safe.get("lead_time_days") or 30))
    safe["years_active"] = max(0, int(safe.get("years_active") or 5))
    safe["is_sole_source"] = int(bool(safe.get("is_sole_source", 0)))
    safe["country"] = str(safe.get("country") or "US")[:50]
    safe["category"] = str(safe.get("category") or "logistics")[:100]
    return safe

"""Time-series demand forecasting with trend and seasonality decomposition."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "forecast_demand",
    "compute_demand_statistics",
]


def _simple_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Compute simple moving average with valid-mode convolution."""
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def _linear_trend(values: np.ndarray) -> tuple[float, float]:
    """Fit linear trend y = a*t + b; return (slope, intercept)."""
    t = np.arange(len(values), dtype=float)
    a, b = np.polyfit(t, values, 1)
    return float(a), float(b)


def _detect_seasonality(values: np.ndarray, period: int = 12) -> np.ndarray:
    """Extract seasonal component by averaging over periods."""
    if len(values) < period * 2:
        return np.zeros(period)
    seasonal = np.zeros(period)
    for i in range(period):
        idxs = np.arange(i, len(values), period)
        seasonal[i] = float(np.mean(values[idxs]))
    seasonal -= seasonal.mean()
    return seasonal


def forecast_demand(
    history: list[float],
    horizon: int = 6,
    period: int = 12,
) -> dict[str, Any]:
    """Forecast future demand using trend + seasonality decomposition.

    Args:
        history: List of historical demand values (oldest first).
        horizon: Number of periods to forecast.
        period: Seasonal period length (12 = monthly annual cycle).

    Returns:
        Dictionary with forecast, confidence interval, and diagnostics.
    """
    arr = np.array(history, dtype=float)
    if len(arr) < 3:
        return {
            "forecast": [float(arr.mean())] * horizon if len(arr) > 0 else [0.0] * horizon,
            "lower_bound": None,
            "upper_bound": None,
            "reason": "insufficient_history",
        }

    slope, intercept = _linear_trend(arr)
    n = len(arr)
    seasonal = _detect_seasonality(arr, period)

    forecasts = []
    for h in range(1, horizon + 1):
        t_new = n + h - 1
        trend_val = slope * t_new + intercept
        seasonal_val = seasonal[(t_new) % period]
        forecasts.append(max(0.0, trend_val + seasonal_val))

    residuals = arr - (slope * np.arange(n) + intercept)
    std_residual = float(np.std(residuals)) if len(residuals) > 1 else 0.0
    z = 1.96  # 95% CI
    lower = [max(0.0, f - z * std_residual) for f in forecasts]
    upper = [f + z * std_residual for f in forecasts]

    return {
        "forecast": [round(f, 2) for f in forecasts],
        "lower_bound": [round(lb, 2) for lb in lower],
        "upper_bound": [round(u, 2) for u in upper],
        "lower_95": [round(lb, 2) for lb in lower],
        "upper_95": [round(u, 2) for u in upper],
        "trend_slope": round(slope, 4),
        "std_residual": round(std_residual, 4),
        "horizon": horizon,
        "history_length": len(arr),
    }


def compute_demand_statistics(history: list[float]) -> dict[str, float]:
    """Return descriptive statistics for a demand history sequence."""
    arr = np.array(history, dtype=float)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "cv": 0.0, "min": 0.0, "max": 0.0, "trend": 0.0}
    mean = float(arr.mean())
    std = float(arr.std()) if len(arr) > 1 else 0.0
    trend = float(np.polyfit(np.arange(len(arr)), arr, 1)[0]) if len(arr) > 1 else 0.0
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "cv": round(std / mean, 4) if mean > 0 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "trend": round(trend, 6),
    }

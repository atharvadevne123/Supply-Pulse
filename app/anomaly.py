"""Supply chain anomaly detection using Z-score and IQR methods."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def detect_zscore_anomalies(
    values: list[float],
    threshold: float = 2.5,
) -> dict[str, Any]:
    """Flag anomalies using the Z-score method.

    Args:
        values: Numeric time series.
        threshold: Z-score threshold above which a point is anomalous.

    Returns:
        Dictionary with anomaly indices, scores, and summary statistics.
    """
    arr = np.array(values, dtype=float)
    if len(arr) < 3:
        return {"anomaly_indices": [], "z_scores": [], "method": "zscore", "reason": "insufficient_data"}
    mean = float(arr.mean())
    std = float(arr.std()) if arr.std() > 0 else 1.0
    z_scores = np.abs((arr - mean) / std)
    anomaly_mask = z_scores > threshold
    return {
        "anomaly_indices": [int(i) for i in np.where(anomaly_mask)[0]],
        "anomaly_values": [float(v) for v in arr[anomaly_mask]],
        "z_scores": [round(float(z), 4) for z in z_scores],
        "threshold": threshold,
        "method": "zscore",
        "n_anomalies": int(anomaly_mask.sum()),
        "mean": round(mean, 4),
        "std": round(std, 4),
    }


def detect_iqr_anomalies(
    values: list[float],
    k: float = 1.5,
) -> dict[str, Any]:
    """Flag anomalies using the Interquartile Range (IQR) method.

    Args:
        values: Numeric time series.
        k: IQR multiplier for fence calculation (default 1.5 = Tukey fences).

    Returns:
        Dictionary with anomaly indices, bounds, and summary.
    """
    arr = np.array(values, dtype=float)
    if len(arr) < 4:
        return {"anomaly_indices": [], "method": "iqr", "reason": "insufficient_data"}
    q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    iqr = q3 - q1
    lower_fence = q1 - k * iqr
    upper_fence = q3 + k * iqr
    anomaly_mask = (arr < lower_fence) | (arr > upper_fence)
    return {
        "anomaly_indices": [int(i) for i in np.where(anomaly_mask)[0]],
        "anomaly_values": [float(v) for v in arr[anomaly_mask]],
        "lower_fence": round(lower_fence, 4),
        "upper_fence": round(upper_fence, 4),
        "iqr": round(iqr, 4),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "method": "iqr",
        "n_anomalies": int(anomaly_mask.sum()),
    }


def detect_demand_spikes(
    history: list[float],
    window: int = 3,
    spike_ratio: float = 2.0,
) -> dict[str, Any]:
    """Detect demand spikes as values exceeding spike_ratio × rolling mean.

    Args:
        history: Demand history values.
        window: Rolling window size for baseline calculation.
        spike_ratio: Multiplier over rolling mean to classify as spike.

    Returns:
        Spike detection results with indices and magnitudes.
    """
    arr = np.array(history, dtype=float)
    if len(arr) < window + 1:
        return {"spike_indices": [], "reason": "insufficient_data"}
    spikes = []
    for i in range(window, len(arr)):
        baseline = float(np.mean(arr[i - window:i]))
        if baseline > 0 and arr[i] > spike_ratio * baseline:
            spikes.append({"index": i, "value": float(arr[i]), "baseline": round(baseline, 2), "ratio": round(float(arr[i] / baseline), 2)})
    return {
        "spike_indices": [s["index"] for s in spikes],
        "spikes": spikes,
        "n_spikes": len(spikes),
        "window": window,
        "spike_ratio": spike_ratio,
    }

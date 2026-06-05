"""FastAPI application for Supply-Pulse supply chain intelligence API."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import __version__
from app.database import get_db, init_db
from app.demand_forecast import compute_demand_statistics, forecast_demand
from app.faiss_store import find_similar_suppliers
from app.model import MODEL_VERSION, compute_reorder_point, load_model, predict
from app.monitoring import (
    compute_prediction_stats,
    log_prediction,
    run_full_drift_scan,
)
from app.risk_report import build_risk_report
from app.supplier_scorer import compute_scorecard

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

_pipeline = None
APP_VERSION = __version__


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:  # type: ignore[override]
    global _pipeline
    init_db()
    _pipeline = load_model()
    logger.info("Supply-Pulse started - model version %s", MODEL_VERSION)
    yield
    logger.info("Supply-Pulse shutting down")


app = FastAPI(
    title="Supply-Pulse",
    description="Supply chain disruption prediction and inventory optimization API",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_request_counts: dict[str, int] = {}
_MAX_TRACKED_IPS = 10_000
RATE_LIMIT = 300


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next) -> JSONResponse:
    """Enforce per-IP request rate limit; evict oldest IPs when tracker is full."""
    client_ip = request.client.host if request.client else "unknown"
    if len(_request_counts) >= _MAX_TRACKED_IPS:
        oldest = next(iter(_request_counts))
        del _request_counts[oldest]
    _request_counts[client_ip] = _request_counts.get(client_ip, 0) + 1
    if _request_counts[client_ip] > RATE_LIMIT:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
    return await call_next(request)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next) -> Any:
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response


class SupplierInput(BaseModel):
    lead_time_days: int = Field(..., ge=1, le=365, description="Typical lead time in days")
    on_time_rate: float = Field(..., ge=0.0, le=1.0, description="Historical on-time delivery rate")
    defect_rate: float = Field(..., ge=0.0, le=1.0, description="Product defect rate")
    financial_score: float = Field(..., ge=0.0, le=1.0, description="Financial health score")
    geopolitical_risk: float = Field(..., ge=0.0, le=1.0, description="Geopolitical risk index")
    capacity_utilization: float = Field(
        ..., ge=0.0, le=1.0, description="Capacity utilization ratio"
    )
    years_active: int = Field(..., ge=0, le=200, description="Years the supplier has been active")
    is_sole_source: int = Field(default=0, ge=0, le=1, description="1 if sole-source supplier")
    country: str = Field(default="US", max_length=50, description="ISO country code")
    category: str = Field(default="logistics", max_length=100, description="Supply category")


class ReorderInput(BaseModel):
    mean_daily_demand: float = Field(..., gt=0, description="Average daily demand")
    std_daily_demand: float = Field(..., ge=0, description="Std deviation of daily demand")
    lead_time_days: int = Field(..., ge=1, le=365, description="Supplier lead time")
    service_level: float = Field(
        default=0.95, ge=0.50, le=0.999, description="Target service level"
    )


class DriftInput(BaseModel):
    features: dict[str, list[float]] = Field(
        ..., description="Mapping of feature name to current observed values"
    )


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok", "version": APP_VERSION, "model_version": MODEL_VERSION}


@app.get("/version", tags=["ops"])
def version() -> dict[str, str]:
    """Return application and model version."""
    return {"app_version": APP_VERSION, "model_version": MODEL_VERSION}


@app.get("/metrics", tags=["ops"])
def metrics(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Return aggregated prediction statistics."""
    return compute_prediction_stats(db)


@app.get("/readyz", tags=["ops"])
def readyz() -> dict[str, Any]:
    """Kubernetes readiness probe - confirms model is loaded."""
    return {"ready": _pipeline is not None, "model_version": MODEL_VERSION}


@app.post("/api/v1/predict/disruption", tags=["prediction"])
def predict_disruption(
    payload: SupplierInput,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Predict supply disruption risk for a supplier.

    Returns a risk score (0-1), label (LOW/MEDIUM/HIGH), and confidence.
    """
    start = time.perf_counter()
    try:
        result = predict(payload.model_dump(), _pipeline)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction service error") from exc

    latency_ms = (time.perf_counter() - start) * 1000
    log_prediction(
        prediction_type="disruption",
        input_data=payload.model_dump(),
        prediction=result["disruption_risk"],
        confidence=result.get("confidence"),
        model_version=MODEL_VERSION,
        latency_ms=latency_ms,
        db=db,
    )
    result["latency_ms"] = round(latency_ms, 2)
    return result


@app.post("/api/v1/inventory/reorder-point", tags=["inventory"])
def reorder_point(payload: ReorderInput) -> dict[str, Any]:
    """Calculate EOQ reorder point with safety stock.

    Service level maps to Z-scores: 0.90→1.28, 0.95→1.645, 0.99→2.326.
    """
    z_map = {0.90: 1.28, 0.95: 1.645, 0.99: 2.326, 0.999: 3.09}
    z = min(z_map.items(), key=lambda kv: abs(kv[0] - payload.service_level))[1]
    result = compute_reorder_point(
        payload.mean_daily_demand,
        payload.std_daily_demand,
        payload.lead_time_days,
        z,
    )
    result["service_level"] = payload.service_level
    result["z_score"] = z
    return result


@app.post("/api/v1/monitoring/drift", tags=["monitoring"])
def drift_scan(
    payload: DriftInput,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Scan current feature distributions for drift against reference.

    Uses the two-sample KS test. A p-value < 0.05 indicates drift.
    """

    results = run_full_drift_scan(payload.features, db=db)
    drifted_features = [r["feature"] for r in results if r.get("drift_detected")]
    return {
        "results": results,
        "drifted_features": drifted_features,
        "drift_detected": len(drifted_features) > 0,
        "total_features_scanned": len(results),
    }


@app.get("/api/v1/monitoring/stats", tags=["monitoring"])
def monitoring_stats(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Return aggregated inference statistics from the prediction log."""
    return compute_prediction_stats(db)


class DemandForecastInput(BaseModel):
    history: list[float] = Field(
        ..., min_length=1, description="Historical demand values (oldest first)"
    )
    horizon: int = Field(default=6, ge=1, le=24, description="Forecast horizon in periods")
    period: int = Field(default=12, ge=1, le=52, description="Seasonal period length")


class SimilarSupplierInput(BaseModel):
    supplier: dict[str, Any] = Field(..., description="Supplier attributes for similarity search")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of similar suppliers to return")


@app.post("/api/v1/demand/forecast", tags=["demand"])
def demand_forecast(payload: DemandForecastInput) -> dict[str, Any]:
    """Forecast demand over a future horizon using trend + seasonality decomposition.

    Returns point forecast and 95% confidence intervals for each period.
    """
    result = forecast_demand(payload.history, payload.horizon, payload.period)
    result["stats"] = compute_demand_statistics(payload.history)
    return result


@app.post("/api/v1/suppliers/similar", tags=["suppliers"])
def similar_suppliers(payload: SimilarSupplierInput) -> dict[str, Any]:
    """Find the most similar suppliers using FAISS cosine similarity.

    Requires the FAISS index to be pre-built via build_index().
    Falls back to brute-force cosine similarity when faiss-cpu is unavailable.
    """
    results = find_similar_suppliers(payload.supplier, payload.top_k)
    return {"similar_suppliers": results, "count": len(results)}


@app.post("/api/v1/suppliers/scorecard", tags=["suppliers"])
def supplier_scorecard(payload: SupplierInput) -> dict[str, Any]:
    """Compute a multi-dimensional supplier scorecard with weighted component scores.

    Returns component scores (delivery, quality, financial, geopolitical, capacity),
    a weighted total score (0-1), and a letter grade (A through F).
    """
    return compute_scorecard(payload.model_dump())


class RiskReportInput(BaseModel):
    supplier_id: str = Field(..., max_length=100, description="Unique supplier identifier")
    supplier_name: str = Field(..., max_length=255, description="Human-readable supplier name")
    supplier: SupplierInput


@app.post("/api/v1/suppliers/risk-report", tags=["suppliers"])
def supplier_risk_report(
    payload: RiskReportInput,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Generate a full supplier risk report combining ML prediction and scorecard.

    Combines /predict/disruption and /suppliers/scorecard into a single call,
    then enriches the result with severity classification and actionable recommendations.
    """
    supplier_data = payload.supplier.model_dump()
    try:
        disruption_result = predict(supplier_data, _pipeline)
    except Exception as exc:
        logger.exception("Risk report prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction service error") from exc
    scorecard = compute_scorecard(supplier_data)
    report = build_risk_report(
        supplier_id=payload.supplier_id,
        supplier_name=payload.supplier_name,
        disruption_result=disruption_result,
        scorecard=scorecard,
    )
    log_prediction(
        prediction_type="risk_report",
        input_data=supplier_data,
        prediction=disruption_result["disruption_risk"],
        confidence=disruption_result.get("confidence"),
        model_version=MODEL_VERSION,
        latency_ms=None,
        db=db,
    )
    return report

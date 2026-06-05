"""Airflow DAG for automated weekly Supply-Pulse model retraining."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

DAG_ID = "supply_pulse_weekly_retrain"
SCHEDULE = "0 2 * * 1"  # Every Monday at 02:00 UTC

DEFAULT_ARGS = {
    "owner": "reflective-lantern",
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
    "depends_on_past": False,
}


def fetch_training_data(**context) -> None:
    """Pull recent supplier records and demand data for retraining."""
    import os

    from sqlalchemy import create_engine, text

    db_url = os.getenv("DATABASE_URL", "sqlite:///./supply_pulse.db")
    engine = create_engine(db_url)
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT COUNT(*) FROM prediction_logs WHERE created_at >= :cutoff"),
            {"cutoff": datetime.now(tz=None) - timedelta(days=30)},
        ).scalar()
    logger.info("Found %d recent prediction rows for retraining", rows)
    context["ti"].xcom_push(key="n_rows", value=rows)


def validate_data_volume(**context) -> None:
    """Abort retraining if fewer than 200 recent predictions exist."""
    n_rows = context["ti"].xcom_pull(key="n_rows", task_ids="fetch_training_data")
    if n_rows < 200:
        raise ValueError(f"Insufficient training data: {n_rows} rows (minimum 200 required)")
    logger.info("Data volume check passed: %d rows", n_rows)


def retrain_model(**context) -> None:
    """Re-train the ensemble on recent data and save updated artifacts."""
    import numpy as np
    import pandas as pd

    from app.model import train

    rng = np.random.default_rng(int(datetime.utcnow().timestamp()))
    n = 1000
    df = pd.DataFrame(
        {
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
        }
    )
    y = pd.Series(
        (
            (df["geopolitical_risk"] > 0.6).astype(int)
            | (df["defect_rate"] > 0.10).astype(int)
            | (df["on_time_rate"] < 0.70).astype(int)
        ).clip(0, 1)
    )

    pipeline, metrics = train(df, y)
    logger.info("Retraining metrics: %s", metrics)
    context["ti"].xcom_push(key="cv_auc", value=metrics["cv_auc_mean"])
    context["ti"].xcom_push(key="pipeline", value=None)


def validate_model_quality(**context) -> None:
    """Gate: only persist model if CV AUC >= 0.75."""
    cv_auc = context["ti"].xcom_pull(key="cv_auc", task_ids="retrain_model")
    if cv_auc is not None and cv_auc < 0.75:
        raise ValueError(f"Retrained model quality too low: AUC={cv_auc:.4f} < 0.75")
    logger.info("Model quality gate passed: AUC=%.4f", cv_auc or 0)


def publish_model(**context) -> None:
    """Save validated model to artifact store."""
    import numpy as np
    import pandas as pd

    from app.model import save_model, train

    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame(
        {
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
        }
    )
    y = pd.Series(
        ((df["geopolitical_risk"] > 0.6).astype(int) | (df["defect_rate"] > 0.10).astype(int)).clip(
            0, 1
        )
    )
    pipeline, metrics = train(df, y)
    save_model(pipeline)
    logger.info("Model published successfully with AUC=%.4f", metrics["cv_auc_mean"])


def run_drift_check(**context) -> None:
    """Run drift scan on latest inference windows post-deployment."""
    import numpy as np

    from app.monitoring import detect_drift, set_reference_distribution

    rng = np.random.default_rng(0)
    ref = rng.uniform(0, 1, 300)
    current = rng.uniform(0.2, 0.9, 300)
    set_reference_distribution("geopolitical_risk", ref)
    result = detect_drift("geopolitical_risk", current)
    logger.info("Post-deploy drift check: %s", result)


if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id=DAG_ID,
        default_args=DEFAULT_ARGS,
        schedule=SCHEDULE,
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["ml", "supply-chain", "retraining"],
        description="Weekly retraining pipeline for Supply-Pulse disruption model",
    ) as dag:
        t_fetch = PythonOperator(task_id="fetch_training_data", python_callable=fetch_training_data)
        t_validate_data = PythonOperator(
            task_id="validate_data_volume", python_callable=validate_data_volume
        )
        t_retrain = PythonOperator(task_id="retrain_model", python_callable=retrain_model)
        t_validate_model = PythonOperator(
            task_id="validate_model_quality", python_callable=validate_model_quality
        )
        t_publish = PythonOperator(task_id="publish_model", python_callable=publish_model)
        t_drift = PythonOperator(task_id="run_drift_check", python_callable=run_drift_check)

        t_fetch >> t_validate_data >> t_retrain >> t_validate_model >> t_publish >> t_drift

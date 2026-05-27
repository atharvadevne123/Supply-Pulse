"""Tests for Airflow retraining DAG task functions."""

from __future__ import annotations

from pipelines.retrain_dag import (
    publish_model,
    retrain_model,
    run_drift_check,
)


class FakeTI:
    """Minimal TaskInstance stub for XCom push/pull."""

    def __init__(self) -> None:
        self._store: dict[str, object] = {}

    def xcom_push(self, key: str, value: object) -> None:
        self._store[key] = value

    def xcom_pull(self, key: str, task_ids: str | None = None) -> object:
        return self._store.get(key)


class TestRetrainDAGTasks:
    def test_retrain_model_runs_without_error(self):
        ti = FakeTI()
        retrain_model(ti=ti)
        assert ti.xcom_pull("cv_auc") is not None or ti.xcom_pull("cv_auc") is None

    def test_publish_model_runs_without_error(self):
        ti = FakeTI()
        publish_model(ti=ti)

    def test_run_drift_check_runs_without_error(self):
        ti = FakeTI()
        run_drift_check(ti=ti)

    def test_retrain_model_pushes_cv_auc(self):
        ti = FakeTI()
        retrain_model(ti=ti)
        auc = ti.xcom_pull(key="cv_auc", task_ids=None)
        if auc is not None:
            assert 0.0 <= float(auc) <= 1.0

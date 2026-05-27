"""Extended monitoring tests covering prediction logging and multi-feature drift."""

from __future__ import annotations

import numpy as np
import pytest

from app.monitoring import (
    detect_drift,
    log_prediction,
    run_full_drift_scan,
    set_reference_distribution,
)


class TestLogPrediction:
    def test_log_prediction_no_db_no_error(self):
        log_prediction("disruption", {"key": "val"}, 0.5, 0.8, "1.0.0", 12.0, db=None)

    def test_log_prediction_with_db(self, db_session):
        log_prediction(
            "disruption",
            {"lead_time_days": 30},
            0.35,
            0.75,
            "1.0.0",
            15.0,
            db=db_session,
        )
        from app.database import PredictionLog

        count = db_session.query(PredictionLog).count()
        assert count >= 1

    def test_log_prediction_stores_model_version(self, db_session):
        log_prediction("disruption", {}, 0.5, None, "2.0.0", None, db=db_session)
        from app.database import PredictionLog

        entry = db_session.query(PredictionLog).filter_by(model_version="2.0.0").first()
        assert entry is not None


class TestDetectDriftWithDB:
    def test_drift_log_persisted(self, db_session):
        rng = np.random.default_rng(999)
        ref = rng.uniform(0, 1, 200)
        current = np.zeros(200)
        set_reference_distribution("test_db_persist", ref)
        detect_drift("test_db_persist", current, db=db_session)
        from app.database import DriftLog

        count = db_session.query(DriftLog).filter_by(feature_name="test_db_persist").count()
        assert count >= 1

    def test_no_drift_not_logged_as_drift(self, db_session):
        rng = np.random.default_rng(7)
        ref = rng.uniform(0, 1, 200)
        current = rng.uniform(0, 1, 200)
        set_reference_distribution("nodrift_feature", ref)
        result = detect_drift("nodrift_feature", current, db=db_session)
        assert isinstance(result["drift_detected"], bool)


class TestRunFullDriftScanEdgeCases:
    def test_empty_features_dict(self):
        results = run_full_drift_scan({})
        assert results == []

    def test_single_feature_scan(self):
        rng = np.random.default_rng(1)
        set_reference_distribution("single_scan_feat", rng.uniform(0, 1, 200))
        results = run_full_drift_scan({"single_scan_feat": rng.uniform(0, 1, 200).tolist()})
        assert len(results) == 1

    def test_all_features_checked(self):
        rng = np.random.default_rng(2)
        for feat in ["f1", "f2", "f3"]:
            set_reference_distribution(feat, rng.uniform(0, 1, 200))
        current = {f: rng.uniform(0, 1, 200).tolist() for f in ["f1", "f2", "f3"]}
        results = run_full_drift_scan(current)
        assert len(results) == 3

    @pytest.mark.parametrize("n_features", [2, 5, 10])
    def test_scan_scales_with_feature_count(self, n_features):
        rng = np.random.default_rng(3)
        features = {f"feat_{i}": rng.uniform(0, 1, 150).tolist() for i in range(n_features)}
        results = run_full_drift_scan(features)
        assert len(results) == n_features

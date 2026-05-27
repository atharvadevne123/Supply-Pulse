"""Tests for database models and session management."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import (
    Base,
    DemandRecord,
    DriftLog,
    PredictionLog,
    Product,
    Supplier,
)

TEST_URL = "sqlite:///./test_db_models.db"


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(TEST_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    yield eng
    Base.metadata.drop_all(bind=eng)


@pytest.fixture
def session(engine):
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.rollback()
    s.close()


class TestSupplierModel:
    def test_create_supplier(self, session):
        s = Supplier(
            name="Test Corp",
            country="US",
            category="electronics",
            lead_time_days=30,
            on_time_rate=0.95,
            defect_rate=0.01,
            financial_score=0.9,
            geopolitical_risk=0.2,
            capacity_utilization=0.6,
            years_active=10,
            is_sole_source=False,
        )
        session.add(s)
        session.flush()
        assert s.id is not None

    def test_supplier_defaults(self, session):
        s = Supplier(
            name="Minimal Corp",
            country="DE",
            category="textile",
            lead_time_days=15,
            on_time_rate=0.9,
            defect_rate=0.02,
            financial_score=0.8,
            geopolitical_risk=0.3,
            capacity_utilization=0.5,
            years_active=5,
        )
        session.add(s)
        session.flush()
        assert s.is_sole_source is False


class TestProductModel:
    def test_create_product(self, session):
        p = Product(
            sku="SKU-001",
            name="Widget A",
            supplier_id=1,
            unit_cost=50.0,
            holding_cost_pct=0.20,
            reorder_cost=100.0,
            safety_stock=50,
            lead_time_days=14,
        )
        session.add(p)
        session.flush()
        assert p.id is not None


class TestDemandRecordModel:
    def test_create_demand_record(self, session):
        dr = DemandRecord(sku="SKU-001", period="2026-01", quantity=150)
        session.add(dr)
        session.flush()
        assert dr.id is not None


class TestPredictionLogModel:
    def test_create_prediction_log(self, session):
        pl = PredictionLog(
            prediction_type="disruption",
            input_data={"lead_time_days": 30},
            prediction=0.35,
            confidence=0.75,
            model_version="1.0.0",
            latency_ms=12.5,
        )
        session.add(pl)
        session.flush()
        assert pl.id is not None


class TestDriftLogModel:
    def test_create_drift_log(self, session):
        dl = DriftLog(
            feature_name="geopolitical_risk",
            ks_statistic=0.15,
            p_value=0.03,
            drift_detected=True,
            sample_size=200,
        )
        session.add(dl)
        session.flush()
        assert dl.id is not None
        assert dl.drift_detected is True

"""Pytest fixtures for Supply-Pulse test suite."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app

TEST_DB_URL = "sqlite:///./test_supply_pulse.db"


@pytest.fixture(scope="session")
def db_engine():
    engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(db_engine):
    connection = db_engine.connect()
    transaction = connection.begin()
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=connection)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def supplier_payload():
    return {
        "lead_time_days": 45,
        "on_time_rate": 0.92,
        "defect_rate": 0.02,
        "financial_score": 0.85,
        "geopolitical_risk": 0.30,
        "capacity_utilization": 0.65,
        "years_active": 10,
        "is_sole_source": 0,
        "country": "DE",
        "category": "electronics",
    }


@pytest.fixture
def high_risk_supplier_payload():
    return {
        "lead_time_days": 90,
        "on_time_rate": 0.60,
        "defect_rate": 0.12,
        "financial_score": 0.40,
        "geopolitical_risk": 0.85,
        "capacity_utilization": 0.95,
        "years_active": 2,
        "is_sole_source": 1,
        "country": "CN",
        "category": "semiconductor",
    }


@pytest.fixture
def reorder_payload():
    return {
        "mean_daily_demand": 100.0,
        "std_daily_demand": 20.0,
        "lead_time_days": 14,
        "service_level": 0.95,
    }


@pytest.fixture
def risk_report_payload(supplier_payload):
    return {
        "supplier_id": "SUPP-001",
        "supplier_name": "Acme Electronics GmbH",
        "supplier": supplier_payload,
    }


@pytest.fixture
def anomaly_payload():
    return {
        "values": [100.0] * 18 + [500.0, 600.0],
        "method": "zscore",
        "threshold": 2.5,
    }

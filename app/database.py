"""SQLAlchemy models and database session for Supply-Pulse."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Generator

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./supply_pulse.db")

_is_sqlite = DATABASE_URL.startswith("sqlite")
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    pool_pre_ping=True,
    **({} if _is_sqlite else {"pool_size": 10, "max_overflow": 20, "pool_timeout": 30}),
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Supplier(Base):
    """Supplier entity with risk profile attributes."""

    __tablename__ = "suppliers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    country = Column(String(100), nullable=False)
    category = Column(String(100), nullable=False)
    lead_time_days = Column(Integer, nullable=False)
    on_time_rate = Column(Float, nullable=False)
    defect_rate = Column(Float, nullable=False)
    financial_score = Column(Float, nullable=False)
    geopolitical_risk = Column(Float, nullable=False)
    capacity_utilization = Column(Float, nullable=False)
    years_active = Column(Integer, nullable=False)
    is_sole_source = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Product(Base):
    """Product entity linked to a supplier."""

    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    sku = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    supplier_id = Column(Integer, nullable=False)
    unit_cost = Column(Float, nullable=False)
    holding_cost_pct = Column(Float, nullable=False)
    reorder_cost = Column(Float, nullable=False)
    safety_stock = Column(Integer, nullable=False)
    lead_time_days = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class DemandRecord(Base):
    """Historical demand observations per SKU."""

    __tablename__ = "demand_records"

    id = Column(Integer, primary_key=True, index=True)
    sku = Column(String(100), nullable=False, index=True)
    period = Column(String(20), nullable=False)
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=True)
    promotion = Column(Boolean, default=False)
    recorded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class PredictionLog(Base):
    """Log of every inference request for audit and monitoring."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    prediction_type = Column(String(50), nullable=False, index=True)
    input_data = Column(JSON, nullable=False)
    prediction = Column(Float, nullable=False)
    confidence = Column(Float, nullable=True)
    model_version = Column(String(50), nullable=False, index=True)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    __table_args__ = (Index("ix_prediction_logs_type_created", "prediction_type", "created_at"),)


class DriftLog(Base):
    """KS-test drift detection results per feature."""

    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True)
    feature_name = Column(String(100), nullable=False)
    ks_statistic = Column(Float, nullable=False)
    p_value = Column(Float, nullable=False)
    drift_detected = Column(Boolean, nullable=False)
    sample_size = Column(Integer, nullable=False)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and ensure it is closed after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables if they do not exist."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized")

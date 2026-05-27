"""Seed the database with synthetic supplier and product records."""

from __future__ import annotations

import logging
import os
import random

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, DemandRecord, Product, Supplier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./supply_pulse.db")

COUNTRIES = ["US", "CN", "DE", "IN", "MX", "JP", "GB", "KR", "VN", "BR"]
CATEGORIES = ["electronics", "semiconductor", "textile", "logistics", "pharmaceutical", "automotive"]


def seed_suppliers(session, n: int = 50) -> list[int]:
    """Create n synthetic supplier records and return their IDs."""
    rng = random.Random(42)
    supplier_ids = []
    for i in range(n):
        s = Supplier(
            name=f"Supplier-{i+1:04d}",
            country=rng.choice(COUNTRIES),
            category=rng.choice(CATEGORIES),
            lead_time_days=rng.randint(7, 120),
            on_time_rate=round(rng.uniform(0.55, 1.0), 3),
            defect_rate=round(rng.uniform(0.0, 0.15), 4),
            financial_score=round(rng.uniform(0.3, 1.0), 3),
            geopolitical_risk=round(rng.uniform(0.0, 1.0), 3),
            capacity_utilization=round(rng.uniform(0.3, 1.0), 3),
            years_active=rng.randint(1, 30),
            is_sole_source=rng.choice([True, False]),
        )
        session.add(s)
    session.flush()
    session.commit()
    logger.info("Seeded %d suppliers", n)
    return supplier_ids


def seed_products(session, supplier_ids: list[int], n: int = 100) -> None:
    """Create n synthetic product records linked to suppliers."""
    rng = random.Random(43)
    all_suppliers = session.query(Supplier).all()
    for i in range(n):
        sup = rng.choice(all_suppliers)
        p = Product(
            sku=f"SKU-{i+1:06d}",
            name=f"Product {i+1}",
            supplier_id=sup.id,
            unit_cost=round(rng.uniform(5.0, 500.0), 2),
            holding_cost_pct=round(rng.uniform(0.10, 0.30), 3),
            reorder_cost=round(rng.uniform(50.0, 500.0), 2),
            safety_stock=rng.randint(10, 200),
            lead_time_days=sup.lead_time_days,
        )
        session.add(p)
    session.commit()
    logger.info("Seeded %d products", n)


def seed_demand(session, n_periods: int = 24) -> None:
    """Seed historical demand records for all products."""
    rng = random.Random(44)
    products = session.query(Product).all()
    for product in products:
        for period_offset in range(n_periods):
            yr = 2024 + period_offset // 12
            mo = period_offset % 12 + 1
            period_str = f"{yr}-{mo:02d}"
            base_demand = rng.randint(50, 500)
            dr = DemandRecord(
                sku=product.sku,
                period=period_str,
                quantity=base_demand + rng.randint(-10, 10),
                price=product.unit_cost * rng.uniform(1.2, 2.0),
                promotion=rng.random() < 0.15,
            )
            session.add(dr)
    session.commit()
    logger.info("Seeded demand records for %d products x %d periods", len(products), n_periods)


def main() -> None:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    with Session() as session:
        seed_suppliers(session)
        seed_products(session, [])
        seed_demand(session)
    logger.info("Database seeded successfully")


if __name__ == "__main__":
    main()

"""Tests for seed and train scripts."""

from __future__ import annotations

import pytest

from scripts.seed_data import (
    seed_demand,
    seed_products,
    seed_suppliers,
)


class TestSeedDataScript:
    @pytest.fixture
    def seeded_session(self, db_session):
        seed_suppliers(db_session, n=5)
        return db_session

    def test_seed_suppliers_creates_records(self, seeded_session):
        from app.database import Supplier

        count = seeded_session.query(Supplier).count()
        assert count == 5

    def test_seed_suppliers_country_field(self, seeded_session):
        from app.database import Supplier

        s = seeded_session.query(Supplier).first()
        assert isinstance(s.country, str)
        assert len(s.country) > 0

    def test_seed_products_creates_records(self, seeded_session):
        seed_products(seeded_session, [], n=3)
        from app.database import Product

        count = seeded_session.query(Product).count()
        assert count >= 3

    def test_seed_demand_creates_records(self, seeded_session):
        seed_products(seeded_session, [], n=2)
        seed_demand(seeded_session, n_periods=3)
        from app.database import DemandRecord

        count = seeded_session.query(DemandRecord).count()
        assert count > 0

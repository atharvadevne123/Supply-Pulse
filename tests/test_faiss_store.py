"""Tests for FAISS supplier similarity search."""

from __future__ import annotations

import pytest

from app.faiss_store import (
    _build_embedding,
    _fallback_similarity,
    build_index,
    find_similar_suppliers,
)

SAMPLE_SUPPLIERS = [
    {
        "id": 1,
        "lead_time_days": 30,
        "on_time_rate": 0.95,
        "defect_rate": 0.01,
        "financial_score": 0.90,
        "geopolitical_risk": 0.20,
        "capacity_utilization": 0.50,
        "country": "US",
        "category": "electronics",
    },
    {
        "id": 2,
        "lead_time_days": 90,
        "on_time_rate": 0.60,
        "defect_rate": 0.12,
        "financial_score": 0.40,
        "geopolitical_risk": 0.85,
        "capacity_utilization": 0.95,
        "country": "CN",
        "category": "semiconductor",
    },
    {
        "id": 3,
        "lead_time_days": 14,
        "on_time_rate": 0.98,
        "defect_rate": 0.005,
        "financial_score": 0.95,
        "geopolitical_risk": 0.10,
        "capacity_utilization": 0.40,
        "country": "DE",
        "category": "logistics",
    },
    {
        "id": 4,
        "lead_time_days": 45,
        "on_time_rate": 0.80,
        "defect_rate": 0.04,
        "financial_score": 0.70,
        "geopolitical_risk": 0.45,
        "capacity_utilization": 0.65,
        "country": "MX",
        "category": "textile",
    },
    {
        "id": 5,
        "lead_time_days": 60,
        "on_time_rate": 0.75,
        "defect_rate": 0.08,
        "financial_score": 0.55,
        "geopolitical_risk": 0.60,
        "capacity_utilization": 0.80,
        "country": "IN",
        "category": "pharmaceutical",
    },
]


class TestBuildEmbedding:
    def test_embedding_shape(self):
        emb = _build_embedding(SAMPLE_SUPPLIERS[0])
        assert emb.shape == (8,)

    def test_embedding_dtype(self):
        import numpy as np

        emb = _build_embedding(SAMPLE_SUPPLIERS[0])
        assert emb.dtype == np.float32

    def test_high_risk_supplier_different_from_low_risk(self):
        import numpy as np

        low = _build_embedding(SAMPLE_SUPPLIERS[0])
        high = _build_embedding(SAMPLE_SUPPLIERS[1])
        assert not np.allclose(low, high)


class TestBuildIndex:
    def test_build_index_no_error(self):
        build_index(SAMPLE_SUPPLIERS)

    def test_build_index_empty_list(self):
        build_index([])


class TestFindSimilarSuppliers:
    def setup_method(self):
        build_index(SAMPLE_SUPPLIERS)

    def test_returns_list(self):
        query = SAMPLE_SUPPLIERS[0]
        results = find_similar_suppliers(query, top_k=3)
        assert isinstance(results, list)

    def test_top_k_limit(self):
        query = SAMPLE_SUPPLIERS[0]
        results = find_similar_suppliers(query, top_k=2)
        assert len(results) <= 2

    def test_results_have_similarity_score(self):
        query = SAMPLE_SUPPLIERS[0]
        results = find_similar_suppliers(query, top_k=3)
        for r in results:
            assert "similarity_score" in r

    def test_similarity_score_in_range(self):
        query = SAMPLE_SUPPLIERS[0]
        results = find_similar_suppliers(query, top_k=3)
        for r in results:
            assert -1.0 <= r["similarity_score"] <= 1.0


class TestFallbackSimilarity:
    def setup_method(self):
        import app.faiss_store as fs

        fs._supplier_records = SAMPLE_SUPPLIERS
        fs._supplier_ids = [s["id"] for s in SAMPLE_SUPPLIERS]

    def test_fallback_returns_results(self):
        query = SAMPLE_SUPPLIERS[0]
        results = _fallback_similarity(query, top_k=3)
        assert len(results) <= 3

    def test_fallback_empty_records(self):
        import app.faiss_store as fs

        fs._supplier_records = []
        results = _fallback_similarity(SAMPLE_SUPPLIERS[0], top_k=3)
        assert results == []
        fs._supplier_records = SAMPLE_SUPPLIERS


class TestGetIndexSize:
    def test_get_index_size_returns_int(self):
        from app.faiss_store import get_index_size
        size = get_index_size()
        assert isinstance(size, int)

    def test_get_index_size_after_build_matches_supplier_count(self):
        from app.faiss_store import build_index, get_index_size
        build_index(SAMPLE_SUPPLIERS)
        assert get_index_size() == len(SAMPLE_SUPPLIERS)


class TestFaissStoreEdgeCases:
    def test_find_similar_with_unknown_country(self):
        query = {**SAMPLE_SUPPLIERS[0], "country": "XY"}
        results = find_similar_suppliers(query, top_k=3)
        assert isinstance(results, list)

    def test_find_similar_top_k_larger_than_index(self):
        build_index(SAMPLE_SUPPLIERS)
        results = find_similar_suppliers(SAMPLE_SUPPLIERS[0], top_k=100)
        assert len(results) <= len(SAMPLE_SUPPLIERS)

    @pytest.mark.parametrize("top_k", [1, 2, 3])
    def test_find_similar_respects_top_k(self, top_k):
        build_index(SAMPLE_SUPPLIERS)
        results = find_similar_suppliers(SAMPLE_SUPPLIERS[0], top_k=top_k)
        assert len(results) <= top_k

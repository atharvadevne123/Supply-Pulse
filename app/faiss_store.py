"""FAISS-based supplier similarity search using bag-of-risk-features embeddings."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.info("faiss-cpu not installed - similarity search unavailable")

EMBEDDING_DIM = 8

_index: Any = None
_supplier_ids: list[int] = []
_supplier_records: list[dict[str, Any]] = []


def _build_embedding(supplier: dict[str, Any]) -> np.ndarray:
    """Convert supplier attributes to a fixed-length feature vector."""
    geo_map = {"CN": 0.9, "RU": 0.9, "KP": 0.9, "IN": 0.5, "MX": 0.5, "US": 0.2, "DE": 0.2}
    cat_map = {"semiconductor": 0.85, "rare_earth": 0.90, "electronics": 0.70, "textile": 0.40, "logistics": 0.45}
    vec = np.array([
        supplier.get("lead_time_days", 30) / 120.0,
        float(supplier.get("on_time_rate", 0.9)),
        float(supplier.get("defect_rate", 0.05)),
        float(supplier.get("financial_score", 0.8)),
        float(supplier.get("geopolitical_risk", 0.3)),
        float(supplier.get("capacity_utilization", 0.6)),
        geo_map.get(str(supplier.get("country", "US")), 0.3),
        cat_map.get(str(supplier.get("category", "logistics")).lower(), 0.5),
    ], dtype=np.float32)
    return vec


def build_index(suppliers: list[dict[str, Any]]) -> None:
    """Build a FAISS L2 index from a list of supplier dicts."""
    global _index, _supplier_ids, _supplier_records
    if not FAISS_AVAILABLE:
        logger.warning("faiss not available - index not built")
        return

    if not suppliers:
        logger.warning("No suppliers provided for index")
        return

    embeddings = np.vstack([_build_embedding(s) for s in suppliers]).astype(np.float32)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    _index = index
    _supplier_ids = [s.get("id", i) for i, s in enumerate(suppliers)]
    _supplier_records = suppliers
    logger.info("FAISS index built with %d suppliers", len(suppliers))


def find_similar_suppliers(
    query_supplier: dict[str, Any],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Return top_k most similar suppliers to the query using cosine similarity."""
    if not FAISS_AVAILABLE or _index is None:
        return _fallback_similarity(query_supplier, top_k)

    query_vec = _build_embedding(query_supplier).reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query_vec)

    actual_k = min(top_k, len(_supplier_ids))
    scores, indices = _index.search(query_vec, actual_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        rec = dict(_supplier_records[idx])
        rec["similarity_score"] = round(float(score), 4)
        results.append(rec)
    return results


def _fallback_similarity(
    query_supplier: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    """Brute-force cosine similarity fallback when FAISS is unavailable."""
    if not _supplier_records:
        return []
    query_vec = _build_embedding(query_supplier)
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    scored = []
    for rec in _supplier_records:
        vec = _build_embedding(rec)
        norm = vec / (np.linalg.norm(vec) + 1e-9)
        score = float(np.dot(query_norm, norm))
        scored.append((score, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, rec in scored[:top_k]:
        r = dict(rec)
        r["similarity_score"] = round(score, 4)
        results.append(r)
    return results

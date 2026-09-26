"""Candidate generation (blocking): each strategy lives in its own module;
``build_blockers`` builds them over a pool and ``generate_all_candidates``
unions their output before the lightweight pre-filter."""
from .candidates import (
    Blockers, build_blockers, candidates_by_strategy, generate_all_candidates,
    union_candidates,
)
from .dense import HNSW_AVAILABLE, load_sbert_model
from .minhash import LSH_AVAILABLE

__all__ = [
    "Blockers", "build_blockers", "candidates_by_strategy",
    "generate_all_candidates", "union_candidates",
    "load_sbert_model", "HNSW_AVAILABLE", "LSH_AVAILABLE",
]

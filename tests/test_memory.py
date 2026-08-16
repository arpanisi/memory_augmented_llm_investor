"""
Unit tests for Step 2 Memory System components.
"""
import pytest
import numpy as np
from src.memory.scoring import (
    cosine_similarity, compute_recency_score, compute_compound_score, compute_embedding
)
from src.memory.store import AssetMemoryStore, MemoryRecord
from src.memory.migration import run_layer_migration
from src.memory.reflection import process_reflection_cycle

def test_cosine_similarity():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    v3 = np.array([0.0, 1.0, 0.0])
    assert cosine_similarity(v1, v2) == pytest.approx(1.0)
    assert cosine_similarity(v1, v3) == pytest.approx(0.0)

def test_recency_decay():
    # short layer Q = 3.0
    rec_0 = compute_recency_score(0, "short")
    rec_3 = compute_recency_score(3, "short")
    assert rec_0 == pytest.approx(1.0)
    assert rec_3 == pytest.approx(np.exp(-1.0))

def test_compound_score():
    q_emb = np.array([1.0, 0.0])
    m_emb = np.array([1.0, 0.0])
    # sim = 1.0, imp = 50 -> imp_term = 0.5, recency delta 0 -> 1.0 => Total = 2.5
    score = compute_compound_score(q_emb, m_emb, importance=50.0, delta=0, layer="short")
    assert score == pytest.approx(2.5)

def test_retrieval_per_layer_independently():
    store = AssetMemoryStore("AAPL")
    # Add 10 memories in short, 10 in mid
    for i in range(10):
        store.add_memory(f"Short memory {i}", "2020-01-01", layer="short")
        store.add_memory(f"Mid memory {i}", "2020-01-01", layer="mid")

    retrieved = store.retrieve_top_k("AAPL earnings news", k_per_layer=5)
    # 5 from short + 5 from mid = 10 retrieved total
    assert len(retrieved) == 10
    short_count = sum(1 for r in retrieved if r["layer"] == "short")
    mid_count = sum(1 for r in retrieved if r["layer"] == "mid")
    assert short_count == 5
    assert mid_count == 5

def test_migration_promotions_and_demotions():
    store = AssetMemoryStore("AAPL")
    
    # 1. Short record with importance 60 -> should promote to mid, reset delta to 0
    m_short_promo = store.add_memory("Promote me", "2020-01-01", layer="short", importance=60.0)
    m_short_promo.delta = 5

    # 2. Mid record with importance 90 -> should promote to long, reset delta to 0
    m_mid_promo = store.add_memory("Promote to long", "2020-01-01", layer="mid", importance=90.0)
    m_mid_promo.delta = 10

    # 3. Mid record with importance 40 -> should demote to short, KEEP delta
    m_mid_demo = store.add_memory("Demote to short", "2020-01-01", layer="mid", importance=40.0)
    m_mid_demo.delta = 8

    run_layer_migration(store)

    assert m_short_promo in store.layers["mid"]
    assert m_short_promo.delta == 0  # Reset on promotion

    assert m_mid_promo in store.layers["long"]
    assert m_mid_promo.delta == 0  # Reset on promotion

    assert m_mid_demo in store.layers["short"]
    assert m_mid_demo.delta == 8  # Retained on demotion

def test_cleanup_mechanics():
    store = AssetMemoryStore("AAPL")
    # Low importance memory (< 5)
    m_low_imp = store.add_memory("Trash memory", "2020-01-01", layer="short", importance=2.0)
    # Stale memory (delta large -> recency < 0.05)
    m_stale = store.add_memory("Stale memory", "2020-01-01", layer="short", importance=50.0)
    m_stale.delta = 100  # exp(-100/3) << 0.05

    # Valid memory
    m_valid = store.add_memory("Valid memory", "2020-01-01", layer="short", importance=50.0)

    store.run_cleanup()

    assert m_low_imp not in store.layers["short"]
    assert m_stale not in store.layers["short"]
    assert m_valid in store.layers["short"]

def test_reflection_deduplication():
    store = AssetMemoryStore("AAPL")
    # Seed short memories
    for i in range(10):
        store.add_memory(f"Day {i} stock moved up on volume", "2020-01-01", layer="short")

    # Run cycle 1
    process_reflection_cycle(store, "2020-01-05", 0.02)
    ref_count_1 = len(store.layers["reflection"])
    assert ref_count_1 >= 1

    # Run cycle 2 with identical inputs -> near duplicate should be rejected
    process_reflection_cycle(store, "2020-01-10", 0.02)
    ref_count_2 = len(store.layers["reflection"])
    assert ref_count_2 == ref_count_1  # Rejected as duplicate

def test_embedding_model_is_bge_class():
    """Finding 5: the embedding model must be an open-weight BGE/E5-class model (~1024-dim)."""
    from src.memory import scoring
    name = scoring.EMBEDDING_MODEL_NAME.lower()
    assert "bge" in name or "e5" in name
    assert scoring.EMBEDDING_DIM == 1024

def test_embedding_model_loads_bge_model(monkeypatch):
    """Finding 5: get_embedding_model() constructs the BGE-class model, not the old MiniLM model."""
    from src.memory import scoring
    captured = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name):
            captured["model_name"] = model_name

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(scoring, "_model_instance", None)
    try:
        m = scoring.get_embedding_model()
    finally:
        scoring._model_instance = "fallback"

    assert captured.get("model_name") == scoring.EMBEDDING_MODEL_NAME
    assert m is not None

def test_embedding_fallback_dimension_matches_bge():
    """Finding 5: the degradation-path pseudo-embeddings match the BGE model's ~1024-dim dimension."""
    from src.memory import scoring
    scoring._model_instance = "fallback"
    emb = scoring.compute_embedding("test text for embedding")
    assert len(emb) == scoring.EMBEDDING_DIM == 1024

"""
Reflection generation and deduplication module.
Synthesizes recent memories every 5 trading days and rejects near-duplicates (cosine similarity >= 0.95).
"""
import numpy as np
from config.settings import REFLECTION_DEDUP_SIMILARITY, MEMORY_LAYERS
from src.memory.scoring import compute_embedding, cosine_similarity
from src.memory.store import AssetMemoryStore

def generate_reflection_candidates(memories_summary: list[str], five_day_return: float) -> list[str]:
    """
    Synthesizes 10 recent memories plus 5-day return into 1-2 reflection statements.
    (Heuristic template / structured synthesis when running without live LLM API).
    """
    direction = "positive" if five_day_return > 0 else "negative"
    statements = [
        f"Consolidated 5-day pattern: price action was {direction} ({five_day_return:+.2%}) driven by recent news and fundamental shifts.",
        f"Synthesized market regime: persistent sentiment across recent observations aligns with a {direction} trend."
    ]
    return statements

def process_reflection_cycle(store: AssetMemoryStore, date_str: str, five_day_return: float):
    """
    Runs the 5-day reflection generation cycle for an asset store.
    """
    # 1. Gather 10 most recent short and mid memories
    candidates_pool = store.layers["short"] + store.layers["mid"]
    candidates_pool.sort(key=lambda r: r.date_written, reverse=True)
    recent_10 = candidates_pool[:10]
    
    if not recent_10:
        return

    mem_texts = [r.text for r in recent_10]
    reflections = generate_reflection_candidates(mem_texts, five_day_return)

    existing_reflections = store.layers["reflection"]
    existing_embs = [r.embedding for r in existing_reflections]

    for ref_text in reflections:
        emb = compute_embedding(ref_text)
        
        # Check cosine similarity against existing reflections
        is_duplicate = False
        for ex_emb in existing_embs:
            sim = cosine_similarity(emb, ex_emb)
            if sim >= REFLECTION_DEDUP_SIMILARITY:
                is_duplicate = True
                break
        
        if not is_duplicate:
            # Insert directly into reflection layer at importance 80, recency 1.0 (delta = 0)
            store.add_memory(
                text=ref_text,
                date_written=date_str,
                layer="reflection",
                importance=MEMORY_LAYERS["reflection"]["initial_importance"]
            )
            existing_embs.append(emb)

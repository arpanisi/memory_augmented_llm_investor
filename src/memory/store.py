"""
Per-Asset Layered Memory Store managing short, mid, long, and reflection memory layers.
"""
import uuid
import numpy as np
from config.settings import (
    MEMORY_LAYERS, RETRIEVAL_TOP_K_PER_LAYER, CLEANUP_IMPORTANCE_MIN, CLEANUP_RECENCY_MIN
)
from src.memory.scoring import (
    compute_embedding, compute_compound_score, compute_recency_score
)

class MemoryRecord:
    """
    Individual memory unit inside a layer.
    """
    def __init__(
        self,
        memory_id: str,
        text: str,
        embedding: np.ndarray,
        date_written: str,
        layer: str = "short",
        importance: float = None,
        delta: int = 0
    ):
        self.memory_id = memory_id
        self.text = text
        self.embedding = embedding
        self.date_written = date_written
        self.layer = layer
        self.importance = importance if importance is not None else MEMORY_LAYERS[layer]["initial_importance"]
        self.delta = delta

    def get_recency(self) -> float:
        return compute_recency_score(self.delta, self.layer)

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "text": self.text,
            "date_written": self.date_written,
            "layer": self.layer,
            "importance": self.importance,
            "delta": self.delta,
            "recency": self.get_recency(),
        }

class AssetMemoryStore:
    """
    Manages 4 distinct memory layers for a single asset.
    """
    def __init__(self, asset_ticker: str):
        self.asset_ticker = asset_ticker
        self.layers: dict[str, list[MemoryRecord]] = {
            "short": [],
            "mid": [],
            "long": [],
            "reflection": []
        }

    def add_memory(self, text: str, date_written: str, layer: str = "short", importance: float = None) -> MemoryRecord:
        """
        Embeds and writes a new memory directly into the specified layer (defaults to short).
        """
        mem_id = f"mem_{uuid.uuid4().hex[:10]}"
        emb = compute_embedding(text)
        rec = MemoryRecord(
            memory_id=mem_id,
            text=text,
            embedding=emb,
            date_written=date_written,
            layer=layer,
            importance=importance,
            delta=0  # Recency resets to 1.0 (delta = 0) on creation
        )
        self.layers[layer].append(rec)
        return rec

    def retrieve_top_k(self, query_text: str, k_per_layer: int = RETRIEVAL_TOP_K_PER_LAYER) -> list[dict]:
        """
        Retrieves top k_per_layer records per layer independently by compound score.
        Returns up to 4 * k_per_layer records total.
        """
        query_emb = compute_embedding(query_text)
        retrieved = []

        for layer_name, records in self.layers.items():
            scored_records = []
            for r in records:
                score = compute_compound_score(
                    query_embedding=query_emb,
                    memory_embedding=r.embedding,
                    importance=r.importance,
                    delta=r.delta,
                    layer=layer_name
                )
                scored_records.append((score, r))
            
            # Sort descending by compound score
            scored_records.sort(key=lambda x: x[0], reverse=True)
            top_k = scored_records[:k_per_layer]
            
            for score, r in top_k:
                d = r.to_dict()
                d["compound_score"] = score
                retrieved.append(d)

        return retrieved

    def step_daily_decay(self):
        """
        Increments delta by 1 for all memories and applies layer-specific daily importance decay.
        """
        for layer_name, records in self.layers.items():
            decay_factor = MEMORY_LAYERS[layer_name]["decay"]
            for r in records:
                r.delta += 1
                r.importance = float(r.importance * decay_factor)

    def apply_feedback(self, memory_ids: list[str], delta_importance: float):
        """
        Revises importance by delta_importance (+18 or -18) clamped to [0, 100].
        Applies only to memories whose ID is in memory_ids.
        """
        target_ids = set(memory_ids)
        for records in self.layers.values():
            for r in records:
                if r.memory_id in target_ids:
                    new_imp = r.importance + delta_importance
                    r.importance = float(np.clip(new_imp, 0.0, 100.0))

    def run_cleanup(self):
        """
        Deletes any memory in any layer whose importance < 5 OR recency < 0.05.
        """
        for layer_name in list(self.layers.keys()):
            cleaned = []
            for r in self.layers[layer_name]:
                if r.importance >= CLEANUP_IMPORTANCE_MIN and r.get_recency() >= CLEANUP_RECENCY_MIN:
                    cleaned.append(r)
            self.layers[layer_name] = cleaned

    def get_all_records_count(self) -> int:
        return sum(len(recs) for recs in self.layers.values())

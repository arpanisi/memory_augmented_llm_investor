"""
Scoring and embedding utilities for the memory system.
Calculates Cosine Similarity, Recency score exp(-delta / Q), Importance decay, and Compound Retrieval Score.
"""
import numpy as np
from config.settings import MEMORY_LAYERS

# Open-weight, self-hosted BGE-family embedding model with ~1024-dim embeddings.
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024

_model_instance = None

def get_embedding_model():
    """
    Lazy-loads the BGE-family sentence-transformers embedding model or returns numpy fallback.
    """
    global _model_instance
    if _model_instance is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception:
            _model_instance = "fallback"
    return _model_instance

def compute_embedding(text: str) -> np.ndarray:
    """
    Computes a normalized embedding vector for the text.
    """
    model = get_embedding_model()
    if model != "fallback":
        emb = model.encode(text, convert_to_numpy=True)
        norm = np.linalg.norm(emb)
        return emb / max(norm, 1e-12)
    else:
        # Fallback deterministic pseudo-embedding for synthetic testing if torch/transformers unavailable.
        # Dimensionality matches the primary BGE model (~1024-dim).
        import hashlib
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        vec = rng.randn(EMBEDDING_DIM)
        return vec / np.linalg.norm(vec)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculates cosine similarity between two 1D numpy vectors.
    """
    v1 = np.asarray(vec1, dtype=float)
    v2 = np.asarray(vec2, dtype=float)
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))

def compute_recency_score(delta: int, layer: str) -> float:
    """
    recency = exp(-delta / Q_layer)
    """
    q_val = MEMORY_LAYERS[layer]["Q"]
    return float(np.exp(-float(delta) / q_val))

def compute_compound_score(query_embedding: np.ndarray, memory_embedding: np.ndarray, importance: float, delta: int, layer: str) -> float:
    """
    score = cosine_similarity(query_embedding, memory_embedding) + min(importance, 100)/100 + recency
    """
    sim = cosine_similarity(query_embedding, memory_embedding)
    imp_term = min(float(importance), 100.0) / 100.0
    rec = compute_recency_score(delta, layer)
    return sim + imp_term + rec

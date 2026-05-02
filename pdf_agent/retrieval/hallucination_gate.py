# retrieval/hallucination_gate.py
import re
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from models import RetrievalResult, RetrievalHit
from config import SIMILARITY_THRESHOLD, GATE1_REFUSE_MESSAGE_TEMPLATE

# New semantic threshold for context reuse (Phase 1 Fix)
CONTEXT_REUSE_THRESHOLD = 0.6 

@dataclass
class Gate1Decision:
    passed: bool
    reason: str
    best_distance: Optional[float]
    threshold: float
    hit_count: int
    top_hit_page: Optional[int] = None
    top_hit_section: Optional[str] = None
    source: str = "new_retrieval"
    reuse_confidence: float = 0.0

def _get_keyword_overlap(query: str, text: str) -> float:
    """Simple word-overlap heuristic with temporal/noise filtering."""
    q_words = set(re.findall(r'\w+', query.lower()))
    t_words = set(re.findall(r'\w+', text.lower()))
    
    # 1. Broad Stopwords
    stop_words = {
        'what', 'is', 'the', 'of', 'in', 'and', 'to', 'for', 'was', 'a', 'with', 'on',
        'who', 'are', 'how', 'which', 'about', 'did', 'said', 'say'
    }
    
    # 2. Temporal/Common Document Context (Noise for similarity-boosting)
    temporal = {
        'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december',
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        '2022', '2023', '2024', '2025'
    }
    
    q_keywords = {w for w in q_words if len(w) > 2 and w not in stop_words and w not in temporal}
    
    if not q_keywords:
        return 0.0
        
    overlap = q_keywords.intersection(t_words)
    return len(overlap) / len(q_keywords)

def evaluate_context_reuse(query: str, previous_hits: List[RetrievalHit]) -> dict:
    """
    Evaluates whether previous retrieval hits are semantically relevant to the current query.
    Phase 1 Fix: Grounded context reuse.
    """
    if not previous_hits:
        return {"reuse_context": False, "confidence": 0.0, "reason": "no_previous_hits"}

    from indexing.embedder import Embedder
    from indexing.vector_store import VectorStore

    try:
        # 1. Compute current query embedding
        embedder = Embedder()
        query_v = embedder.embed_query(query)

        # 2. Fetch previous chunk embeddings from Chroma
        store = VectorStore()
        ids = [h.chunk_id for h in previous_hits]
        res = store.collection.get(ids=ids, include=["embeddings"])
        
        # Note: mapping by ID because Chroma may return them in different order
        id_to_vector = {res["ids"][i]: res["embeddings"][i] for i in range(len(res["ids"]))}
        
        similarities = []
        for h in previous_hits:
            v_chunk = id_to_vector.get(h.chunk_id)
            if v_chunk is not None:
                # Cosine similarity for normalized vectors is just dot product
                sim = np.dot(query_v, v_chunk)
                similarities.append(sim)

        if not similarities:
            return {"reuse_context": False, "confidence": 0.0, "reason": "embeddings_fetch_failed"}

        # 3. Aggregate
        max_sim = max(similarities)
        avg_sim = sum(similarities) / len(similarities)

        # 4. Decision
        if max_sim >= CONTEXT_REUSE_THRESHOLD:
            return {
                "reuse_context": True,
                "confidence": max_sim,
                "reason": f"sufficient overlap (max_sim={max_sim:.4f}, avg_sim={avg_sim:.4f})",
                "avg_sim": avg_sim
            }
        
        return {
            "reuse_context": False,
            "confidence": max_sim,
            "reason": f"insufficient relevance (max_sim={max_sim:.4f} < {CONTEXT_REUSE_THRESHOLD})"
        }

    except Exception as e:
        print(f"[hallucination_gate] Context validation error: {e}")
        return {"reuse_context": False, "confidence": 0.0, "reason": f"error: {e}"}

def evaluate_retrieval_gate(result: RetrievalResult, is_continuation: bool = False) -> Gate1Decision:
    """
    Evaluates whether the retrieved chunks are relevant enough to answer the query.
    CRITICAL FIX: Continuation bypass removed. Revalidation expected at orchestration layer.
    """
    threshold = SIMILARITY_THRESHOLD
    
    # Continuation branch REMOVED entirely per security requirement.
    # Logic moved to evaluate_context_reuse and orchestration (chat_panel).

    # 1. Check if retrieval failed entirely
    if not result.success:
        return Gate1Decision(
            passed=False,
            reason=f"Retrieval error: {result.error}",
            best_distance=None,
            threshold=threshold,
            hit_count=0
        )

    # 2. Check for empty hits
    if not result.hits:
        return Gate1Decision(
            passed=False,
            reason="No relevant context found in the document.",
            best_distance=None,
            threshold=threshold,
            hit_count=0
        )

    # 3. Evaluate best hit distance
    best_hit = result.hits[0]
    best_dist = best_hit.distance
    
    # Hybrid logic - reward keyword overlap
    overlap_score = _get_keyword_overlap(result.query, best_hit.text)
    
    effective_threshold = threshold
    if overlap_score >= 0.4:
        effective_threshold += 0.15 
    elif overlap_score >= 0.2:
        effective_threshold += 0.05
    
    if best_dist > effective_threshold:
        reason = GATE1_REFUSE_MESSAGE_TEMPLATE.format(
            distance=f"{best_dist:.4f}",
            threshold=f"{effective_threshold:.4f}"
        )
        return Gate1Decision(
            passed=False,
            reason=reason,
            best_distance=best_dist,
            threshold=effective_threshold,
            hit_count=len(result.hits),
            top_hit_page=best_hit.page_start,
            top_hit_section=best_hit.section_title
        )

    # 4. Success Case
    return Gate1Decision(
        passed=True,
        reason="Retrieval confidence met.",
        best_distance=best_dist,
        threshold=threshold,
        hit_count=len(result.hits),
        top_hit_page=best_hit.page_start,
        top_hit_section=best_hit.section_title
    )

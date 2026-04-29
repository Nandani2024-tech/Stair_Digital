# retrieval/hallucination_gate.py
import re
from dataclasses import dataclass
from typing import Optional
from models import RetrievalResult
from config import SIMILARITY_THRESHOLD, GATE1_REFUSE_MESSAGE_TEMPLATE

@dataclass
class Gate1Decision:
    passed: bool
    reason: str
    best_distance: Optional[float]
    threshold: float
    hit_count: int
    top_hit_page: Optional[int]
    top_hit_section: Optional[str]

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
    # Exclude months and common years found in financial reports
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

def evaluate_retrieval_gate(result: RetrievalResult, is_continuation: bool = False) -> Gate1Decision:
    """
    Evaluates whether the retrieved chunks are relevant enough to answer the query.
    This acts as Phase 7 Gate 1: Hallucination Barrier.
    """
    threshold = SIMILARITY_THRESHOLD
    
    if is_continuation:
        best_hit = result.hits[0] if result.success and result.hits else None
        return Gate1Decision(
            passed=True,
            reason="context_continuation_safe",
            best_distance=getattr(best_hit, 'distance', 0.0) if best_hit else 0.0,
            threshold=threshold,
            hit_count=len(result.hits) if result.success and result.hits else 0,
            top_hit_page=getattr(best_hit, 'page_start', None) if best_hit else None,
            top_hit_section=getattr(best_hit, 'section_title', None) if best_hit else None
        )
    
    # 1. Check if retrieval failed entirely
    if not result.success:
        return Gate1Decision(
            passed=False,
            reason=f"Retrieval error: {result.error}",
            best_distance=None,
            threshold=threshold,
            hit_count=0,
            top_hit_page=None,
            top_hit_section=None
        )

    # 2. Check for empty hits
    if not result.hits:
        return Gate1Decision(
            passed=False,
            reason="No relevant context found in the document.",
            best_distance=None,
            threshold=threshold,
            hit_count=0,
            top_hit_page=None,
            top_hit_section=None
        )

    # 3. Evaluate best hit distance
    best_hit = result.hits[0]
    best_dist = best_hit.distance
    
    # NEW: Hybrid logic - reward keyword overlap
    # If we have > 50% keyword overlap, we are much more confident 
    # even if semantic distance is a bit high.
    overlap_score = _get_keyword_overlap(result.query, best_hit.text)
    
    effective_threshold = threshold
    if overlap_score >= 0.4:  # At least 40% keyword match
        effective_threshold += 0.15 # Relax threshold significantly
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

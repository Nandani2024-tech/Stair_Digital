from typing import List
from models import RetrievalHit
from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL

class Reranker:
    def __init__(self):
        try:
            self.model = CrossEncoder(RERANKER_MODEL)
        except Exception as e:
            print(f"[Reranker] Initialization Error: {e}")
            self.model = None
            
    def rerank(self, query: str, hits: List[RetrievalHit]) -> List[RetrievalHit]:
        if not hits or not self.model:
            return hits
            
        pairs = [(query, hit.text) for hit in hits]
        
        try:
            print(f"[reranker] Running CrossEncoder on {len(hits)} hits...")
            scores = self.model.predict(pairs)
            
            # Map scores to hits and sort
            scored_hits = list(zip(hits, scores))
            scored_hits.sort(key=lambda x: x[1], reverse=True)
            
            # Re-rank outputs
            reranked = [hit for hit, score in scored_hits]
            for rank, hit in enumerate(reranked):
                hit.rank = rank + 1  # update rank natively preserving metadata
                
            return reranked
        except Exception as e:
            print(f"[Reranker] Prediction Error: {e}")
            return hits

# retrieval/searcher.py
from typing import Optional
from models import RetrievalHit, RetrievalResult
from indexing.embedder import Embedder
from indexing.vector_store import VectorStore
from config import TOP_K_RETRIEVAL

def search_document(query: str, doc_id: str, top_k: Optional[int] = None) -> RetrievalResult:
    """
    Search for relevant chunks within a specific document.
    """
    if not query or not query.strip():
        return RetrievalResult(
            success=False, query=query, doc_id=doc_id, error="Query text is empty."
        )
    
    if not doc_id:
        return RetrievalResult(
            success=False, query=query, doc_id=None, error="No document ID provided."
        )

    if top_k is None:
        top_k = TOP_K_RETRIEVAL

    try:
        print(f"[searcher] Searching doc_id={doc_id} query={query!r} top_k={top_k}")
        
        # 1. Embed Query
        embedder = Embedder()
        query_vector = embedder.embed_query(query)

        # 2. Query Vector Store
        store = VectorStore()
        results = store.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where={"doc_id": doc_id},
            include=["documents", "metadatas", "distances"]
        )

        # 3. Parse Hits
        hits: list[RetrievalHit] = []
        
        # Chroma returns results as lists of lists (one list per query vector)
        res_ids       = results.get("ids", [[]])[0]
        res_docs      = results.get("documents", [[]])[0]
        res_metas     = results.get("metadatas", [[]])[0]
        res_distances = results.get("distances", [[]])[0]

        for i in range(len(res_ids)):
            meta = res_metas[i]
            hit = RetrievalHit(
                chunk_id=res_ids[i],
                doc_id=meta.get("doc_id"),
                text=res_docs[i],
                page_start=meta.get("page_start"),
                page_end=meta.get("page_end"),
                section_title=meta.get("section_title") or None,
                token_count=meta.get("token_count"),
                is_ocr_derived=meta.get("is_ocr_derived", False),
                distance=res_distances[i],
                rank=i + 1
            )
            hits.append(hit)

        print(f"[searcher] Retrieved {len(hits)} hit(s)")
        if hits:
            top = hits[0]
            print(f"[searcher] Top hit: page={top.page_start} section={top.section_title!r} dist={top.distance:.4f}")

        return RetrievalResult(
            success=True,
            query=query,
            doc_id=doc_id,
            hits=hits
        )

    except Exception as e:
        print(f"[searcher] ERROR: {e}")
        return RetrievalResult(
            success=False,
            query=query,
            doc_id=doc_id,
            error=f"Retrieval failed: {e}"
        )

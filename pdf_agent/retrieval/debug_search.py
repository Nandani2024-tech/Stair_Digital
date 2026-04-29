# retrieval/debug_search.py
import sys
import os

# Ensure root is in sys.path
sys.path.insert(0, os.getcwd())

from ingestion.pipeline import run_ingestion_pipeline
from indexing.index_builder import build_index
from retrieval.searcher import search_document
from retrieval.hallucination_gate import evaluate_retrieval_gate

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m retrieval.debug_search <path_or_doc_id> <query>")
        sys.exit(1)

    target = sys.argv[1]
    query  = sys.argv[2]
    
    doc_id = None

    # If it looks like a PDF path, ingest it first
    if target.lower().endswith(".pdf"):
        print(f"\n[debug_search] Processing PDF: {target}")
        ingest = run_ingestion_pipeline(target)
        if not ingest.success:
            print(f"[FAILED][INGEST] {ingest.error}")
            sys.exit(1)
        
        print(f"[INGEST OK] doc_id={ingest.doc_id}")
        
        print("[debug_search] Indexing...")
        indexed = build_index(ingest.chunks)
        if not indexed.success:
            print(f"[FAILED][INDEX] {indexed.error}")
            sys.exit(1)
        
        doc_id = ingest.doc_id
    else:
        # Assume it's a doc_id
        doc_id = target

    print(f"\n[debug_search] Searching doc_id={doc_id}")
    print(f"[debug_search] Query: {query!r}\n")

    result = search_document(query, doc_id)

    if not result.success:
        print(f"[FAILED][SEARCH] {result.error}")
        sys.exit(1)

    print(f"[SUCCESS] Found {len(result.hits)} hit(s)")

    # Phase 7: Gate 1
    gate = evaluate_retrieval_gate(result)
    if not gate.passed:
        print(f"\n[GATE1] BLOCKED")
        print(f"Reason: {gate.reason}")
        print(f"Best Dist: {gate.best_distance}")
        print(f"Threshold: {gate.threshold}\n")
    else:
        print(f"\n[GATE1] PASSED")
        print(f"Reason: {gate.reason}")
        print(f"Best Dist: {gate.best_distance:.4f}\n")

    print("-" * 40)
    for hit in result.hits:
        pages = f"p{hit.page_start}" if hit.page_start == hit.page_end else f"p{hit.page_start}-{hit.page_end}"
        section = hit.section_title or "(no section)"
        
        print(f"Rank {hit.rank} | Distance: {hit.distance:.4f} | Sim (Diag): {hit.diagnostic_similarity:.4f}")
        print(f"Location: {pages} | Section: {section}")
        print(f"Snippet: {hit.text[:200]!r}...")
        print("-" * 40)

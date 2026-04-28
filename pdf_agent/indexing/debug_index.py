# indexing/debug_index.py
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.getcwd())

from ingestion.pipeline import run_ingestion_pipeline
from indexing.index_builder import build_index
from indexing.vector_store import VectorStore

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m indexing.debug_index <pdf_path>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"\nRunning ingestion + indexing on: {path}\n")

    ingest = run_ingestion_pipeline(path)
    if not ingest.success:
        print(f"[FAILED][INGESTION] {ingest.error}")
        sys.exit(1)

    print(
        f"[INGESTION OK] doc_id={ingest.doc_id} "
        f"pages={ingest.page_count} chunks={ingest.chunk_count}"
    )

    indexed = build_index(ingest.chunks)
    if not indexed.success:
        print(f"[FAILED][INDEXING] {indexed.error}")
        sys.exit(1)

    print(
        f"[INDEXING OK] indexed={indexed.indexed_chunk_count} "
        f"deleted_old={indexed.deleted_existing_count} "
        f"collection_total={indexed.collection_count}"
    )

    store = VectorStore()
    sample = store.collection.get(
        where={"doc_id": ingest.doc_id},
        limit=1,
        include=["documents", "metadatas"]
    )

    ids = sample.get("ids", [])
    docs = sample.get("documents", [])
    metas = sample.get("metadatas", [])

    if ids:
        print("\n--- Sample stored record ---")
        print(f"ID:       {ids[0]}")
        print(f"Metadata: {metas[0]}")
        print(f"Text:     {docs[0][:120]!r}...")
    else:
        print("\n[WARN] No stored records found for sample fetch.")

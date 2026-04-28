# indexing/index_builder.py
from dataclasses import dataclass
from typing import Optional
from models import Chunk
from indexing.embedder import Embedder
from indexing.vector_store import VectorStore

@dataclass
class IndexingResult:
    success: bool
    doc_id: Optional[str]
    indexed_chunk_count: int
    deleted_existing_count: int
    collection_count: int
    error: Optional[str]

def build_index(chunks: list[Chunk]) -> IndexingResult:
    if not chunks:
        return IndexingResult(
            success=False,
            doc_id=None,
            indexed_chunk_count=0,
            deleted_existing_count=0,
            collection_count=0,
            error="No chunks provided for indexing.",
        )

    doc_ids = {c.doc_id for c in chunks}
    if len(doc_ids) != 1:
        return IndexingResult(
            success=False,
            doc_id=None,
            indexed_chunk_count=0,
            deleted_existing_count=0,
            collection_count=0,
            error=f"Chunks contain multiple doc_ids: {sorted(list(doc_ids))}",
        )

    doc_id = chunks[0].doc_id

    try:
        print(f"[index_builder] Starting indexing for doc_id={doc_id} with {len(chunks)} chunk(s).")
        embedder = Embedder()
        store = VectorStore()

        deleted_count = store.delete_document_chunks(doc_id)

        texts = [c.text for c in chunks]
        embeddings = embedder.embed_documents(texts)

        indexed_count = store.add_chunks(chunks, embeddings)
        collection_count = store.count()

        print(
            f"[index_builder] Indexing complete for doc_id={doc_id}. "
            f"indexed={indexed_count}, deleted_old={deleted_count}, collection_total={collection_count}"
        )

        return IndexingResult(
            success=True,
            doc_id=doc_id,
            indexed_chunk_count=indexed_count,
            deleted_existing_count=deleted_count,
            collection_count=collection_count,
            error=None,
        )
    except Exception as e:
        return IndexingResult(
            success=False,
            doc_id=doc_id,
            indexed_chunk_count=0,
            deleted_existing_count=0,
            collection_count=0,
            error=f"Indexing failed: {e}",
        )

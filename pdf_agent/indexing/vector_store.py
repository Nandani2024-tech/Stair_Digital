# indexing/vector_store.py
import chromadb
from models import Chunk
from config import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME
        )
        print(
            f"[vector_store] Collection ready: {CHROMA_COLLECTION_NAME!r} "
            f"at {CHROMA_DB_DIR}"
        )

    def get_or_create_collection(self):
        return self.collection

    def _chunk_to_metadata(self, chunk: Chunk) -> dict:
        return {
            "doc_id": chunk.doc_id,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "section_title": chunk.section_title or "",
            "token_count": chunk.token_count,
            "is_ocr_derived": chunk.is_ocr_derived,
        }

    def delete_document_chunks(self, doc_id: str) -> int:
        if not doc_id:
            return 0
        existing = self.collection.get(
            where={"doc_id": doc_id},
            include=["metadatas"]
        )
        ids = existing.get("ids", []) or []
        deleted_count = len(ids)
        if ids:
            self.collection.delete(ids=ids)
        print(f"[vector_store] Deleted {deleted_count} existing chunk(s) for doc_id={doc_id}")
        return deleted_count

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk/embedding count mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings."
            )

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [self._chunk_to_metadata(c) for c in chunks]

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        total = self.collection.count()
        print(f"[vector_store] Added {len(chunks)} chunk(s). Collection total={total}")
        return len(chunks)

    def count(self) -> int:
        return self.collection.count()

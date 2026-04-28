# indexing/embedder.py
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

class Embedder:
    def __init__(self):
        self.model_name = EMBEDDING_MODEL_NAME
        self._model = None

    def _get_model(self):
        if self._model is None:
            print(f"[embedder] Loading embedding model: {self.model_name}")
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                raise RuntimeError(f"Failed to load embedding model '{self.model_name}': {e}")
            print("[embedder] Model loaded successfully.")
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        print(f"[embedder] Embedding {len(texts)} document(s).")
        model = self._get_model()
        vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty.")
        model = self._get_model()
        vector = model.encode([text], convert_to_numpy=True, normalize_embeddings=False)
        return vector[0].tolist()

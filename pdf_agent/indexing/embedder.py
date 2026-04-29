# indexing/embedder.py
from config import EMBEDDING_MODEL_NAME

_model_cache: dict = {}

def _get_model(model_name: str):
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        print(f"[embedder] Loading embedding model: {model_name}")
        try:
            _model_cache[model_name] = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{model_name}': {e}")
        print("[embedder] Model loaded successfully.")
    return _model_cache[model_name]

class Embedder:
    def __init__(self):
        self.model_name = EMBEDDING_MODEL_NAME

    def _get_model(self):
        return _get_model(self.model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        print(f"[embedder] Embedding {len(texts)} document(s) with normalization.")
        model = self._get_model()
        vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty.")
        model = self._get_model()
        vector = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return vector[0].tolist()

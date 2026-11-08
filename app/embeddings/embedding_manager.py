"""Lazy SentenceTransformer wrapper with E5 query and passage prefixes."""
from typing import List, Optional


class EmbeddingManager:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self.dimension = 384 if "MiniLM-L6-v2" in model_name else (384 if "e5-small" in model_name else None)

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            dimension_fn = getattr(self._model, "get_embedding_dimension", None)
            if dimension_fn is None:  # compatibility with older sentence-transformers
                dimension_fn = self._model.get_sentence_embedding_dimension
            self.dimension = int(dimension_fn())
        return self._model

    def format_query(self, text: str) -> str:
        return text if "e5" not in self.model_name.lower() or text.lower().startswith("query: ") else f"query: {text}"

    def format_passage(self, text: str) -> str:
        return text if "e5" not in self.model_name.lower() or text.lower().startswith("passage: ") else f"passage: {text}"

    def encode_single_query(self, text: str) -> List[float]:
        return [float(x) for x in self._load().encode(self.format_query(text), normalize_embeddings=True)]

    def encode_passages(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        values = self._load().encode([self.format_passage(t) for t in texts], batch_size=self.batch_size, normalize_embeddings=True)
        return [[float(x) for x in row] for row in values]

"""Dense semantic retriever using SentenceTransformers and ChromaDB."""

import logging
from typing import List, Dict, Any, Optional
from app.embeddings.embedding_manager import EmbeddingManager
from app.storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Retrieves chunks by semantic vector similarity."""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        source_filter: Optional[str] = None,
        programme_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not query.strip() or self.vector_store.count() == 0:
            return []

        query_emb = self.embedding_manager.encode_single_query(query)
        hits = self.vector_store.query_dense(
            query_embedding=query_emb,
            top_k=top_k,
            source_filter=source_filter,
            programme_filter=programme_filter,
        )
        return hits

"""ChromaDB vector store wrapper with model compatibility checks, safe document replacement, and metadata filtering."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from app.ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Encapsulates ChromaDB persistent storage and enforces index metadata integrity."""

    def __init__(
        self,
        db_path: Path | str,
        collection_name: str = "module_handbooks",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dimension: int = 384,
    ):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.embedding_dimension = embedding_dimension

        self._client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        col = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": self.embedding_model_name,
                "embedding_dimension": self.embedding_dimension,
            },
        )
        # Check model compatibility if collection already has existing records
        existing_meta = col.metadata or {}
        if col.count() > 0:
            stored_model = existing_meta.get("embedding_model")
            stored_dim = existing_meta.get("embedding_dimension")
            if stored_model and stored_model != self.embedding_model_name:
                logger.warning(
                    f"Vector index was created with model '{stored_model}' but current configuration uses '{self.embedding_model_name}'. Incompatible embeddings may cause degraded retrieval. Run re-index to update."
                )
        return col

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> int:
        """Add chunks and embeddings in batches to the collection."""
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks provided but {len(embeddings)} embeddings given"
            )

        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # Chroma batch insertion
        batch_size = 250
        for i in range(0, len(ids), batch_size):
            end = i + batch_size
            self._collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end],
            )
        return len(chunks)

    def delete_by_source(self, source_filename: str) -> None:
        """Safely remove all chunks associated with a source PDF."""
        try:
            self._collection.delete(where={"source_filename": source_filename})
        except Exception as e:
            logger.debug(f"Chroma delete by source_filename failed: {e}")
        try:
            self._collection.delete(where={"source": source_filename})
        except Exception as e:
            logger.debug(f"Chroma delete by source failed: {e}")

    def query_dense(
        self,
        query_embedding: List[float],
        top_k: int = 6,
        source_filter: Optional[str] = None,
        programme_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k nearest chunks using cosine similarity."""
        total_count = self._collection.count()
        if total_count == 0:
            return []

        where_clause = None
        if source_filter:
            where_clause = {"source_filename": source_filter}
        elif programme_filter:
            where_clause = {"programme_name": programme_filter}

        n_results = min(top_k, total_count)
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_clause:
            kwargs["where"] = where_clause

        res = self._collection.query(**kwargs)

        hits = []
        if res["documents"] and res["documents"][0]:
            for doc, meta, dist in zip(
                res["documents"][0],
                res["metadatas"][0],
                res["distances"][0],
            ):
                # Cosine distance to similarity score
                similarity = max(0.0, round(1.0 - float(dist), 4))
                hits.append(
                    {
                        "chunk_id": meta.get("chunk_id", ""),
                        "text": doc,
                        "source": meta.get("source_filename", meta.get("source", "unknown")),
                        "source_filename": meta.get("source_filename", meta.get("source", "unknown")),
                        "page_number": int(meta.get("page_number", 0)),
                        "programme_name": meta.get("programme_name", ""),
                        "section_heading": meta.get("section_heading", ""),
                        "language": meta.get("language", "en"),
                        "score": similarity,
                        "retrieval_type": "dense",
                    }
                )
        return hits

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Retrieve all documents and metadatas for keyword/BM25 indexing."""
        total = self._collection.count()
        if total == 0:
            return []
        data = self._collection.get(include=["documents", "metadatas"])
        chunks = []
        for doc_id, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
            chunks.append(
                {
                    "chunk_id": doc_id,
                    "text": doc,
                    "source": meta.get("source_filename", meta.get("source", "unknown")),
                    "source_filename": meta.get("source_filename", meta.get("source", "unknown")),
                    "page_number": int(meta.get("page_number", 0)),
                    "programme_name": meta.get("programme_name", ""),
                    "section_heading": meta.get("section_heading", ""),
                    "language": meta.get("language", "en"),
                }
            )
        return chunks

    def count(self) -> int:
        return self._collection.count()

    def get_sources(self) -> List[str]:
        if self.count() == 0:
            return []
        meta = self._collection.get(include=["metadatas"]).get("metadatas", [])
        return sorted(list({m.get("source_filename", m.get("source", "")) for m in meta if m}))

    def clear(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._get_or_create_collection()

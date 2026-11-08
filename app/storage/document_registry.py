"""Persistent idempotency registry for ingested source documents."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, Optional


@dataclass
class IndexedDocumentEntry:
    filename: str
    file_hash: str
    total_pages: int
    num_chunks: int
    num_modules: int
    programme_name: str
    language: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int


class DocumentRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._entries: Dict[str, dict] = json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            self._entries = {}

    def should_reindex(self, filename, file_hash, embedding_model, chunk_size, chunk_overlap, force_reindex=False):
        if force_reindex:
            return True
        item = self._entries.get(filename)
        return not item or any(item.get(k) != v for k, v in {
            "file_hash": file_hash, "embedding_model": embedding_model,
            "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
        }.items())

    def record_document(self, **values):
        entry = IndexedDocumentEntry(**values)
        self._entries[entry.filename] = asdict(entry)
        self.path.write_text(json.dumps(self._entries, indent=2, ensure_ascii=False) + "\n")

    def get(self, filename: str) -> Optional[dict]:
        return self._entries.get(filename)

    def remove(self, filename: str):
        self._entries.pop(filename, None)
        self.path.write_text(json.dumps(self._entries, indent=2, ensure_ascii=False) + "\n")

"""Smoke test the local PDF to persistent vector index path without model downloads."""
from unittest.mock import patch
from app.cli import ingest
from app.storage.vector_store import VectorStore

class _FakeEmbeddings:
    dimension=3
    def encode_passages(self,texts): return [[1.0,0.0,0.0] for _ in texts]


def test_pdf_ingestion_writes_retrievable_chunks(tmp_path):
    from pathlib import Path
    pdf=Path(__file__).resolve().parents[1] / "data" / "visual_computing.pdf"
    db=tmp_path / "chroma"
    with patch("app.cli._components", return_value=(_FakeEmbeddings(),VectorStore(db,embedding_dimension=3))):
        ingest([pdf],db_dir=db)
    store=VectorStore(db,embedding_dimension=3)
    assert store.count() > 0
    assert store.get_sources() == ["visual_computing.pdf"]
    assert store.get_all_chunks()[0]["page_number"] >= 1

"""Boundary tests for page chunking and metadata preservation."""
from app.ingestion.chunker import DocumentChunker


def test_chunks_respect_size_and_keep_source_metadata():
    text = ("Algorithms and data structures are central to computer science. " * 20).strip()
    chunks = DocumentChunker(chunk_size=120, chunk_overlap=20, min_chunk_chars=10).chunk_page(
        text, "sample.pdf", "Computer Science MSc", 4, "doc-hash", "en"
    )
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 120 for chunk in chunks)
    assert all(chunk.metadata["source_filename"] == "sample.pdf" for chunk in chunks)
    assert all(chunk.page_number == 4 for chunk in chunks)

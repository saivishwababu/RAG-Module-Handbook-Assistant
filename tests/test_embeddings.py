"""Tests for embedding model loading, dimension verification, and prefix handling."""

import pytest
from app.embeddings.embedding_manager import EmbeddingManager


def test_embedding_dimension():
    """Verify default model has dimension 384 and produces valid float vectors."""
    manager = EmbeddingManager(model_name="sentence-transformers/all-MiniLM-L6-v2")
    assert manager.dimension == 384

    emb = manager.encode_single_query("Algorithms and Data Structures")
    assert len(emb) == 384
    assert all(isinstance(x, float) for x in emb)


def test_e5_prefixing():
    """Verify that E5 models format queries with 'query: ' and passages with 'passage: '."""
    e5_manager = EmbeddingManager(model_name="intfloat/multilingual-e5-small")
    assert e5_manager.format_query("Prerequisites for ML") == "query: Prerequisites for ML"
    assert e5_manager.format_passage("Course covers gradients") == "passage: Course covers gradients"

    # Avoid duplicate prefixing
    assert e5_manager.format_query("query: already prefixed") == "query: already prefixed"
    assert e5_manager.format_passage("passage: already prefixed") == "passage: already prefixed"


def test_batch_encoding_consistency():
    """Batch encoding must produce identical count and dimensions."""
    manager = EmbeddingManager(model_name="sentence-transformers/all-MiniLM-L6-v2")
    texts = ["Computer Graphics", "Information Systems", "Automated Reasoning"]
    embs = manager.encode_passages(texts)
    assert len(embs) == 3
    for e in embs:
        assert len(e) == 384

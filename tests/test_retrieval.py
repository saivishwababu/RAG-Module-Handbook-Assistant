"""Tests for dense, BM25, and hybrid reciprocal rank fusion retrieval."""

import pytest
from app.retrieval.keyword_retriever import KeywordRetriever, tokenize_text
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.hybrid_retriever import HybridRetriever


def test_bm25_exact_term_matching():
    """BM25 must accurately prioritize exact technical tokens and codes."""
    corpus = [
        {
            "chunk_id": "c1",
            "text": "General computer science elective covering software engineering and project management.",
            "source_filename": "informatik.pdf",
            "page_number": 10,
        },
        {
            "chunk_id": "c2",
            "text": "The course Audio-Visual Communication and Networks AVCN has 6 SWS and 9 ECTS.",
            "source_filename": "embedded-systems.pdf",
            "page_number": 5,
        },
        {
            "chunk_id": "c3",
            "text": "Algorithms and Data Structures AlgoDat covers graph algorithms and sorting.",
            "source_filename": "informatik.pdf",
            "page_number": 5,
        },
    ]

    retriever = KeywordRetriever(corpus)

    # Exact code query
    hits = retriever.retrieve("AVCN", top_k=2)
    assert len(hits) > 0
    assert hits[0]["chunk_id"] == "c2"

    # Exact German or English query
    hits_algodat = retriever.retrieve("AlgoDat 9 ECTS", top_k=2)
    assert len(hits_algodat) > 0
    assert hits_algodat[0]["chunk_id"] == "c3"


def test_bm25_source_filtering():
    """BM25 must respect source_filter argument."""
    corpus = [
        {
            "chunk_id": "c1",
            "text": "Machine Learning module description",
            "source_filename": "informatik.pdf",
            "page_number": 35,
        },
        {
            "chunk_id": "c2",
            "text": "Machine Learning in Visual Computing",
            "source_filename": "visual_computing.pdf",
            "page_number": 9,
        },
    ]
    retriever = KeywordRetriever(corpus)
    hits = retriever.retrieve("Machine Learning", top_k=5, source_filter="visual_computing.pdf")
    assert len(hits) == 1
    assert hits[0]["source_filename"] == "visual_computing.pdf"


def test_rrf_hybrid_fusion():
    """Hybrid RRF must combine dense and sparse rankings without score collapse."""

    class MockDenseRetriever:
        def retrieve(self, query, top_k=6, **kwargs):
            return [
                {
                    "chunk_id": "doc_dense_1",
                    "text": "Neural Networks deep learning",
                    "source": "informatik.pdf",
                    "page_number": 20,
                    "score": 0.88,
                },
                {
                    "chunk_id": "doc_shared",
                    "text": "Machine Learning foundations",
                    "source": "informatik.pdf",
                    "page_number": 35,
                    "score": 0.85,
                },
            ]

    class MockSparseRetriever:
        def retrieve(self, query, top_k=6, **kwargs):
            return [
                {
                    "chunk_id": "doc_shared",
                    "text": "Machine Learning foundations",
                    "source": "informatik.pdf",
                    "page_number": 35,
                    "score": 12.4,
                },
                {
                    "chunk_id": "doc_sparse_1",
                    "text": "Exact code ML prerequisite",
                    "source": "informatik.pdf",
                    "page_number": 36,
                    "score": 11.2,
                },
            ]

    hybrid = HybridRetriever(
        dense_retriever=MockDenseRetriever(),
        keyword_retriever=MockSparseRetriever(),
        rrf_k=60,
        dense_weight=0.6,
        bm25_weight=0.4,
    )

    results = hybrid.retrieve("Machine Learning", top_k=3)
    assert len(results) == 3
    # doc_shared appeared in BOTH dense (#2) and sparse (#1) -> should have highest fused score!
    assert results[0]["chunk_id"] == "doc_shared"

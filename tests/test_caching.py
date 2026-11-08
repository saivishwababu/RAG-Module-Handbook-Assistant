"""Tests for cache policies, eviction rules, semantic response caching, and guardrails."""

import time
import pytest
from app.caching.cache_policy import FIFOCache, LRUCache, LFUCache
from app.caching.retrieval_cache import RetrievalCache, make_retrieval_cache_key
from app.caching.semantic_cache import (
    SemanticResponseCache,
    is_factually_compatible,
    extract_critical_entities,
)


def test_fifo_cache_eviction():
    """FIFO must evict oldest inserted key when capacity is reached."""
    cache = FIFOCache(max_entries=2)
    cache.put("k1", "v1")
    cache.put("k2", "v2")
    cache.put("k3", "v3")  # Should evict k1

    assert cache.get("k1") is None
    assert cache.get("k2") == "v2"
    assert cache.get("k3") == "v3"
    assert cache.stats.evictions == 1


def test_lru_cache_eviction():
    """LRU must preserve recently accessed entry and evict least recently used."""
    cache = LRUCache(max_entries=2)
    cache.put("k1", "v1")
    cache.put("k2", "v2")
    # Access k1 to make it most recently used
    assert cache.get("k1") == "v1"
    cache.put("k3", "v3")  # Should evict k2 (not k1!)

    assert cache.get("k1") == "v1"
    assert cache.get("k2") is None
    assert cache.get("k3") == "v3"


def test_lfu_cache_eviction():
    """LFU must evict least frequently accessed entry."""
    cache = LFUCache(max_entries=2)
    cache.put("k1", "v1")
    cache.put("k2", "v2")
    # Access k1 twice
    cache.get("k1")
    cache.get("k1")
    cache.put("k3", "v3")  # Should evict k2 because freq(k2)=1 < freq(k1)=3

    assert cache.get("k1") == "v1"
    assert cache.get("k2") is None
    assert cache.get("k3") == "v3"


def test_retrieval_cache_invalidation():
    """Retrieval cache must invalidate all entries when corpus version changes."""
    cache = RetrievalCache(policy="LRU", max_entries=10)
    cache.put("query", [{"source": "test.pdf"}], top_k=6)
    assert cache.get("query", top_k=6) is not None

    # Change corpus version
    cache.set_corpus_version("v2_new_hash")
    assert cache.get("query", top_k=6) is None


def test_semantic_cache_factual_compatibility_guardrail():
    """Questions with differing numbers/credits must NOT share cached responses."""
    q_6ects = "Which modules have 6 ECTS?"
    q_9ects = "Which modules have 9 ECTS?"

    # Even if sentence embeddings are similar, entity numbers differ
    assert is_factually_compatible(q_6ects, q_9ects) is False

    # Identical entities should be compatible
    assert is_factually_compatible("Tell me about 9 ECTS courses", "List courses with 9 ECTS") is True


def test_semantic_cache_hit_and_eviction():
    """Semantic cache should match high-similarity queries and track avoided LLM calls."""
    cache = SemanticResponseCache(similarity_threshold=0.90, max_entries=5)

    q1 = "What are the entrance requirements for Algorithms and Data Structures?"
    emb1 = [0.1] * 384
    resp1 = "Requirements are C, C++, Java [Source: informatik.pdf, page 5]."

    cache.put(q1, emb1, resp1, [{"filename": "informatik.pdf", "page_number": 5}])

    # Exact query hit
    hit = cache.get(q1, emb1)
    assert hit is not None
    assert hit[0] == resp1
    assert cache.stats.hits == 1
    assert cache.stats.llm_requests_avoided == 1

    # Miss on distant query
    emb_distant = [-0.1] * 384
    miss = cache.get("Tell me about dining halls", emb_distant)
    assert miss is None
    assert cache.stats.misses == 1

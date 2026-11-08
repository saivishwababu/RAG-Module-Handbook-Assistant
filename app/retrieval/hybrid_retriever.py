"""Hybrid retriever combining Dense vector search and BM25 with Reciprocal Rank Fusion (RRF)."""

import logging
from typing import List, Dict, Any, Optional
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.keyword_retriever import KeywordRetriever
from app.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Executes dense semantic retrieval and sparse BM25 retrieval, fusing results
    using Reciprocal Rank Fusion (RRF) to avoid score-scale mismatch.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        keyword_retriever: KeywordRetriever,
        reranker: Optional[CrossEncoderReranker] = None,
        rrf_k: int = 60,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ):
        self.dense_retriever = dense_retriever
        self.keyword_retriever = keyword_retriever
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        source_filter: Optional[str] = None,
        programme_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid retrieval with RRF ranking and optional reranking."""
        candidate_k = max(top_k * 2, 10)

        # 1. Dense retrieval
        dense_hits = self.dense_retriever.retrieve(
            query=query,
            top_k=candidate_k,
            source_filter=source_filter,
            programme_filter=programme_filter,
        )

        # 2. Sparse / BM25 retrieval
        sparse_hits = self.keyword_retriever.retrieve(
            query=query,
            top_k=candidate_k,
            source_filter=source_filter,
            programme_filter=programme_filter,
        )

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        chunks_by_id: Dict[str, Dict[str, Any]] = {}
        dense_ranks: Dict[str, int] = {}
        bm25_ranks: Dict[str, int] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            cid = hit.get("chunk_id") or f"{hit['source']}__p{hit['page_number']}"
            dense_ranks[cid] = rank
            chunks_by_id[cid] = hit
            score = self.dense_weight / (self.rrf_k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score

        for rank, hit in enumerate(sparse_hits, start=1):
            cid = hit.get("chunk_id") or f"{hit['source']}__p{hit['page_number']}"
            bm25_ranks[cid] = rank
            if cid not in chunks_by_id:
                chunks_by_id[cid] = hit
            score = self.bm25_weight / (self.rrf_k + rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + score

        # Assemble fused items
        fused: List[Dict[str, Any]] = []
        for cid, score in rrf_scores.items():
            item = dict(chunks_by_id[cid])
            item["score"] = round(score, 5)
            item["dense_rank"] = dense_ranks.get(cid)
            item["bm25_rank"] = bm25_ranks.get(cid)
            fused.append(item)

        # Sort by descending RRF score
        fused.sort(key=lambda x: x["score"], reverse=True)

        # 4. Optional Cross-Encoder Reranking
        if self.reranker and self.reranker.enabled:
            candidates = fused[:candidate_k]
            reranked = self.reranker.rerank(query=query, candidates=candidates, top_k=top_k)
            return reranked

        return fused[:top_k]

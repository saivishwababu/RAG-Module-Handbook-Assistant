"""Keyword / Sparse BM25 retriever for exact codes, acronyms, ECTS, and technical terms."""

import re
import logging
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Plus, BM25Okapi

logger = logging.getLogger(__name__)


def tokenize_text(text: str) -> List[str]:
    """Tokenize query and documents for BM25, supporting English, German, and alphanumeric codes."""
    if not text:
        return []
    # Tokenize preserving alphanumeric characters, dashes, and German umlauts
    tokens = re.findall(r"[A-Za-z0-9äöüÄÖÜß\-_]{2,}", text.lower())
    return tokens


class KeywordRetriever:
    """Sparse retriever implementing BM25Okapi."""

    def __init__(self, corpus: Optional[List[Dict[str, Any]]] = None):
        self.corpus: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        if corpus:
            self.index_corpus(corpus)

    def index_corpus(self, corpus: List[Dict[str, Any]]) -> None:
        """Build or refresh BM25 index over documents."""
        self.corpus = corpus
        if not corpus:
            self.bm25 = None
            return

        tokenized_corpus = [tokenize_text(doc.get("text", "")) for doc in corpus]
        self.bm25 = BM25Plus(tokenized_corpus)
        logger.debug(f"BM25Plus index built with {len(corpus)} documents.")

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        source_filter: Optional[str] = None,
        programme_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top matching chunks using BM25."""
        if not self.bm25 or not self.corpus or not query.strip():
            return []

        tokens = tokenize_text(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        # Filter and rank
        scored_docs = []
        for idx, score in enumerate(scores):
            if score <= 0.0:
                continue
            doc = self.corpus[idx]
            if source_filter and doc.get("source_filename") != source_filter:
                continue
            if programme_filter and doc.get("programme_name") != programme_filter:
                continue

            scored_docs.append(
                {
                    "chunk_id": doc.get("chunk_id", ""),
                    "text": doc.get("text", ""),
                    "source": doc.get("source_filename", doc.get("source", "unknown")),
                    "source_filename": doc.get("source_filename", doc.get("source", "unknown")),
                    "page_number": int(doc.get("page_number", 0)),
                    "programme_name": doc.get("programme_name", ""),
                    "section_heading": doc.get("section_heading", ""),
                    "language": doc.get("language", "en"),
                    "score": round(float(score), 4),
                    "retrieval_type": "keyword",
                }
            )

        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]

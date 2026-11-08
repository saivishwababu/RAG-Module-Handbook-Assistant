"""Optional Cross-Encoder reranking for hybrid retrieval candidates."""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks candidate chunks using a trained cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enabled: bool = False,
    ):
        self.model_name = model_name
        self.enabled = enabled
        self._model = None

    def _load_model(self):
        if self._model is None and self.enabled:
            try:
                from sentence_transformers import CrossEncoder

                logger.info(f"Loading cross-encoder reranker: {self.model_name}")
                self._model = CrossEncoder(self.model_name)
            except Exception as e:
                logger.warning(
                    f"Could not load cross-encoder reranker '{self.model_name}': {e}. Falling back to RRF ranking."
                )
                self.enabled = False

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 6,
    ) -> List[Dict[str, Any]]:
        """Rerank candidates based on cross-encoder scoring."""
        if not self.enabled or not candidates or not query.strip():
            return candidates[:top_k]

        self._load_model()
        if not self._model:
            return candidates[:top_k]

        pairs = [[query, c.get("text", "")] for c in candidates]
        try:
            scores = self._model.predict(pairs)
            for c, s in zip(candidates, scores):
                c["rerank_score"] = round(float(s), 4)

            ranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            return ranked[:top_k]
        except Exception as e:
            logger.warning(f"Reranking prediction failed: {e}. Returning unranked candidates.")
            return candidates[:top_k]

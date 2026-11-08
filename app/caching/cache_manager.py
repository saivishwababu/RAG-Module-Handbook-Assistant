"""Convenience facade for retrieval cache lifecycle."""
from app.caching.retrieval_cache import RetrievalCache

class CacheManager:
    def __init__(self, max_entries=512): self.retrieval=RetrievalCache(max_entries=max_entries)
    def invalidate(self, corpus_version): self.retrieval.set_corpus_version(corpus_version)

"""Retrieval cache keyed by query, filter and corpus version."""
import hashlib, json
from app.caching.cache_policy import LRUCache

def make_retrieval_cache_key(query, top_k=6, source_filter=None, programme_filter=None, corpus_version=""):
    body=json.dumps([query.strip().casefold(),top_k,source_filter,programme_filter,corpus_version],ensure_ascii=False,separators=(",",":"))
    return hashlib.sha256(body.encode()).hexdigest()

class RetrievalCache:
    def __init__(self, policy="LRU", max_entries=512):
        cls={"LRU":LRUCache}.get(policy.upper(),LRUCache); self._cache=cls(max_entries); self.corpus_version=""
    @property
    def stats(self): return self._cache.stats
    def set_corpus_version(self,version):
        if version != self.corpus_version: self.corpus_version=version; self._cache.clear()
    def get(self,query,top_k=6,source_filter=None,programme_filter=None):
        return self._cache.get(make_retrieval_cache_key(query,top_k,source_filter,programme_filter,self.corpus_version))
    def put(self,query,value,top_k=6,source_filter=None,programme_filter=None):
        self._cache.put(make_retrieval_cache_key(query,top_k,source_filter,programme_filter,self.corpus_version),value)

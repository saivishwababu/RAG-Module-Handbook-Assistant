"""Similarity based response cache with factual entity guardrails."""
import re
import numpy as np
from app.caching.cache_policy import LRUCache

def extract_critical_entities(text):
    return set(re.findall(r"\b\d+(?:\.\d+)?\b",text.lower()))

def is_factually_compatible(a,b):
    return extract_critical_entities(a)==extract_critical_entities(b)

class SemanticResponseCache:
    def __init__(self,similarity_threshold=0.92,max_entries=128):
        self.similarity_threshold=similarity_threshold; self._cache=LRUCache(max_entries); self._keys=[]
    @property
    def stats(self): return self._cache.stats
    def put(self,query,embedding,response,citations):
        key=str(len(self._keys)); self._keys.append(key)
        self._cache.put(key,(query,np.asarray(embedding,dtype=float),response,citations))
    def get(self,query,embedding):
        target=np.asarray(embedding,dtype=float); best=None; best_score=-1
        for key in list(self._cache._data.keys()):
            item=self._cache._data.get(key)
            if item is None: continue
            old_query,vector,response,citations=item
            if vector.shape != target.shape or not is_factually_compatible(query,old_query): continue
            denom=np.linalg.norm(vector)*np.linalg.norm(target)
            score=float(np.dot(vector,target)/denom) if denom else 0.0
            if score>best_score: best=(key,response,citations); best_score=score
        if best and best_score>=self.similarity_threshold:
            self._cache.get(best[0]); self._cache.stats.llm_requests_avoided+=1
            return best[1],best[2]
        self._cache.stats.misses+=1
        return None

"""Thread-safe bounded FIFO, LRU and LFU in-memory caches."""
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    llm_requests_avoided: int = 0

class _BaseCache:
    def __init__(self, max_entries=128):
        if max_entries < 1: raise ValueError("max_entries must be positive")
        self.max_entries=max_entries; self._data=OrderedDict(); self.stats=CacheStats(); self._lock=RLock()
    def get(self,key):
        with self._lock:
            if key not in self._data: self.stats.misses+=1; return None
            self.stats.hits+=1; value=self._data[key]; self._on_get(key); return value
    def put(self,key,value):
        with self._lock:
            if key in self._data: self._data[key]=value; self._on_update(key); return
            if len(self._data)>=self.max_entries: self._evict(); self.stats.evictions+=1
            self._data[key]=value; self._on_insert(key)
    def _on_get(self,key): pass
    def _on_update(self,key): pass
    def _on_insert(self,key): pass
    def _evict(self): self._data.popitem(last=False)
    def clear(self):
        with self._lock: self._data.clear()

class FIFOCache(_BaseCache): pass
class LRUCache(_BaseCache):
    def _on_get(self,key): self._data.move_to_end(key)
    def _on_update(self,key): self._data.move_to_end(key)
class LFUCache(_BaseCache):
    def __init__(self,max_entries=128): super().__init__(max_entries); self._counts={}
    def _on_insert(self,key): self._counts[key]=1
    def _on_get(self,key): self._counts[key]+=1
    def _on_update(self,key): self._counts[key]+=1
    def _evict(self):
        key=min(self._data, key=lambda k:self._counts[k]); self._data.pop(key); self._counts.pop(key,None)
    def clear(self): super().clear(); self._counts.clear()

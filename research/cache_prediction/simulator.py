"""Deterministic FIFO, LRU, LFU and score based cache simulations."""
from collections import OrderedDict
from dataclasses import dataclass
import time
import numpy as np
from research.cache_prediction.features import extract_causal_features_and_labels

@dataclass
class SimulationMetrics:
    policy: str
    hits: int
    misses: int
    evictions: int
    hit_ratio: float
    byte_hit_ratio: float
    total_latency_ms: float

class CacheSimulator:
    def __init__(self,capacity=25):
        if capacity<1: raise ValueError("capacity must be positive")
        self.capacity=capacity
    def _run(self,trace,policy,model=None):
        cache=OrderedDict(); freq={}; predicted_reuse={}; hits=misses=evictions=total_bytes=hit_bytes=0; start=time.perf_counter()
        causal_features = extract_causal_features_and_labels(trace)[0] if policy == "ML-Guided" and model is not None and trace else None
        for index,ev in enumerate(trace):
            total_bytes+=ev.entry_size_bytes
            if causal_features is not None:
                predicted_reuse[ev.query_id]=float(model.predict_proba(causal_features[index:index+1])[0][1])
            if ev.query_id in cache:
                hits+=1; hit_bytes+=ev.entry_size_bytes;
                if policy != "FIFO": cache.move_to_end(ev.query_id)
                freq[ev.query_id]=freq.get(ev.query_id,0)+1
            else:
                misses+=1
                if len(cache)>=self.capacity:
                    if policy=="LFU": key=min(cache,key=lambda k:(freq.get(k,0),list(cache).index(k)))
                    elif policy=="ML-Guided": key=min(cache,key=lambda k:(predicted_reuse.get(k,0.0),list(cache).index(k)))
                    else: key=next(iter(cache))
                    cache.pop(key); freq.pop(key,None); evictions+=1
                cache[ev.query_id]=ev.entry_size_bytes; freq[ev.query_id]=1
        n=len(trace); elapsed=(time.perf_counter()-start)*1000
        return SimulationMetrics(policy,hits,misses,evictions,hits/n if n else 0.,hit_bytes/total_bytes if total_bytes else 0.,elapsed)
    def run_fifo(self,trace): return self._run(trace,"FIFO")
    def run_lru(self,trace): return self._run(trace,"LRU")
    def run_lfu(self,trace): return self._run(trace,"LFU")
    def run_ml_guided(self,trace,model,features=None): return self._run(trace,"ML-Guided",model)

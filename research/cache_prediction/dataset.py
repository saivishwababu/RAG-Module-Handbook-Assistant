"""Synthetic reproducible request traces for cache policy evaluation."""
from dataclasses import dataclass
import random

@dataclass(frozen=True)
class CacheTraceEvent:
    query_id: str
    timestamp: float
    query_text: str
    entry_size_bytes: int

def generate_zipfian_trace(num_requests=1000,seed=42,num_queries=100):
    rng=random.Random(seed); count=max(1,num_queries)
    weights=[1/(rank+1) for rank in range(count)]
    ids=[f"q{i}" for i in range(count)]
    now=0.; result=[]
    for _ in range(max(0,num_requests)):
        now+=rng.expovariate(1/2)
        qid=rng.choices(ids,weights=weights,k=1)[0]
        text=f"Synthetic query {qid}"
        result.append(CacheTraceEvent(qid,now,text,rng.randint(256,4096)))
    return result

"""Evaluate and benchmark all cache policies (FIFO, LRU, LFU, ML-Guided) on identical workloads."""

from typing import List, Dict, Any
from research.cache_prediction.dataset import generate_zipfian_trace
from research.cache_prediction.simulator import CacheSimulator, SimulationMetrics
from research.cache_prediction.train import train_cache_predictor


def run_cache_prediction_experiment(
    capacity: int = 25,
    num_requests: int = 1200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute end-to-end comparative experiment across eviction policies."""
    # 1. Train ML model on independent prior trace
    model, train_metrics = train_cache_predictor(num_requests=1000, seed=seed)

    # 2. Generate unseen evaluation trace
    eval_trace = generate_zipfian_trace(num_requests=num_requests, seed=seed + 99)

    # 3. Simulate all policies on identical evaluation trace
    sim = CacheSimulator(capacity=capacity)
    fifo_res = sim.run_fifo(eval_trace)
    lru_res = sim.run_lru(eval_trace)
    lfu_res = sim.run_lfu(eval_trace)
    ml_res = sim.run_ml_guided(eval_trace, model, None)

    results = {
        "workload": {
            "capacity": capacity,
            "total_requests": num_requests,
            "unique_queries": len(set(ev.query_id for ev in eval_trace)),
        },
        "ml_model_metrics": train_metrics,
        "policy_comparison": [
            {
                "policy": fifo_res.policy,
                "hit_ratio": fifo_res.hit_ratio,
                "byte_hit_ratio": fifo_res.byte_hit_ratio,
                "hits": fifo_res.hits,
                "misses": fifo_res.misses,
                "evictions": fifo_res.evictions,
                "simulation_time_ms": fifo_res.total_latency_ms,
            },
            {
                "policy": lru_res.policy,
                "hit_ratio": lru_res.hit_ratio,
                "byte_hit_ratio": lru_res.byte_hit_ratio,
                "hits": lru_res.hits,
                "misses": lru_res.misses,
                "evictions": lru_res.evictions,
                "simulation_time_ms": lru_res.total_latency_ms,
            },
            {
                "policy": lfu_res.policy,
                "hit_ratio": lfu_res.hit_ratio,
                "byte_hit_ratio": lfu_res.byte_hit_ratio,
                "hits": lfu_res.hits,
                "misses": lfu_res.misses,
                "evictions": lfu_res.evictions,
                "simulation_time_ms": lfu_res.total_latency_ms,
            },
            {
                "policy": ml_res.policy,
                "hit_ratio": ml_res.hit_ratio,
                "byte_hit_ratio": ml_res.byte_hit_ratio,
                "hits": ml_res.hits,
                "misses": ml_res.misses,
                "evictions": ml_res.evictions,
                "simulation_time_ms": ml_res.total_latency_ms,
            },
        ],
    }
    return results

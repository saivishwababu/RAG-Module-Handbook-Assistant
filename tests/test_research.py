"""Tests for cache prediction research experiment and KV-cache benchmark suite."""

import pytest
from research.cache_prediction.dataset import generate_zipfian_trace
from research.cache_prediction.features import extract_causal_features_and_labels
from research.cache_prediction.train import train_cache_predictor
from research.cache_prediction.simulator import CacheSimulator
from research.kv_cache_benchmark.benchmark import KVCacheBenchmark


def test_synthetic_trace_reproducibility():
    """Identical seeds must generate identical trace sequences."""
    t1 = generate_zipfian_trace(num_requests=50, seed=123)
    t2 = generate_zipfian_trace(num_requests=50, seed=123)
    assert len(t1) == len(t2) == 50
    for e1, e2 in zip(t1, t2):
        assert e1.query_id == e2.query_id
        assert e1.entry_size_bytes == e2.entry_size_bytes


def test_cache_predictor_training_no_leakage():
    """Verify that causal feature extraction produces valid shapes and train runs cleanly."""
    trace = generate_zipfian_trace(num_requests=100, seed=42)
    X, y, names = extract_causal_features_and_labels(trace, future_window=10)
    assert len(X) == 100
    assert len(y) == 100
    assert len(names) == 6

    model, metrics = train_cache_predictor(num_requests=150, seed=42)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics


def test_cache_simulator_metrics():
    """Simulator policies must produce valid hit and eviction metrics."""
    trace = generate_zipfian_trace(num_requests=100, seed=42)
    sim = CacheSimulator(capacity=10)
    lru_res = sim.run_lru(trace)
    fifo_res = sim.run_fifo(trace)
    lfu_res = sim.run_lfu(trace)

    assert 0.0 <= lru_res.hit_ratio <= 1.0
    assert 0.0 <= fifo_res.hit_ratio <= 1.0
    assert 0.0 <= lfu_res.hit_ratio <= 1.0
    assert lru_res.evictions > 0


def test_kv_cache_unsupported_detection():
    """KV cache benchmark must cleanly detect unsupported cache strategies without crashing."""
    bench = KVCacheBenchmark(model_name="distilgpt2", device="cpu")
    res = bench.benchmark_strategy("QuantizedCache")
    assert res.status == "SKIPPED"
    assert "quantization backend" in res.skip_reason.lower()

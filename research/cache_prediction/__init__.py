"""Cache prediction research module exports."""

from research.cache_prediction.dataset import generate_zipfian_trace, CacheTraceEvent
from research.cache_prediction.features import extract_causal_features_and_labels
from research.cache_prediction.simulator import CacheSimulator, SimulationMetrics
from research.cache_prediction.train import train_cache_predictor
from research.cache_prediction.evaluate import run_cache_prediction_experiment

__all__ = [
    "generate_zipfian_trace",
    "CacheTraceEvent",
    "extract_causal_features_and_labels",
    "CacheSimulator",
    "SimulationMetrics",
    "train_cache_predictor",
    "run_cache_prediction_experiment",
]

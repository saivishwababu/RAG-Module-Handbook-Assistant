"""Transformer KV Cache Benchmarking suite for Small Language Models (SLMs)."""

import time
import os
import gc
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, StaticCache

logger = logging.getLogger(__name__)


@dataclass
class KVBenchmarkResult:
    cache_strategy: str
    model_name: str
    transformers_version: str
    device: str
    prompt_tokens: int
    generated_tokens: int
    init_overhead_ms: float
    prefill_latency_ms_median: float
    prefill_latency_ms_p95: float
    decode_throughput_tok_per_sec_median: float
    decode_throughput_tok_per_sec_p95: float
    total_latency_ms_median: float
    peak_memory_mb: float
    status: str  # "COMPLETED" or "SKIPPED"
    skip_reason: Optional[str] = None


class KVCacheBenchmark:
    """Systematic KV cache performance evaluator across cache architectures."""

    def __init__(
        self,
        model_name: str = "distilgpt2",
        device: Optional[str] = None,
        warmup_runs: int = 1,
        benchmark_runs: int = 3,
    ):
        self.model_name = model_name
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs

        self._tokenizer = None
        self._model = None

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading SLM '{self.model_name}' on {self.device}...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                local_files_only=True,
            ).to(self.device)
            self._model.eval()

    def _get_peak_memory_mb(self) -> float:
        """Measure peak memory on MPS / CUDA / CPU."""
        if self.device == "cuda" and torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        elif self.device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
            return round(torch.mps.current_allocated_memory() / (1024 * 1024), 2)
        return 0.0

    def benchmark_strategy(
        self,
        strategy_name: str,
        prompt: str = "The Faculty of Mathematics and Computer Science at Saarland University offers MSc programmes in",
        max_new_tokens: int = 32,
    ) -> KVBenchmarkResult:
        """Execute rigorous benchmarking for a single cache strategy."""
        try:
            self._load_model()
        except Exception as e:
            return KVBenchmarkResult(
                cache_strategy=strategy_name,
                model_name=self.model_name,
                transformers_version=transformers.__version__,
                device=self.device,
                prompt_tokens=0,
                generated_tokens=0,
                init_overhead_ms=0.0,
                prefill_latency_ms_median=0.0,
                prefill_latency_ms_p95=0.0,
                decode_throughput_tok_per_sec_median=0.0,
                decode_throughput_tok_per_sec_p95=0.0,
                total_latency_ms_median=0.0,
                peak_memory_mb=0.0,
                status="SKIPPED",
                skip_reason=f"Model loading failed: {e}",
            )

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        # Check strategy support and instantiate cache constructor
        def init_cache():
            t0 = time.perf_counter()
            if strategy_name.lower() == "dynamiccache":
                c = DynamicCache()
            elif strategy_name.lower() == "staticcache":
                # StaticCache requires max_cache_len and batch_size
                max_cache = prompt_len + max_new_tokens + 16
                c = StaticCache(
                    config=self._model.config,
                    max_batch_size=1,
                    max_cache_len=max_cache,
                    device=self.device,
                    dtype=torch.float32,
                )
            elif strategy_name.lower() == "quantizedcache":
                raise NotImplementedError(
                    "QuantizedCache requires specialized quantization backend ('quanto' or 'HQQ') not bundled by default."
                )
            elif strategy_name.lower() == "none (no kv-cache)":
                c = None
            else:
                raise ValueError(f"Unsupported cache strategy: {strategy_name}")
            init_time = (time.perf_counter() - t0) * 1000
            return c, init_time

        # Validate cache creation
        try:
            test_c, init_ms = init_cache()
        except NotImplementedError as e:
            return KVBenchmarkResult(
                cache_strategy=strategy_name,
                model_name=self.model_name,
                transformers_version=transformers.__version__,
                device=self.device,
                prompt_tokens=prompt_len,
                generated_tokens=0,
                init_overhead_ms=0.0,
                prefill_latency_ms_median=0.0,
                prefill_latency_ms_p95=0.0,
                decode_throughput_tok_per_sec_median=0.0,
                decode_throughput_tok_per_sec_p95=0.0,
                total_latency_ms_median=0.0,
                peak_memory_mb=0.0,
                status="SKIPPED",
                skip_reason=str(e),
            )
        except Exception as e:
            return KVBenchmarkResult(
                cache_strategy=strategy_name,
                model_name=self.model_name,
                transformers_version=transformers.__version__,
                device=self.device,
                prompt_tokens=prompt_len,
                generated_tokens=0,
                init_overhead_ms=0.0,
                prefill_latency_ms_median=0.0,
                prefill_latency_ms_p95=0.0,
                decode_throughput_tok_per_sec_median=0.0,
                decode_throughput_tok_per_sec_p95=0.0,
                total_latency_ms_median=0.0,
                peak_memory_mb=0.0,
                status="SKIPPED",
                skip_reason=f"Initialization error: {type(e).__name__}: {e}",
            )

        # 1. Warm-up runs
        for _ in range(self.warmup_runs):
            c, _ = init_cache()
            with torch.no_grad():
                if c is not None:
                    _ = self._model.generate(
                        **inputs,
                        max_new_tokens=min(10, max_new_tokens),
                        past_key_values=c,
                        use_cache=True,
                    )
                else:
                    _ = self._model.generate(
                        **inputs,
                        max_new_tokens=min(10, max_new_tokens),
                        use_cache=False,
                    )

        # 2. Benchmark iterations
        prefill_latencies = []
        decode_throughputs = []
        total_latencies = []
        generated_token_count = 0

        for _ in range(self.benchmark_runs):
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

            c, _ = init_cache()

            t_start = time.perf_counter()

            # Measure prefill step (first token generation)
            with torch.no_grad():
                out = self._model(input_ids, past_key_values=c, use_cache=(c is not None))
                next_token_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                t_first_token = time.perf_counter()
                prefill_ms = (t_first_token - t_start) * 1000

                # Decode subsequent tokens
                current_ids = next_token_id
                generated_ids = [next_token_id.item()]
                updated_cache = out.past_key_values

                for _ in range(max_new_tokens - 1):
                    if updated_cache is not None:
                        step_out = self._model(
                            current_ids,
                            past_key_values=updated_cache,
                            use_cache=True,
                        )
                    else:
                        full_ids = torch.cat(
                            [input_ids, torch.tensor([generated_ids], device=self.device)],
                            dim=-1,
                        )
                        step_out = self._model(full_ids, use_cache=False)

                    next_token = torch.argmax(step_out.logits[:, -1, :], dim=-1, keepdim=True)
                    generated_ids.append(next_token.item())
                    current_ids = next_token
                    updated_cache = step_out.past_key_values

            t_end = time.perf_counter()
            total_time_ms = (t_end - t_start) * 1000
            decode_time_sec = t_end - t_first_token
            num_decode_tokens = len(generated_ids) - 1
            throughput = (
                num_decode_tokens / decode_time_sec if decode_time_sec > 0 else 0.0
            )

            prefill_latencies.append(prefill_ms)
            decode_throughputs.append(throughput)
            total_latencies.append(total_time_ms)
            generated_token_count = len(generated_ids)

        import numpy as np

        return KVBenchmarkResult(
            cache_strategy=strategy_name,
            model_name=self.model_name,
            transformers_version=transformers.__version__,
            device=self.device,
            prompt_tokens=prompt_len,
            generated_tokens=generated_token_count,
            init_overhead_ms=round(init_ms, 3),
            prefill_latency_ms_median=round(float(np.median(prefill_latencies)), 2),
            prefill_latency_ms_p95=round(float(np.percentile(prefill_latencies, 95)), 2),
            decode_throughput_tok_per_sec_median=round(float(np.median(decode_throughputs)), 2),
            decode_throughput_tok_per_sec_p95=round(float(np.percentile(decode_throughputs, 95)), 2),
            total_latency_ms_median=round(float(np.median(total_latencies)), 2),
            peak_memory_mb=self._get_peak_memory_mb(),
            status="COMPLETED",
        )

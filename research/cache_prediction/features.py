"""Causal feature extraction for cache entry reuse prediction with strict temporal isolation."""

from typing import List, Dict, Any, Tuple
import numpy as np
from research.cache_prediction.dataset import CacheTraceEvent


def extract_causal_features_and_labels(
    trace: List[CacheTraceEvent],
    future_window: int = 30,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Construct feature vectors and future-reuse targets from a sequential trace.
    Guarantees no data leakage by computing features strictly from past events.
    """
    feature_names = [
        "past_frequency",
        "recency_delta_sec",
        "avg_inter_arrival_sec",
        "reuse_distance_past",
        "request_len_chars",
        "entry_size_bytes",
    ]

    history_times: Dict[str, List[float]] = {}
    history_keys_seen: List[str] = []

    X_list: List[List[float]] = []
    y_list: List[int] = []

    n = len(trace)

    for i, event in enumerate(trace):
        qid = event.query_id
        t = event.timestamp

        # 1. Past frequency
        past_times = history_times.get(qid, [])
        freq = float(len(past_times))

        # 2. Recency delta
        if past_times:
            recency = float(t - past_times[-1])
        else:
            recency = 9999.0

        # 3. Average inter-arrival time
        if len(past_times) >= 2:
            gaps = [past_times[j] - past_times[j - 1] for j in range(1, len(past_times))]
            avg_inter_arrival = float(np.mean(gaps))
        else:
            avg_inter_arrival = recency

        # 4. Past reuse distance: unique keys seen since last occurrence of this key
        if qid in history_keys_seen:
            last_pos = len(history_keys_seen) - 1 - history_keys_seen[::-1].index(qid)
            unique_since = len(set(history_keys_seen[last_pos + 1 :]))
            reuse_dist = float(unique_since)
        else:
            reuse_dist = 500.0

        # 5. Static request features
        req_len = float(len(event.query_text))
        entry_size = float(event.entry_size_bytes)

        feature_row = [
            freq,
            recency,
            avg_inter_arrival,
            reuse_dist,
            req_len,
            entry_size,
        ]

        # 6. Future label: will qid appear in the next `future_window` requests?
        future_slice = [trace[k].query_id for k in range(i + 1, min(i + 1 + future_window, n))]
        will_reuse = 1 if qid in future_slice else 0

        X_list.append(feature_row)
        y_list.append(will_reuse)

        # Update history
        if qid not in history_times:
            history_times[qid] = []
        history_times[qid].append(t)
        history_keys_seen.append(qid)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y, feature_names

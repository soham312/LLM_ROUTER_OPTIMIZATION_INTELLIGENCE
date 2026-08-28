"""
STAGE 9b: Pure data-shaping helpers for the Streamlit dashboard.

Kept separate from app.py (the rendering script) so they can be unit tested
directly with pytest, without needing a Streamlit script-run context.
"""

from collections import Counter, deque
from typing import Any, Dict, List

import pandas as pd


def moving_average(values: List[float], window: int) -> List[float]:
    """Smooths a noisy per-query metric into a rolling trend line."""
    result = []
    dq: deque = deque()
    running_sum = 0.0
    for v in values:
        dq.append(v)
        running_sum += v
        if len(dq) > window:
            running_sum -= dq.popleft()
        result.append(running_sum / len(dq))
    return result


def rolling_rate(flags: List[bool], window: int) -> List[float]:
    """Rolling fraction of True values - drives the escalation rate trend."""
    dq: deque = deque()
    running_sum = 0
    result = []
    for f in flags:
        val = 1 if f else 0
        dq.append(val)
        running_sum += val
        if len(dq) > window:
            running_sum -= dq.popleft()
        result.append(running_sum / len(dq))
    return result


def rolling_model_share(models: List[str], window: int) -> pd.DataFrame:
    """
    Rolling proportion of traffic each model received, over a sliding window.

    Why rolling instead of cumulative share? Cumulative share barely moves
    once thousands of queries have been logged, which would hide the bandit
    reacting to a mid-run distribution shift or shock (Stage 4/6). A sliding
    window keeps the chart responsive to what's happening right now.
    """
    unique_models = sorted(set(models))
    if not unique_models:
        return pd.DataFrame()

    dq: deque = deque()
    counts: Counter = Counter()
    rows = []
    for m in models:
        dq.append(m)
        counts[m] += 1
        if len(dq) > window:
            counts[dq.popleft()] -= 1
        total = len(dq)
        rows.append({um: counts[um] / total for um in unique_models})

    return pd.DataFrame(rows, index=range(1, len(models) + 1))


def compute_baseline_comparison(
    summary: Dict[str, Any],
    per_model_stats: Dict[str, Dict[str, Any]],
    raw_logs: List[Dict[str, Any]],
    baseline_models: List[str],
) -> pd.DataFrame:
    """
    Projects what a static "always route to model X" policy would have cost,
    scored, and taken in latency, using that model's own observed per-query
    averages from the same traffic - then lines it up against what the
    bandit router actually achieved in aggregate.

    Why project from observed per-model averages instead of re-running
    traffic once per baseline? The router already explores every arm (and
    always falls back to the fallback model on escalation), so each model's
    logged per-query averages are a fair estimate of what routing every
    query to it would have looked like, without paying for N extra
    simulation passes.
    """
    total_queries = summary.get("total_queries", 0)
    latencies = [log.get("actual_latency_ms", 0.0) for log in raw_logs]

    rows = [{
        "policy": "Router (actual)",
        "total_cost": summary.get("total_cost", 0.0),
        "avg_quality": summary.get("avg_judge_score", 0.0),
        "median_latency_ms": float(pd.Series(latencies).median()) if latencies else 0.0,
    }]

    for model in baseline_models:
        stats = per_model_stats.get(model)
        if not stats or stats["query_count"] == 0:
            # The bandit never explored this arm, so we have no observed
            # data to project a baseline from - surface that honestly
            # instead of guessing.
            rows.append({
                "policy": f"Always {model}",
                "total_cost": None,
                "avg_quality": None,
                "median_latency_ms": None,
            })
            continue

        avg_cost_per_query = stats["total_cost"] / stats["query_count"]
        rows.append({
            "policy": f"Always {model}",
            "total_cost": avg_cost_per_query * total_queries,
            "avg_quality": stats["avg_score"],
            "median_latency_ms": stats["p50_latency"],
        })

    return pd.DataFrame(rows).set_index("policy")

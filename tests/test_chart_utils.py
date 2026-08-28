import pandas as pd
import pytest

from dashboard.chart_utils import (
    compute_baseline_comparison,
    moving_average,
    rolling_model_share,
    rolling_rate,
)


def test_moving_average_expands_until_window_then_slides():
    # Window of 3 over [1, 2, 3, 4]:
    # step1: avg(1) = 1
    # step2: avg(1,2) = 1.5
    # step3: avg(1,2,3) = 2
    # step4: avg(2,3,4) = 3 (window is now full and slides)
    result = moving_average([1, 2, 3, 4], window=3)
    assert result == pytest.approx([1.0, 1.5, 2.0, 3.0])


def test_rolling_rate_tracks_escalation_fraction_in_window():
    # Window of 2 over [False, True, True, False]:
    # step1: [F] -> 0/1
    # step2: [F,T] -> 1/2
    # step3: [T,T] -> 2/2
    # step4: [T,F] -> 1/2
    result = rolling_rate([False, True, True, False], window=2)
    assert result == pytest.approx([0.0, 0.5, 1.0, 0.5])


def test_rolling_model_share_sums_to_one_each_step():
    models = ["a", "a", "b", "b", "b"]
    share_df = rolling_model_share(models, window=2)

    assert list(share_df.columns) == ["a", "b"]
    # Every row (regardless of window fill state) should be a valid
    # probability distribution over the models seen so far.
    for _, row in share_df.iterrows():
        assert row.sum() == pytest.approx(1.0)

    # Last window is ["b", "b"] -> 100% b
    assert share_df.iloc[-1]["b"] == pytest.approx(1.0)
    assert share_df.iloc[-1]["a"] == pytest.approx(0.0)


def test_rolling_model_share_empty_input_returns_empty_frame():
    assert rolling_model_share([], window=5).empty


def test_compute_baseline_comparison_projects_from_observed_averages():
    summary = {
        "total_queries": 10,
        "total_cost": 1.0,
        "avg_judge_score": 0.75,
    }
    per_model_stats = {
        "mistral": {"query_count": 4, "total_cost": 0.8, "avg_score": 0.9, "p50_latency": 300.0},
    }
    raw_logs = [{"actual_latency_ms": v} for v in [100.0, 200.0, 300.0]]

    comparison = compute_baseline_comparison(summary, per_model_stats, raw_logs, ["mistral"])

    router_row = comparison.loc["Router (actual)"]
    assert router_row["total_cost"] == pytest.approx(1.0)
    assert router_row["avg_quality"] == pytest.approx(0.75)
    assert router_row["median_latency_ms"] == pytest.approx(200.0)

    # mistral avg cost per query = 0.8 / 4 = 0.2, projected over 10 queries = 2.0
    baseline_row = comparison.loc["Always mistral"]
    assert baseline_row["total_cost"] == pytest.approx(2.0)
    assert baseline_row["avg_quality"] == pytest.approx(0.9)
    assert baseline_row["median_latency_ms"] == pytest.approx(300.0)


def test_compute_baseline_comparison_marks_unexplored_arms_as_missing():
    summary = {"total_queries": 5, "total_cost": 0.5, "avg_judge_score": 0.6}
    comparison = compute_baseline_comparison(summary, {}, [], ["phi3"])

    # pandas coerces None to NaN once the column holds other float values.
    baseline_row = comparison.loc["Always phi3"]
    assert pd.isna(baseline_row["total_cost"])
    assert pd.isna(baseline_row["avg_quality"])
    assert pd.isna(baseline_row["median_latency_ms"])

"""
STAGE 9b: Streamlit Dashboard UI.

Renders the metrics computed by the Stage 9a DashboardDataLayer as a live,
browsable dashboard: cost/quality/latency vs static baselines, how routing
decisions are distributed over time, and how often the router escalates.

Why Streamlit and not a hand-rolled Flask/JS app?
For a portfolio project, the visualization itself isn't the thing being
evaluated - the routing intelligence is. Streamlit turns the data layer's
already-tested aggregation functions into an interactive UI with a fraction
of the code a custom frontend would need, so the JSONL telemetry can speak
for itself.

Run with: streamlit run dashboard/app.py
"""

import time
from collections import Counter

import pandas as pd
import streamlit as st

from dashboard.chart_utils import (
    compute_baseline_comparison,
    moving_average,
    rolling_model_share,
    rolling_rate,
)
from dashboard.data_layer import DashboardDataLayer
from router.client import UnifiedLLMClient

DEFAULT_LOG_PATH = "observability/router_logs.jsonl"


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------

st.set_page_config(page_title="LLM Router Dashboard", layout="wide")
st.title("LLM Inference Router - Live Dashboard")
st.caption(
    "Cost, quality, and latency the bandit router actually achieved, "
    "compared against what a static 'always use model X' policy would cost."
)

with st.sidebar:
    st.header("Data Source")
    log_path = st.text_input("Telemetry log file", value=DEFAULT_LOG_PATH)
    window = st.slider("Rolling window (queries)", min_value=5, max_value=200, value=20, step=5)

    st.header("Baselines")
    # Order by price so the default comparison is the cheapest vs. most
    # expensive tier - the two ends of the cost/quality tradeoff the router
    # is meant to navigate.
    priced_models = sorted(
        UnifiedLLMClient.PRICING_PER_1M_TOKENS,
        key=lambda m: UnifiedLLMClient.PRICING_PER_1M_TOKENS[m],
    )
    cheapest, priciest = priced_models[0], priced_models[-1]
    baseline_models = st.multiselect(
        "Static policies to compare against",
        options=priced_models,
        default=[cheapest, priciest],
    )

    st.header("Live Updates")
    auto_refresh = st.checkbox("Auto-refresh", value=False)
    refresh_secs = st.slider("Refresh interval (s)", 2, 30, 5, disabled=not auto_refresh)
    st.button("Refresh now")

data_layer = DashboardDataLayer(log_path)
raw_logs = data_layer.load_raw_logs()

if not raw_logs:
    st.info(
        f"No telemetry found at `{log_path}` yet. Run the router (or "
        "`experiments/simulator.py`) with a `StructuredLogger` attached to "
        "start populating this dashboard."
    )
else:
    summary = data_layer.get_summary_metrics()
    per_model_stats = data_layer.get_per_model_stats()

    # --- KPI row -------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Queries", f"{summary['total_queries']:,}")
    k2.metric("Total Cost", f"${summary['total_cost']:.4f}")
    k3.metric("Avg Judge Score", f"{summary['avg_judge_score']:.2f}")
    k4.metric("Escalation Rate", f"{summary['overall_escalation_rate'] * 100:.1f}%")

    st.divider()

    # --- Cost / quality / latency vs. static baselines ------------------
    st.subheader("Router vs. Static Baselines")
    st.caption(
        "Baselines project what always routing every query to a single "
        "model would have cost, using that model's own observed per-query "
        "averages from this traffic."
    )
    comparison = compute_baseline_comparison(summary, per_model_stats, raw_logs, baseline_models)
    st.dataframe(
        comparison.style.format(
            {
                "total_cost": "${:.4f}",
                "avg_quality": "{:.2f}",
                "median_latency_ms": "{:.1f} ms",
            },
            na_rep="no data (arm never routed to)",
        ),
        width="stretch",
    )

    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        st.caption("Total Cost ($)")
        st.bar_chart(comparison["total_cost"].dropna())
    with bc2:
        st.caption("Avg Quality (judge score)")
        st.bar_chart(comparison["avg_quality"].dropna())
    with bc3:
        st.caption("Median Latency (ms)")
        st.bar_chart(comparison["median_latency_ms"].dropna())

    st.divider()

    # --- Live trends over time ------------------------------------------
    st.subheader("Cost / Quality / Latency Over Time")
    st.caption(f"Rolling average over the last {window} queries.")

    costs = [log.get("actual_cost", 0.0) for log in raw_logs]
    scores = [log.get("judge_score", 0.0) for log in raw_logs]
    lats = [log.get("actual_latency_ms", 0.0) for log in raw_logs]
    step_index = range(1, len(raw_logs) + 1)

    trend_df = pd.DataFrame(
        {
            "cost ($)": moving_average(costs, window),
            "quality (judge score)": moving_average(scores, window),
            "latency (ms)": moving_average(lats, window),
        },
        index=step_index,
    )

    tc1, tc2, tc3 = st.columns(3)
    tc1.line_chart(trend_df["cost ($)"])
    tc2.line_chart(trend_df["quality (judge score)"])
    tc3.line_chart(trend_df["latency (ms)"])

    st.divider()

    # --- Routing distribution over time ----------------------------------
    st.subheader("Routing Distribution Over Time")
    st.caption(f"Share of traffic per model, rolling over the last {window} queries.")
    models_over_time = [log.get("model_used", "unknown") for log in raw_logs]
    share_df = rolling_model_share(models_over_time, window)
    if not share_df.empty:
        st.area_chart(share_df)

    st.divider()

    # --- Escalation frequency ---------------------------------------------
    st.subheader("Escalation Frequency")
    esc_col1, esc_col2 = st.columns([2, 1])

    with esc_col1:
        escalated_flags = [bool(log.get("escalated")) for log in raw_logs]
        esc_rate_df = pd.DataFrame(
            {"escalation rate": rolling_rate(escalated_flags, window)},
            index=step_index,
        )
        st.caption(f"Rolling escalation rate over the last {window} queries.")
        st.line_chart(esc_rate_df)

    with esc_col2:
        reasons = Counter(
            log.get("escalation_reason") for log in raw_logs if log.get("escalated")
        )
        st.caption("Escalation reasons")
        if reasons:
            st.bar_chart(pd.Series(dict(reasons)))
        else:
            st.write("No escalations logged yet.")

    st.divider()

    with st.expander("Raw telemetry (most recent 50 events)"):
        st.dataframe(pd.DataFrame(raw_logs[-50:]), width="stretch")

# "Live" auto-refresh: rerun the whole script on a timer so the dashboard
# picks up newly appended JSONL lines without a manual reload. Kept as a
# plain sleep + rerun loop instead of a dependency (e.g. streamlit-autorefresh)
# to preserve the project's zero-extra-cost, minimal-dependency approach.
if auto_refresh:
    time.sleep(refresh_secs)
    st.rerun()

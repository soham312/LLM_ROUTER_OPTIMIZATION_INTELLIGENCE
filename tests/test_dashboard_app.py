import json

from streamlit.testing.v1 import AppTest


def _write_fixture_log(tmp_path):
    log_file = tmp_path / "app_logs.jsonl"
    events = [
        {"model_used": "mistral", "actual_cost": 0.05, "actual_latency_ms": 200,
         "judge_score": 0.9, "escalated": False, "escalation_reason": None},
        {"model_used": "llama3.2:1b", "actual_cost": 0.01, "actual_latency_ms": 50,
         "judge_score": 0.4, "escalated": True, "escalation_reason": "low_confidence"},
        {"model_used": "phi3", "actual_cost": 0.02, "actual_latency_ms": 90,
         "judge_score": 0.7, "escalated": False, "escalation_reason": None},
    ] * 5

    with open(log_file, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    return str(log_file)


def test_dashboard_renders_empty_state_without_error(tmp_path):
    # Point at a path that is guaranteed not to exist - the app should show
    # the "no telemetry yet" info message rather than crashing.
    missing_path = str(tmp_path / "does_not_exist.jsonl")

    at = AppTest.from_file("dashboard/app.py")
    at.run(timeout=30)
    at.text_input[0].set_value(missing_path).run(timeout=30)

    assert not at.exception
    assert len(at.info) == 1


def test_dashboard_renders_full_layout_with_data(tmp_path):
    log_path = _write_fixture_log(tmp_path)

    at = AppTest.from_file("dashboard/app.py")
    at.run(timeout=30)
    at.text_input[0].set_value(log_path).run(timeout=30)

    assert not at.exception
    # KPI row + baseline comparison table + charts should all be present.
    assert len(at.metric) == 4
    assert len(at.dataframe) >= 1

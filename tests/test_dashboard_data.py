import json
import pytest
from dashboard.data_layer import DashboardDataLayer

@pytest.fixture
def dummy_log_file(tmp_path):
    log_file = tmp_path / "test_dashboard_logs.jsonl"
    
    events = [
        {"model_used": "mistral", "actual_cost": 0.05, "actual_latency_ms": 200, "judge_score": 0.9, "escalated": False},
        {"model_used": "mistral", "actual_cost": 0.05, "actual_latency_ms": 300, "judge_score": 0.8, "escalated": False},
        {"model_used": "llama3", "actual_cost": 0.01, "actual_latency_ms": 50, "judge_score": 0.4, "escalated": True},
    ]
    
    with open(log_file, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
            
    return str(log_file)

def test_get_summary_metrics(dummy_log_file):
    data_layer = DashboardDataLayer(dummy_log_file)
    summary = data_layer.get_summary_metrics()
    
    assert summary["total_queries"] == 3
    assert summary["total_cost"] == pytest.approx(0.11)
    assert summary["avg_judge_score"] == pytest.approx((0.9 + 0.8 + 0.4) / 3)
    assert summary["overall_escalation_rate"] == pytest.approx(1 / 3)

def test_get_per_model_stats(dummy_log_file):
    data_layer = DashboardDataLayer(dummy_log_file)
    stats = data_layer.get_per_model_stats()
    
    assert "mistral" in stats
    assert "llama3" in stats
    
    mistral_stats = stats["mistral"]
    assert mistral_stats["query_count"] == 2
    assert mistral_stats["total_cost"] == pytest.approx(0.10)
    assert mistral_stats["escalation_rate"] == 0.0
    # p50 of 200 and 300 is 250
    assert mistral_stats["p50_latency"] == 250.0
    
    llama_stats = stats["llama3"]
    assert llama_stats["query_count"] == 1
    assert llama_stats["escalation_rate"] == 1.0

def test_empty_logs(tmp_path):
    empty_file = tmp_path / "empty.jsonl"
    empty_file.touch()
    
    data_layer = DashboardDataLayer(str(empty_file))
    summary = data_layer.get_summary_metrics()
    assert summary["total_queries"] == 0
    assert summary["total_cost"] == 0.0

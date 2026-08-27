import json
import pytest
from observability.metrics import SLATracker

def test_sla_tracker_processing():
    tracker = SLATracker(window_size=10)
    
    # Process some dummy events for model_a
    for i in range(100):
        tracker.process_event({
            "model_used": "model_a",
            "actual_latency_ms": 100.0 + i, # 100 to 199
            "escalated": i % 5 == 0 # 20% escalation rate
        })
        
    metrics = tracker.get_metrics()
    
    assert "model_a" in metrics
    # Because window size is 10, it only looks at the last 10 elements (i: 90 to 99)
    # Latencies: 190, 191, ..., 199
    # p50 of 190-199 is around 194.5
    assert 194.0 <= metrics["model_a"]["p50_latency_ms"] <= 195.0
    
    # Escalated elements in 90-99: 90, 95 -> 2 out of 10 -> 0.2 rate
    assert metrics["model_a"]["escalation_rate"] == 0.2

def test_sla_tracker_file_processing(tmp_path):
    log_file = tmp_path / "test_logs.jsonl"
    
    events = [
        {"model_used": "model_b", "actual_latency_ms": 50.0, "escalated": False},
        {"model_used": "model_b", "actual_latency_ms": 150.0, "escalated": True},
        {"model_used": "model_b", "actual_latency_ms": 100.0, "escalated": False}
    ]
    
    with open(log_file, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
            
    tracker = SLATracker()
    tracker.process_log_file(str(log_file))
    
    metrics = tracker.get_metrics()
    
    assert "model_b" in metrics
    # latencies: 50, 100, 150. p50 = 100
    assert metrics["model_b"]["p50_latency_ms"] == 100.0
    # 1 out of 3 escalated
    assert metrics["model_b"]["escalation_rate"] == pytest.approx(1.0 / 3.0)

import os
import json
import pytest
from observability.logger import StructuredLogger

def test_structured_logger_writes_jsonl(tmp_path):
    log_file = tmp_path / "test_router_logs.jsonl"
    logger = StructuredLogger(log_filepath=str(log_file))
    
    mock_routing_result = {
        "query": "What is 2+2?",
        "model_used": "llama3.2:1b",
        "bandit_expected_reward": 0.85,
        "bandit_uncertainty": 0.1,
        "judge_score": 0.9,
        "escalated": False,
        "escalation_reason": None,
        # A mocked response object as a dict
        "response": {
            "latency_ms": 150.5,
            "simulated_cost": 0.002
        }
    }
    
    logger.log_decision(mock_routing_result)
    
    assert log_file.exists()
    
    with open(log_file, "r") as f:
        lines = f.readlines()
        
    assert len(lines) == 1
    
    # Verify JSON parsing and contents
    logged_data = json.loads(lines[0])
    
    assert "timestamp" in logged_data
    assert logged_data["query"] == "What is 2+2?"
    assert logged_data["model_used"] == "llama3.2:1b"
    assert logged_data["bandit_expected_reward"] == 0.85
    assert logged_data["actual_latency_ms"] == 150.5
    assert logged_data["actual_cost"] == 0.002
    assert logged_data["escalated"] is False

import pytest
from unittest.mock import MagicMock
from observability.alerts import AlertManager

def test_alert_manager_no_violations():
    manager = AlertManager(max_p95_latency=100.0, max_escalation_rate=0.1)
    
    healthy_metrics = {
        "model_a": {
            "p95_latency_ms": 50.0,
            "escalation_rate": 0.05
        }
    }
    
    alerts = manager.check_metrics(healthy_metrics)
    assert len(alerts) == 0

def test_alert_manager_latency_violation():
    manager = AlertManager(max_p95_latency=100.0)
    
    degraded_metrics = {
        "model_a": {
            "p95_latency_ms": 150.0,
            "escalation_rate": 0.05
        }
    }
    
    alerts = manager.check_metrics(degraded_metrics)
    assert len(alerts) == 1
    assert alerts[0]["model"] == "model_a"
    assert alerts[0]["metric"] == "p95_latency_ms"

def test_alert_manager_fires_hook():
    mock_hook = MagicMock()
    manager = AlertManager(max_escalation_rate=0.2, alert_hook=mock_hook)
    
    failing_metrics = {
        "model_b": {
            "p95_latency_ms": 50.0,
            "escalation_rate": 0.5  # High escalation rate
        }
    }
    
    alerts = manager.check_metrics(failing_metrics)
    
    assert len(alerts) == 1
    mock_hook.assert_called_once_with(
        "model_b", 
        "escalation_rate", 
        "SLA Violation for model_b: escalation_rate is 0.50, exceeding threshold 0.20"
    )

def test_alert_manager_handles_hook_exception():
    # If the hook raises an exception, the manager should catch it and not crash.
    def failing_hook(model, metric, msg):
        raise ValueError("Simulated network failure")
        
    manager = AlertManager(max_escalation_rate=0.2, alert_hook=failing_hook)
    
    metrics = {"model_c": {"escalation_rate": 0.3}}
    
    # Should not raise exception
    alerts = manager.check_metrics(metrics)
    
    assert len(alerts) == 1

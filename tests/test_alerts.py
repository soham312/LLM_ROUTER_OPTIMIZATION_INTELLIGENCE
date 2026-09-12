import logging

import httpx
import pytest
from unittest.mock import MagicMock, patch

from observability.alerts import AlertManager, HTTPAlertHook

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


# --- HTTPAlertHook -------------------------------------------------------

def _mock_response(status_code=200, text="OK"):
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.text = text
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"{status_code} error", request=MagicMock(), response=response
        )
    else:
        response.raise_for_status.side_effect = None
    return response

def test_http_alert_hook_posts_the_correct_payload():
    hook = HTTPAlertHook(url="https://example.com/webhook", timeout=3.0)

    with patch("observability.alerts.httpx.post", return_value=_mock_response(200)) as mock_post:
        hook("model_a", "p95_latency_ms", "SLA Violation for model_a: p95_latency_ms is 999.00")

    mock_post.assert_called_once_with(
        "https://example.com/webhook",
        json={
            "model": "model_a",
            "metric": "p95_latency_ms",
            "message": "SLA Violation for model_a: p95_latency_ms is 999.00",
        },
        timeout=3.0,
    )

def test_http_alert_hook_timeout_is_caught_and_logged(caplog):
    hook = HTTPAlertHook(url="https://example.com/webhook", timeout=1.0)

    with patch("observability.alerts.httpx.post", side_effect=httpx.ReadTimeout("timed out")):
        with caplog.at_level(logging.ERROR):
            hook("model_a", "p95_latency_ms", "msg")  # must not raise

    assert "timed out" in caplog.text

def test_http_alert_hook_connection_error_is_caught_and_logged(caplog):
    hook = HTTPAlertHook(url="https://example.com/webhook")

    with patch("observability.alerts.httpx.post", side_effect=httpx.ConnectError("connection refused")):
        with caplog.at_level(logging.ERROR):
            hook("model_a", "p95_latency_ms", "msg")  # must not raise

    assert "connection refused" in caplog.text
    assert "ConnectError" in caplog.text

def test_http_alert_hook_non_2xx_is_caught_and_logged(caplog):
    hook = HTTPAlertHook(url="https://example.com/webhook")

    with patch("observability.alerts.httpx.post", return_value=_mock_response(503, text="Service Unavailable")):
        with caplog.at_level(logging.ERROR):
            hook("model_a", "p95_latency_ms", "msg")  # must not raise

    assert "503" in caplog.text
    assert "Service Unavailable" in caplog.text

def test_http_alert_hook_failures_are_distinguishable_not_collapsed():
    # A timeout and a connection error must not be logged identically -
    # otherwise there's no way to tell them apart after the fact.
    hook = HTTPAlertHook(url="https://example.com/webhook")

    with patch("observability.alerts.httpx.post", side_effect=httpx.ReadTimeout("timed out")):
        with patch("observability.alerts.logger") as timeout_logger:
            hook("model_a", "metric", "msg")
        timeout_message = timeout_logger.error.call_args[0][0]

    with patch("observability.alerts.httpx.post", side_effect=httpx.ConnectError("refused")):
        with patch("observability.alerts.logger") as conn_logger:
            hook("model_a", "metric", "msg")
        conn_message = conn_logger.error.call_args[0][0]

    with patch("observability.alerts.httpx.post", return_value=_mock_response(500, text="boom")):
        with patch("observability.alerts.logger") as status_logger:
            hook("model_a", "metric", "msg")
        status_message = status_logger.error.call_args[0][0]

    assert len({timeout_message, conn_message, status_message}) == 3
    assert "timed out" in timeout_message
    assert "ConnectError" in conn_message
    assert "500" in status_message

def test_http_alert_hook_wired_into_alert_manager_never_crashes_check_metrics():
    with patch("observability.alerts.httpx.post", side_effect=httpx.ConnectError("endpoint is down")):
        hook = HTTPAlertHook(url="https://example.com/webhook")
        manager = AlertManager(max_escalation_rate=0.2, alert_hook=hook)

        alerts = manager.check_metrics({"model_c": {"escalation_rate": 0.3}})

    # check_metrics completed normally and still reported the violation,
    # despite the webhook endpoint being completely unreachable.
    assert len(alerts) == 1
    assert alerts[0]["model"] == "model_c"

def test_http_alert_hook_success_does_not_log_an_error(caplog):
    hook = HTTPAlertHook(url="https://example.com/webhook")

    with patch("observability.alerts.httpx.post", return_value=_mock_response(200)):
        with caplog.at_level(logging.ERROR):
            hook("model_a", "metric", "msg")

    assert caplog.text == ""

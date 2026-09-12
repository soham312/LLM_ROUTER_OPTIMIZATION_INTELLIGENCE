import logging
from typing import Dict, List, Callable, Optional

import httpx

logger = logging.getLogger(__name__)


class HTTPAlertHook:
    """
    A callable alert_hook that POSTs each SLA violation as a JSON body
    ({"model", "metric", "message"}) to a configurable webhook URL - a
    PagerDuty/Slack incoming-webhook endpoint, or any HTTP endpoint that
    accepts a JSON body.

    Failures never propagate: a request timeout, a connection error
    (refused, DNS failure, TLS error, etc.), and a non-2xx response are
    each caught and logged with the specific cause, not collapsed into a
    single generic message. AlertManager.check_metrics calls this once
    per violation and must keep evaluating every model/metric even if
    the webhook endpoint is down or misconfigured - a dead alerting
    endpoint must never be able to crash metric checks.
    """

    def __init__(self, url: str, timeout: float = 5.0):
        self.url = url
        self.timeout = timeout

    def __call__(self, model: str, metric: str, message: str) -> None:
        payload = {"model": model, "metric": metric, "message": message}
        try:
            response = httpx.post(self.url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except httpx.TimeoutException as e:
            logger.error(f"Alert webhook to {self.url} timed out after {self.timeout}s: {e}")
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Alert webhook to {self.url} returned HTTP {e.response.status_code}: "
                f"{e.response.text[:200]!r}"
            )
        except httpx.RequestError as e:
            # Connection refused, DNS failure, TLS error, and any other
            # transport-level failure that isn't specifically a timeout.
            logger.error(f"Alert webhook to {self.url} failed: {type(e).__name__}: {e}")
        except Exception as e:
            # Not expected to be reachable given the above, but this hook
            # must never be the thing that crashes check_metrics.
            logger.error(f"Unexpected error posting alert webhook to {self.url}: {type(e).__name__}: {e}")


class AlertManager:
    """
    STAGE 8c: Alerting based on SLA Degradation.

    Monitors metrics produced by SLATracker. If latency or escalation
    rates exceed thresholds, it triggers alerts. `alert_hook` is any
    callable of (model, metric, message) -> None invoked once per
    violation; pass HTTPAlertHook(url) to fire a real webhook to
    PagerDuty, Slack, or any other HTTP endpoint, or supply your own
    callable for a different integration. Whatever the hook does,
    check_metrics guarantees it can't crash a metrics check: any
    exception the hook raises is caught and logged here too, on top of
    whatever HTTPAlertHook already isolates internally.
    """
    
    def __init__(self, 
                 max_p95_latency: float = 2000.0, 
                 max_p99_latency: float = 3000.0, 
                 max_escalation_rate: float = 0.2,
                 alert_hook: Optional[Callable[[str, str, str], None]] = None):
        
        self.thresholds = {
            "p95_latency_ms": max_p95_latency,
            "p99_latency_ms": max_p99_latency,
            "escalation_rate": max_escalation_rate
        }
        
        self.alert_hook = alert_hook
        self.active_alerts: List[Dict[str, str]] = []
        
    def check_metrics(self, current_metrics: Dict[str, Dict[str, float]]):
        """
        Evaluates SLA metrics for all models against defined thresholds.
        """
        self.active_alerts.clear()
        
        for model, metrics in current_metrics.items():
            for metric_name, threshold in self.thresholds.items():
                val = metrics.get(metric_name)
                
                if val is not None and val > threshold:
                    msg = f"SLA Violation for {model}: {metric_name} is {val:.2f}, exceeding threshold {threshold:.2f}"
                    
                    # Log as a CRITICAL error
                    logger.critical(msg)
                    
                    # Store alert
                    self.active_alerts.append({
                        "model": model,
                        "metric": metric_name,
                        "message": msg
                    })
                    
                    # Fire external webhook if provided
                    if self.alert_hook:
                        try:
                            self.alert_hook(model, metric_name, msg)
                        except Exception as e:
                            logger.error(f"Failed to fire alert hook: {e}")
                            
        return self.active_alerts

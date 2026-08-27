import logging
from typing import Dict, List, Callable, Optional

logger = logging.getLogger(__name__)

class AlertManager:
    """
    STAGE 8c: Alerting based on SLA Degradation.
    
    Monitors metrics produced by SLATracker. If latency or escalation rates 
    exceed thresholds, it triggers alerts. In production, this would fire 
    webhooks to PagerDuty or Slack.
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

import json
import logging
from typing import Dict, List, Any
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)

class SLATracker:
    """
    STAGE 8b: Per-model SLA Tracking.
    
    Why track rolling percentiles and escalation rates?
    In production, average (mean) latency is a deceptive metric because it hides 
    long-tail outliers. Tracking p95 and p99 latency gives a true picture of the 
    user experience. Tracking the escalation rate per model helps identify if a 
    specific model is suddenly degrading in quality or hallucinating more frequently, 
    triggering the judge's fallback.
    """
    
    def __init__(self, window_size: int = 1000):
        # We use a rolling window to ensure metrics reflect current reality, 
        # not the distant past.
        self.window_size = window_size
        
        # Deque for fast O(1) appends and pops on a rolling window
        self.latency_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.escalation_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        
    def process_event(self, event: Dict[str, Any]):
        """
        Processes a single telemetry event (parsed from JSONL).
        """
        model = event.get("model_used")
        if not model:
            return
            
        latency = event.get("actual_latency_ms", 0.0)
        escalated = 1 if event.get("escalated", False) else 0
        
        self.latency_history[model].append(latency)
        self.escalation_history[model].append(escalated)
        
    def process_log_file(self, filepath: str):
        """
        Loads and processes all events from a JSONL log file.
        """
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                        self.process_event(event)
                    except json.JSONDecodeError:
                        logger.warning("Skipped invalid JSON line in log file.")
        except FileNotFoundError:
            logger.error(f"Log file {filepath} not found.")

    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Computes rolling SLA metrics (p50, p95, p99, escalation rate) for all models.
        """
        metrics = {}
        
        for model in self.latency_history.keys():
            latencies = list(self.latency_history[model])
            escalations = list(self.escalation_history[model])
            
            if not latencies:
                continue
                
            metrics[model] = {
                "p50_latency_ms": float(np.percentile(latencies, 50)),
                "p95_latency_ms": float(np.percentile(latencies, 95)),
                "p99_latency_ms": float(np.percentile(latencies, 99)),
                "escalation_rate": sum(escalations) / len(escalations)
            }
            
        return metrics

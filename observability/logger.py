import json
import logging
import os
from typing import Dict, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class StructuredLogger:
    """
    STAGE 8a: Structured Logging for Observability.
    
    Why JSON Lines (JSONL)?
    In a production ML system, flat text logs are useless for analysis. 
    JSONL allows us to log every decision event (context, bandit internals, 
    judge feedback, latency, and cost) in a structured format. This makes it 
    trivial to ingest these logs into data warehouses (Snowflake, BigQuery) 
    or observability tools (Datadog, Grafana) to monitor distribution shifts 
    and model degradations in real-time.
    """
    
    def __init__(self, log_filepath: str = "observability/router_logs.jsonl"):
        self.log_filepath = log_filepath
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.log_filepath), exist_ok=True)
        
    def log_decision(self, routing_result: Dict[str, Any]):
        """
        Extracts relevant telemetry from the routing result and writes it as a JSON line.
        """
        # Extract components safely
        response_obj = routing_result.get("response")
        
        # Determine cost and latency
        latency = 0.0
        cost = 0.0
        if response_obj:
            # Check if it's an object with attributes or a dict
            if hasattr(response_obj, 'latency_ms'):
                latency = response_obj.latency_ms
                cost = response_obj.simulated_cost
            elif isinstance(response_obj, dict):
                latency = response_obj.get('latency_ms', 0.0)
                cost = response_obj.get('simulated_cost', 0.0)
        
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": routing_result.get("query", ""),
            "model_used": routing_result.get("model_used", ""),
            "bandit_expected_reward": float(routing_result.get("bandit_expected_reward", 0.0)),
            "bandit_uncertainty": float(routing_result.get("bandit_uncertainty", 0.0)),
            "actual_cost": float(cost),
            "actual_latency_ms": float(latency),
            "judge_score": float(routing_result.get("judge_score", 0.0)),
            "escalated": bool(routing_result.get("escalated", False)),
            "escalation_reason": routing_result.get("escalation_reason", None)
        }
        
        try:
            with open(self.log_filepath, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write structured log: {e}")

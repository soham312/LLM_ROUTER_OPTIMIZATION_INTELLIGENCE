import json
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict

class DashboardDataLayer:
    """
    STAGE 9a: Dashboard Data Aggregation.
    
    Parses the JSONL telemetry logs (Stage 8a) and aggregates key metrics 
    for cost, quality, latency, routing distribution, and escalations.
    This serves as the clean data backend for the upcoming Streamlit UI.
    """
    
    def __init__(self, log_filepath: str):
        self.log_filepath = log_filepath
        
    def load_raw_logs(self) -> List[Dict[str, Any]]:
        """Loads and parses the raw JSONL logs."""
        logs = []
        try:
            with open(self.log_filepath, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            logs.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            pass
        return logs

    def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Computes high-level aggregated metrics for the dashboard overview.
        """
        logs = self.load_raw_logs()
        if not logs:
            return {
                "total_queries": 0,
                "total_cost": 0.0,
                "avg_judge_score": 0.0,
                "overall_escalation_rate": 0.0
            }
            
        total_cost = sum(log.get("actual_cost", 0.0) for log in logs)
        avg_score = np.mean([log.get("judge_score", 0.0) for log in logs])
        escalations = sum(1 for log in logs if log.get("escalated"))
        
        return {
            "total_queries": len(logs),
            "total_cost": float(total_cost),
            "avg_judge_score": float(avg_score),
            "overall_escalation_rate": float(escalations / len(logs))
        }

    def get_per_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Computes detailed SLA and usage statistics segmented by model.
        """
        logs = self.load_raw_logs()
        stats = defaultdict(lambda: {
            "count": 0, 
            "latencies": [], 
            "escalations": 0,
            "cost": 0.0,
            "scores": []
        })
        
        for log in logs:
            model = log.get("model_used", "unknown")
            stats[model]["count"] += 1
            stats[model]["latencies"].append(log.get("actual_latency_ms", 0.0))
            stats[model]["cost"] += log.get("actual_cost", 0.0)
            stats[model]["scores"].append(log.get("judge_score", 0.0))
            if log.get("escalated"):
                stats[model]["escalations"] += 1
                
        # Finalize metrics
        final_stats = {}
        total_queries = len(logs) if logs else 1
        
        for model, data in stats.items():
            lats = data["latencies"]
            final_stats[model] = {
                "routing_percentage": float(data["count"] / total_queries) * 100,
                "query_count": data["count"],
                "total_cost": float(data["cost"]),
                "avg_score": float(np.mean(data["scores"])) if data["scores"] else 0.0,
                "p50_latency": float(np.percentile(lats, 50)) if lats else 0.0,
                "p95_latency": float(np.percentile(lats, 95)) if lats else 0.0,
                "escalation_rate": float(data["escalations"] / data["count"]) if data["count"] else 0.0
            }
            
        return final_stats

    def get_timeseries_data(self) -> Dict[str, List[float]]:
        """
        Extracts rolling/sequential data for plotting timeline charts.
        Since we might not have uniform timestamps in a mock, we return 
        sequential arrays.
        """
        logs = self.load_raw_logs()
        
        return {
            "costs": [log.get("actual_cost", 0.0) for log in logs],
            "scores": [log.get("judge_score", 0.0) for log in logs],
            "models": [log.get("model_used", "unknown") for log in logs]
        }

import random
import logging
from typing import List, Dict, Generator
from router.router_core import OptimizationRouter
from judge.judge import LLMJudge

logger = logging.getLogger(__name__)

# Sample query datasets representing different underlying topics
DATASETS = {
    "chat": [
        "How's the weather today?",
        "Tell me a joke.",
        "Write a polite email declining an invitation.",
        "What are some good hobbies to pick up?",
        "Summarize the plot of Hamlet."
    ],
    "math": [
        "What is the integral of x^2?",
        "Solve for x: 2x + 5 = 15",
        "Explain the Pythagorean theorem.",
        "What is the probability of rolling two sixes with fair dice?",
        "Calculate the eigenvalues of a 2x2 identity matrix."
    ],
    "code": [
        "Write a Python function to reverse a string.",
        "Explain what a closure is in JavaScript.",
        "How do I fix a NullPointerException in Java?",
        "Write a SQL query to join two tables.",
        "Implement binary search in C++."
    ]
}

class TrafficSimulator:
    """
    STAGE 6: Streaming traffic simulation with distribution shifts.
    
    Why this approach?
    Real production traffic is not an IID static batch. It exhibits 
    distribution shifts (e.g. users ask more coding questions during the week, 
    more chat/fun questions on weekends). Furthermore, models can experience 
    degradations (shocks). This simulator explicitly tests the bandit's 
    ability to adapt to these non-stationary realities over time.
    """
    
    def __init__(self, router: OptimizationRouter, judge: LLMJudge, datasets: Dict[str, List[str]] = None):
        self.router = router
        self.judge = judge
        self.datasets = datasets or DATASETS
        
    def _traffic_generator(self, 
                           n_queries: int, 
                           initial_distribution: Dict[str, float], 
                           shift_at_step: int, 
                           new_distribution: Dict[str, float]) -> Generator[str, None, None]:
        """
        Generates a stream of queries, applying a distribution shift at a specific step.
        """
        for step in range(n_queries):
            dist = initial_distribution if step < shift_at_step else new_distribution
            
            # Select topic based on distribution
            topics = list(dist.keys())
            probs = list(dist.values())
            chosen_topic = random.choices(topics, weights=probs, k=1)[0]
            
            # Sample a query from the chosen topic
            query = random.choice(self.datasets[chosen_topic])
            yield query

    def run_simulation(self, 
                       n_queries: int = 100, 
                       initial_dist: Dict[str, float] = None,
                       shift_at: int = 50,
                       new_dist: Dict[str, float] = None,
                       shock_model_at: int = -1,
                       shock_model: str = "llama3.2:1b",
                       shock_penalty: float = 0.5) -> List[Dict]:
        """
        Runs the simulation over a stream of traffic.
        Returns a log of results which can be used for offline validation (Stage 7).
        """
        if initial_dist is None:
            initial_dist = {"chat": 0.8, "math": 0.1, "code": 0.1}
        if new_dist is None:
            new_dist = {"chat": 0.1, "math": 0.1, "code": 0.8}
            
        logger.info(f"Starting simulation for {n_queries} queries (Mock mode: {self.router.client.mock_mode})...")
        results_log = []
        
        gen = self._traffic_generator(n_queries, initial_dist, shift_at, new_dist)
        
        for i, query in enumerate(gen):
            # Apply shock if requested
            if i == shock_model_at:
                logger.info(f"Step {i}: Simulating shock on {shock_model}")
                self.judge.set_shock(shock_model, shock_penalty)
                
            # Process query
            result = self.router.route_and_execute(query)
            
            # Add step information to result
            result["step"] = i
            result["topic_distribution"] = initial_dist if i < shift_at else new_dist
            
            results_log.append(result)
            
        return results_log

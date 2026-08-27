import pytest
from unittest.mock import MagicMock
from experiments.validation import ValidationHarness
from router.router_core import OptimizationRouter
from router.client import LLMResponse

def test_sequential_ab_test():
    # Mock router
    mock_router = MagicMock(spec=OptimizationRouter)
    mock_router.route_and_execute.return_value = {"final_reward": 0.8}
    
    # Mock client and judge for the baselines
    mock_router.client = MagicMock()
    mock_router.client.generate.return_value = LLMResponse(
        id="1", model="test", response_text="test", prompt_tokens=10, 
        completion_tokens=10, total_tokens=20, latency_ms=10.0, 
        simulated_cost=0.1, is_mock=True
    )
    
    mock_router.judge = MagicMock()
    # Return score 0.5
    mock_router.judge.evaluate.return_value = (0.5, {})
    
    queries = ["q1", "q2", "q3"]
    baselines = ["always-cheap"]
    
    results = ValidationHarness.sequential_ab_test(queries, mock_router, baselines)
    
    assert "router" in results
    assert "always-cheap" in results
    
    assert len(results["router"]) == 3
    assert len(results["always-cheap"]) == 3
    
    # Router gets 0.8 per step -> [0.8, 1.6, 2.4]
    assert results["router"][0] == 0.8
    assert results["router"][2] == pytest.approx(2.4)
    
    # Baseline gets 0.5 score - (0.1 cost * 0.1) = 0.49 per step -> [0.49, 0.98, 1.47]
    assert results["always-cheap"][0] == pytest.approx(0.49)
    assert results["always-cheap"][2] == pytest.approx(1.47)

def test_doubly_robust_ope():
    # Logged data
    logged_data = [
        {"model_used": "A", "final_reward": 1.0, "propensity": 0.5},
        {"model_used": "B", "final_reward": 0.0, "propensity": 0.5}
    ]
    
    # Target policy: always choose A
    def target_policy(step_data):
        return "A"
        
    # Reward estimator: predicts 0.8 for A, 0.2 for B
    def reward_estimator(step_data, arm):
        return 0.8 if arm == "A" else 0.2
        
    # Calculate expected DR for step 1 (target=A, log=A):
    # dr1 = 0.8 + (1.0 - 0.8) / 0.5 = 0.8 + 0.4 = 1.2
    
    # Calculate expected DR for step 2 (target=A, log=B):
    # dr2 = 0.8 (since target disagrees with log, we just use estimate)
    
    # Mean DR = (1.2 + 0.8) / 2 = 1.0
    
    dr_estimate = ValidationHarness.doubly_robust_off_policy_evaluation(
        logged_data, target_policy, reward_estimator
    )
    
    assert dr_estimate == pytest.approx(1.0)

def test_doubly_robust_empty_logs():
    assert ValidationHarness.doubly_robust_off_policy_evaluation([], lambda x: "A", lambda x, y: 0.0) == 0.0

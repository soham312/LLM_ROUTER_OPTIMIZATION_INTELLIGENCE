import pytest
from unittest.mock import MagicMock
from experiments.simulator import TrafficSimulator
from router.router_core import OptimizationRouter
from judge.judge import LLMJudge

def test_traffic_generator_distribution_shift():
    # Setup dummies
    simulator = TrafficSimulator(router=MagicMock(), judge=MagicMock())
    
    initial_dist = {"chat": 1.0, "math": 0.0, "code": 0.0}
    new_dist = {"chat": 0.0, "math": 1.0, "code": 0.0}
    
    gen = simulator._traffic_generator(
        n_queries=10, 
        initial_distribution=initial_dist, 
        shift_at_step=5, 
        new_distribution=new_dist
    )
    
    queries = list(gen)
    assert len(queries) == 10
    
    # First 5 should be chat queries
    for q in queries[:5]:
        assert q in simulator.datasets["chat"]
        
    # Last 5 should be math queries
    for q in queries[5:]:
        assert q in simulator.datasets["math"]

def test_run_simulation():
    # Mock router to return a dummy result
    mock_router = MagicMock(spec=OptimizationRouter)
    mock_router.client = MagicMock()
    mock_router.client.mock_mode = True
    def dummy_route(*args, **kwargs):
        return {
            "query": "dummy",
            "model_used": "mistral",
            "judge_score": 0.8,
            "final_reward": 0.7,
        }
    mock_router.route_and_execute.side_effect = dummy_route
    
    # Mock judge to observe shock
    mock_judge = MagicMock(spec=LLMJudge)
    
    simulator = TrafficSimulator(router=mock_router, judge=mock_judge)
    
    # Run a tiny simulation
    results = simulator.run_simulation(
        n_queries=4, 
        shift_at=2,
        shock_model_at=3,
        shock_model="model_a",
        shock_penalty=0.4
    )
    
    assert len(results) == 4
    
    # Verify shock was called at step 3
    mock_judge.set_shock.assert_called_once_with("model_a", 0.4)
    
    # Verify step metadata was injected
    assert results[0]["step"] == 0
    assert results[3]["step"] == 3
    
    # Verify router was called
    assert mock_router.route_and_execute.call_count == 4

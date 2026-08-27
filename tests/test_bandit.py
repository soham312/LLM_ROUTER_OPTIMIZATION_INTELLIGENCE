import numpy as np
import pytest
from router.bandit import LinUCBRouter

def test_linucb_initialization():
    models = ["model_a", "model_b"]
    bandit = LinUCBRouter(models=models, embedding_dim=10, alpha=1.0, gamma=0.9)
    
    assert list(bandit.A.keys()) == models
    assert bandit.A["model_a"].shape == (10, 10)
    assert bandit.b["model_a"].shape == (10,)
    assert np.allclose(bandit.A["model_a"], np.eye(10))

def test_linucb_update():
    bandit = LinUCBRouter(models=["model_a"], embedding_dim=2, alpha=1.0, gamma=1.0)
    context = np.array([1.0, 0.5])
    
    # Update with reward
    bandit.update("model_a", context, 1.0)
    
    # A = A + x*x.T
    expected_A = np.eye(2) + np.outer(context, context)
    assert np.allclose(bandit.A["model_a"], expected_A)
    
    # b = b + r*x
    expected_b = context * 1.0
    assert np.allclose(bandit.b["model_a"], expected_b)

def test_linucb_decay_and_adaptation():
    # We use a fast decay factor to make the test short
    bandit = LinUCBRouter(models=["model_a", "model_b"], embedding_dim=2, alpha=0.5, gamma=0.5)
    context = np.array([1.0, 0.0])
    
    # Initially give model_a high rewards
    for _ in range(5):
        bandit.update("model_a", context, 1.0)
        bandit.update("model_b", context, 0.1)
        
    best, exp_a, _ = bandit.select_model(context)
    assert best == "model_a"
    
    # Simulate a mid-run "shock": model_a suddenly gives 0 reward, model_b gives 1.0
    # Because gamma=0.5, past memory is quickly forgotten
    for _ in range(5):
        bandit.update("model_a", context, 0.0)
        bandit.update("model_b", context, 1.0)
        
    best, exp_b, _ = bandit.select_model(context)
    assert best == "model_b"

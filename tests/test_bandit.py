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

    best, exp_a, _, is_forced = bandit.select_model(context)
    assert best == "model_a"
    assert is_forced is False

    # Simulate a mid-run "shock": model_a suddenly gives 0 reward, model_b gives 1.0
    # Because gamma=0.5, past memory is quickly forgotten
    for _ in range(5):
        bandit.update("model_a", context, 0.0)
        bandit.update("model_b", context, 1.0)

    best, exp_b, _, is_forced = bandit.select_model(context)
    assert best == "model_b"


def test_linucb_warm_start_tries_every_arm_once_before_ucb():
    """
    A brand-new arm always has expected_reward == 0.0 (theta is the zero
    vector), so without a forced warm-start, no arm could ever earn its
    first real trial - the router's confidence gate would always escalate
    away from it. select_model must therefore try every arm once,
    unconditionally, flagged via is_forced_exploration, before UCB
    comparisons are trusted.
    """
    models = ["model_a", "model_b", "model_c"]
    bandit = LinUCBRouter(models=models, embedding_dim=2, alpha=1.0, gamma=0.99)
    context = np.array([1.0, 0.0])

    seen = []
    for _ in range(len(models)):
        model, expected_reward, _, is_forced = bandit.select_model(context)
        assert is_forced is True
        assert expected_reward == 0.0
        seen.append(model)
        bandit.update(model, context, reward=0.5)

    # Every arm was tried exactly once, in order.
    assert seen == models

    # Warm-start is now exhausted - the next pick is a genuine UCB comparison.
    _, _, _, is_forced = bandit.select_model(context)
    assert is_forced is False


def test_linucb_low_confidence_arm_can_still_be_updated_after_warm_start():
    """
    This is the deadlock the warm-start fix closes: previously, an arm with
    expected_reward stuck at 0.0 could never be selected long enough to earn
    a real update, because the router's confidence gate always routed
    around it. After warm-start, every arm has at least one real data point,
    so its expected_reward can move away from exactly zero.
    """
    bandit = LinUCBRouter(models=["only_arm"], embedding_dim=2, alpha=1.0, gamma=0.99)
    context = np.array([1.0, 0.0])

    model, expected_reward, _, is_forced = bandit.select_model(context)
    assert expected_reward == 0.0
    assert is_forced is True

    bandit.update(model, context, reward=1.0)

    _, expected_reward_after, _, is_forced_after = bandit.select_model(context)
    assert is_forced_after is False
    assert expected_reward_after > 0.0


def test_best_known_model_ties_break_toward_the_preferred_model():
    """
    Immediately after warm-start, every arm's theta is still built from a
    single data point (or, for arms excluded from a given comparison,
    genuinely nothing) - real ties are common early on. `prefer` gives a
    predictable, configured tie-break for that bootstrap phase, without
    ever overriding real evidence once arms actually differ.
    """
    bandit = LinUCBRouter(models=["a", "b", "c"], embedding_dim=2, alpha=1.0, gamma=0.99)
    context = np.array([1.0, 0.0])

    # All three arms are untouched - a clean three-way tie at 0.0.
    model, reward = bandit.best_known_model(context, prefer="b")
    assert model == "b"
    assert reward == 0.0

    # Without a preference, ties resolve to `models` order.
    model, _ = bandit.best_known_model(context)
    assert model == "a"

    # `exclude` removes a candidate even if it would have won.
    model, _ = bandit.best_known_model(context, exclude="a", prefer="a")
    assert model == "b"


def test_best_known_model_prefers_real_evidence_over_the_configured_preference():
    """
    The whole point of this method: once an arm's real track record is
    worse than another arm's, the configured `prefer` name must NOT win
    anymore - otherwise a degraded "trusted" model could never be escalated
    away from (see router_core's dynamic escalation-target fix).
    """
    bandit = LinUCBRouter(models=["a", "b"], embedding_dim=2, alpha=1.0, gamma=0.99)
    context = np.array([1.0, 0.0])

    bandit.update("a", context, reward=1.0)   # "a" now has a strong, real track record
    bandit.update("b", context, reward=0.0)   # "b" ("preferred") has a real, poor one

    model, reward_a = bandit.best_known_model(context, prefer="b")
    assert model == "a"

    _, reward_b = bandit.best_known_model(context, exclude="a")
    assert reward_a > reward_b


def test_best_known_model_returns_none_when_every_arm_is_excluded():
    bandit = LinUCBRouter(models=["only_arm"], embedding_dim=2, alpha=1.0, gamma=0.99)
    context = np.array([1.0, 0.0])

    model, reward = bandit.best_known_model(context, exclude="only_arm")
    assert model is None
    assert reward == 0.0

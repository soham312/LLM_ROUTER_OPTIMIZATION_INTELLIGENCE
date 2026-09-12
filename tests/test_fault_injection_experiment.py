import random
from unittest.mock import patch

import pytest

from experiments.fault_injection import DEFAULT_QUERIES, FaultInjectionExperiment, _execute_one, _tag_query
from judge.judge import LLMJudge
from router.bandit import LinUCBRouter
from router.client import UnifiedLLMClient
from router.embeddings import ContextEmbedder
from router.router_core import OptimizationRouter


def build_router(models):
    client = UnifiedLLMClient(mock_mode=True)
    embedder = ContextEmbedder(mock_mode=True)
    bandit = LinUCBRouter(models=models, embedding_dim=embedder.embedding_dim)
    judge = LLMJudge(client)
    return OptimizationRouter(
        client=client,
        embedder=embedder,
        bandit=bandit,
        judge=judge,
        fallback_model=models[-1],
    )


def picked_backend(router, query):
    context = router.embedder.get_embedding([{"role": "user", "content": query}])
    model, _, _, _ = router.bandit.select_model(context)
    return model


# --- _execute_one exercises the router's own route_and_execute() path ---

def test_no_fault_reports_success_direct():
    # Note: final_backend can still legitimately differ from
    # chosen_backend here even with no fault armed at all - the router's
    # own pre-existing low-judge-score escalation (step 6 of
    # route_and_execute) can swap models based purely on response
    # quality. "success_direct" only claims the *attempted* backend
    # never raised, not that no swap of any kind occurred.
    router = build_router(["llama3.2:1b", "mistral"])
    record = _execute_one(router, "hello", 0)

    assert record["outcome"] == "success_direct"
    assert record["succeeded"] is True
    assert record["failed_over"] is False
    assert record["primary_failure_reason"] is None

def test_hard_down_primary_reports_success_failover():
    router = build_router(["llama3.2:1b", "mistral"])
    chosen = picked_backend(router, "hello")
    router.client.inject_fault(chosen, hard_down=True)

    record = _execute_one(router, "hello", 0)

    assert record["chosen_backend"] == chosen
    assert record["outcome"] == "success_failover"
    assert record["succeeded"] is True
    assert record["failed_over"] is True
    assert record["final_backend"] != chosen
    assert record["primary_failure_reason"] is not None
    assert record["primary_failure_code"] is not None

def test_no_alternate_backend_when_pool_has_only_the_faulted_model():
    router = build_router(["mistral"])
    router.client.inject_fault("mistral", hard_down=True)

    record = _execute_one(router, "hello", 0)

    assert record["outcome"] == "failed_no_alternate_backend"
    assert record["succeeded"] is False
    assert record["failover_target"] is None

def test_failover_target_also_faulted_reports_failed_failover():
    router = build_router(["llama3.2:1b", "mistral"])
    chosen = picked_backend(router, "hello")
    other = "mistral" if chosen != "mistral" else "llama3.2:1b"
    router.client.inject_fault(chosen, hard_down=True)
    router.client.inject_fault(other, hard_down=True)

    record = _execute_one(router, "hello", 0)

    assert record["outcome"] == "failed_failover_target_error"
    assert record["succeeded"] is False
    assert record["failover_target"] == other
    assert record["failover_failure_reason"] is not None
    assert record["failover_failure_code"] is not None


# --- FaultInjectionExperiment --------------------------------------------

def test_experiment_totals_are_internally_consistent():
    router = build_router(list(UnifiedLLMClient.PRICING_PER_1M_TOKENS.keys()))
    experiment = FaultInjectionExperiment(
        router=router,
        faulted_backend="llama3.2:1b",
        n_requests=30,
        hard_down=True,
        seed=7,
    )

    summary = experiment.run()
    totals = summary["totals"]

    assert totals["total_requests"] == 30
    assert len(summary["records"]) == 30
    assert totals["successful"] + totals["failed"] == totals["total_requests"]
    assert (
        totals["successful_direct_no_fault_triggered"] + totals["successful_failovers"]
        == totals["successful"]
    )

def test_hard_down_with_multi_model_pool_always_fails_over_successfully():
    # With exactly one backend down out of several, every request routed
    # to it should find a healthy alternate.
    router = build_router(list(UnifiedLLMClient.PRICING_PER_1M_TOKENS.keys()))
    experiment = FaultInjectionExperiment(
        router=router,
        faulted_backend="llama3.2:1b",
        n_requests=25,
        hard_down=True,
        seed=3,
    )

    summary = experiment.run()

    for record in summary["records"]:
        if record["chosen_backend"] == "llama3.2:1b":
            assert record["outcome"] == "success_failover"
        else:
            assert record["outcome"] == "success_direct"
    assert summary["totals"]["failed"] == 0

def test_probability_fault_produces_a_mix_of_direct_and_failover_successes():
    # ContextEmbedder's mock embeddings are seeded from Python's str
    # hash(), which is randomized per-process (PYTHONHASHSEED) unless
    # disabled - so which default query the bandit would route to which
    # arm isn't reproducible across runs. Forcing select_model's choice
    # isolates what this test actually checks: that a 0.5 per-request
    # probability, applied many times to the same chosen backend via the
    # router's real route_and_execute() path, yields both outcomes.
    router = build_router(["llama3.2:1b", "mistral"])
    router.client.inject_fault("llama3.2:1b", probability=0.5)
    random.seed(123)

    with patch.object(
        router.bandit, "select_model", return_value=("llama3.2:1b", 0.0, 0.0, True)
    ):
        outcomes = [_execute_one(router, f"query {i}", i)["outcome"] for i in range(60)]

    assert "success_direct" in outcomes
    assert "success_failover" in outcomes

def test_experiment_rejects_backend_not_in_pool():
    router = build_router(["mistral", "phi3"])
    experiment = FaultInjectionExperiment(router=router, faulted_backend="not-a-model", n_requests=5)

    with pytest.raises(ValueError):
        experiment.run()

def test_fault_is_cleared_after_run_even_on_failure():
    router = build_router(["llama3.2:1b", "mistral"])
    experiment = FaultInjectionExperiment(
        router=router,
        faulted_backend="llama3.2:1b",
        n_requests=10,
        hard_down=True,
    )

    experiment.run()

    assert not router.client.is_faulted("llama3.2:1b")

def test_failure_cause_breakdown_never_collapses_distinct_reasons():
    router = build_router(["llama3.2:1b", "mistral"])
    experiment = FaultInjectionExperiment(
        router=router,
        faulted_backend="llama3.2:1b",
        n_requests=20,
        hard_down=True,
        error_types=["connection_refused", "timeout", "server_error"],
        seed=99,
    )

    summary = experiment.run()
    trigger_breakdown = summary["failure_cause_breakdown"]["primary_fault_trigger_reason"]

    valid_codes = {"connection_refused", "timeout", "http_500", "http_502", "http_503", "http_504"}
    for reason_code in trigger_breakdown:
        assert reason_code in valid_codes or reason_code.startswith("unclassified_error")

def test_tag_query_never_produces_the_same_string_twice_in_practice():
    # Not a formal uniqueness guarantee (it's a string format, not a
    # dedup set), but warmup/req prefixes plus a monotonic counter should
    # never collide across a realistic run size.
    tags = {_tag_query(DEFAULT_QUERIES[i % len(DEFAULT_QUERIES)], f"req {i}") for i in range(500)}
    assert len(tags) == 500

def test_no_two_requests_in_a_run_share_identical_query_text():
    # Regression guard for the plateau bug: ContextEmbedder's mock
    # embedding is deterministic per exact string, so identical literal
    # query text repeating is what let a bandit arm's comparison freeze
    # at a small, fixed set of contexts. Every request's actual query
    # text must now be unique.
    router = build_router(["llama3.2:1b", "mistral"])
    experiment = FaultInjectionExperiment(
        router=router,
        faulted_backend="mistral",
        n_requests=40,
        hard_down=True,
        seed=5,
    )

    summary = experiment.run()
    queries = [r["query"] for r in summary["records"]]

    assert len(queries) == len(set(queries)) == 40

def test_failover_count_scales_with_n_requests_instead_of_plateauing():
    # Regression guard for the exact bug reported: with mistral hard
    # down, successful failovers used to plateau near a small constant
    # regardless of n_requests, because the faulted arm's bandit state
    # froze while only ~10 distinct repeating query strings were ever
    # used. Re-embedding a fresh context per request should make the
    # failover count grow roughly with n_requests instead.
    small_router = build_router(list(UnifiedLLMClient.PRICING_PER_1M_TOKENS.keys()))
    small = FaultInjectionExperiment(
        router=small_router, faulted_backend="mistral", n_requests=40, hard_down=True, seed=42
    ).run()

    large_router = build_router(list(UnifiedLLMClient.PRICING_PER_1M_TOKENS.keys()))
    large = FaultInjectionExperiment(
        router=large_router, faulted_backend="mistral", n_requests=150, hard_down=True, seed=42
    ).run()

    small_failovers = small["totals"]["successful_failovers"]
    large_failovers = large["totals"]["successful_failovers"]

    # A plateau (the bug) would make these nearly equal regardless of the
    # 5x request-count difference; scaling should make the larger run's
    # count clearly, substantially larger.
    assert large_failovers > small_failovers * 2

def test_experiment_exercises_the_routers_own_failover_not_a_copy():
    """
    Confirms _execute_one drives OptimizationRouter.route_and_execute()
    itself (and therefore its bandit.update() side effects) rather than
    reimplementing routing/failover independently - regression guard for
    the refactor that moved failover logic out of this module.
    """
    router = build_router(["llama3.2:1b", "mistral"])
    with patch.object(router, "route_and_execute", wraps=router.route_and_execute) as spy:
        _execute_one(router, "hello", 0)
    spy.assert_called_once_with("hello")

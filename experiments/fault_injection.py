"""
Fault-injection / failover experiment.

Question this answers: when the router picks a backend that's down or
flaky, does it route around it - and when it doesn't, exactly why not?

Faults are synthetic and controlled, via UnifiedLLMClient.inject_fault
(see router/client.py). No real Ollama call is ever made here - mock
mode only - which is what makes failures reproducible.

The failover logic itself lives in OptimizationRouter.route_and_execute()
(router/router_core.py): a raised backend exception triggers exactly one
retry against the bandit's own best-known other arm, and if that also
fails, route_and_execute raises RoutingFailoverError with both the
primary and failover failure reasons preserved as attributes. This
experiment does not reimplement any of that - it just fires requests
through the real router and reads what it reports (result["model_used"],
result["backend_failed_over"], result["backend_failure_reason"], or the
raised RoutingFailoverError's attributes on total failure).

The bandit is warmed up (every arm exercised via ordinary, fault-free
route_and_execute calls) BEFORE any fault is injected, so select_model()
makes a real UCB-driven pick from the start of the faulted phase rather
than round-robining the pool in a fixed warm-start order (see
LinUCBRouter's docstring in router/bandit.py). Once the fault is armed,
requests go through the same route_and_execute() path production traffic
would - including its normal bandit.update() calls - so this measures the
real router's behavior, not an isolated slice of it. One consequence
worth knowing: route_and_execute never calls bandit.update() for a
backend that raised (there's no judge score to reward), so the bandit
has no signal telling it to route away from a hard-down backend - it
will keep getting selected at whatever rate it already was until
something else (a judge-score-based signal) changes that.

Every request is sent as a uniquely-tagged instance of one of the base
DEFAULT_QUERIES topics (see _tag_query), never the identical literal
string twice in a run. This matters more than it looks: with a hard-down
backend, its bandit arm is frozen (per the paragraph above) while every
other arm keeps getting updated - and ContextEmbedder's mock embedding
is deterministic per *exact* string (router/embeddings.py hashes the
literal text). Reusing a small fixed set of literal query strings means
the bandit only ever compares a tiny handful of fixed context vectors;
once a competing arm's shared linear model happens to drift past the
faulted arm's frozen score at one of those exact vectors, that context
can never route back to the faulted arm again for the rest of the run -
a one-way ratchet, one per distinct string, that makes the *count* of
observed failovers plateau at a small constant no matter how large
n_requests gets, long before the fault itself is actually "recovered
from" in any meaningful sense. Tagging every request with a unique
suffix keeps generating fresh context vectors instead, so the measured
failover rate reflects the fault_probability/hard_down configuration
sustained over the whole run, not an artifact of a handful of repeating
embeddings.
"""

import argparse
import json
import logging
import os
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from judge.judge import LLMJudge
from router.bandit import LinUCBRouter
from router.client import UnifiedLLMClient
from router.embeddings import ContextEmbedder
from router.router_core import OptimizationRouter, RoutingFailoverError

logger = logging.getLogger(__name__)

DEFAULT_QUERIES = [
    "How's the weather today?",
    "Tell me a joke.",
    "Write a polite email declining an invitation.",
    "What is the integral of x^2?",
    "Solve for x: 2x + 5 = 15",
    "Explain the Pythagorean theorem.",
    "Write a Python function to reverse a string.",
    "Explain what a closure is in JavaScript.",
    "How do I fix a NullPointerException in Java?",
    "Write a SQL query to join two tables.",
]


def _tag_query(base_query: str, tag: str) -> str:
    """
    Appends a unique tag to `base_query` so its embedding (see
    ContextEmbedder.get_embedding, which hashes the literal string) is
    distinct from every other request in the run - see the module
    docstring for why reusing identical text causes a false plateau in
    observed failover counts. "warmup"/"req" prefixes additionally keep
    the warm-up phase's tags from ever colliding with the main phase's.
    """
    return f"{base_query} [{tag}]"


def _warm_up_bandit(router: OptimizationRouter, queries: List[str]) -> None:
    """
    Exercises route_and_execute() with no fault active so every arm
    clears LinUCBRouter's forced warm-start phase before the experiment
    begins. Without this, select_model() would just return the first
    model in the pool, in a fixed order, for the whole run.
    """
    n_models = len(router.bandit.models)
    warm_up_rounds = router.bandit.min_pulls_before_ucb * n_models * 2
    for i in range(warm_up_rounds):
        base_query = queries[i % len(queries)]
        router.route_and_execute(_tag_query(base_query, f"warmup {i}"))


def _execute_one(router: OptimizationRouter, query: str, index: int) -> Dict[str, Any]:
    """
    Fires a single request through the real router and normalizes the
    outcome into one flat record. Every branch is explicit:

    - success_direct:              no fault triggered on the attempted backend
    - success_failover:             the attempted backend failed; the router's
                                     own failover to another arm succeeded
    - failed_no_alternate_backend:  the attempted backend failed and the
                                     bandit had no other arm to offer
    - failed_failover_target_error: the attempted backend failed, and the
                                     backend the router failed over to
                                     failed too
    """
    try:
        result = router.route_and_execute(query)
    except RoutingFailoverError as exc:
        return {
            "index": index,
            "query": query,
            "chosen_backend": exc.primary_model,
            "succeeded": False,
            "failed_over": False,
            "final_backend": None,
            "primary_failure_reason": exc.primary_reason,
            "primary_failure_code": exc.primary_code,
            "failover_target": exc.failover_model,
            "failover_failure_reason": exc.failover_reason,
            "failover_failure_code": exc.failover_code,
            "outcome": (
                "failed_no_alternate_backend"
                if exc.failover_model is None
                else "failed_failover_target_error"
            ),
        }

    failed_over = bool(result.get("backend_failed_over"))
    return {
        "index": index,
        "query": query,
        "chosen_backend": result["attempted_backend"],
        "succeeded": True,
        "failed_over": failed_over,
        "final_backend": result["model_used"],
        "primary_failure_reason": result.get("backend_failure_reason"),
        "primary_failure_code": result.get("backend_failure_code"),
        "failover_target": result["model_used"] if failed_over else None,
        "failover_failure_reason": None,
        "failover_failure_code": None,
        "outcome": "success_failover" if failed_over else "success_direct",
    }


@dataclass
class FaultInjectionExperiment:
    router: OptimizationRouter
    faulted_backend: str
    n_requests: int = 200
    fault_probability: float = 1.0
    hard_down: bool = False
    error_types: Optional[List[str]] = None
    queries: List[str] = field(default_factory=lambda: list(DEFAULT_QUERIES))
    seed: Optional[int] = 42

    def run(self) -> Dict[str, Any]:
        if self.faulted_backend not in self.router.bandit.models:
            raise ValueError(
                f"'{self.faulted_backend}' is not in the router's model pool "
                f"{self.router.bandit.models}"
            )

        if self.seed is not None:
            random.seed(self.seed)

        _warm_up_bandit(self.router, self.queries)

        self.router.client.inject_fault(
            self.faulted_backend,
            probability=self.fault_probability,
            hard_down=self.hard_down,
            error_types=tuple(self.error_types) if self.error_types else None,
        )

        records: List[Dict[str, Any]] = []
        try:
            for i in range(self.n_requests):
                base_query = self.queries[i % len(self.queries)]
                query = _tag_query(base_query, f"req {i}")
                records.append(_execute_one(self.router, query, i))
        finally:
            # Always disarm, even if something above raised, so a faulted
            # client is never accidentally left armed for later reuse.
            self.router.client.clear_fault(self.faulted_backend)

        return self._summarize(records)

    def _summarize(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(records)
        to_faulted = [r for r in records if r["chosen_backend"] == self.faulted_backend]
        to_faulted_succeeded_directly = [r for r in to_faulted if r["outcome"] == "success_direct"]

        successful = [r for r in records if r["succeeded"]]
        successful_direct = [r for r in successful if not r["failed_over"]]
        successful_failovers = [r for r in successful if r["failed_over"]]
        failed = [r for r in records if not r["succeeded"]]

        primary_failure_breakdown = Counter(
            r["primary_failure_code"] for r in records if r["primary_failure_code"]
        )
        failed_failover_outcome_breakdown = Counter(r["outcome"] for r in failed)
        failover_target_error_breakdown = Counter(
            r["failover_failure_code"] for r in failed if r["failover_failure_code"]
        )

        summary = {
            "run_config": {
                "faulted_backend": self.faulted_backend,
                "fault_probability": self.fault_probability,
                "hard_down": self.hard_down,
                "error_types": list(self.error_types) if self.error_types else "all",
                "n_requests": self.n_requests,
                "seed": self.seed,
                "model_pool": list(self.router.bandit.models),
            },
            "totals": {
                "total_requests": total,
                "requests_routed_to_faulted_backend": len(to_faulted),
                "requests_to_faulted_backend_that_succeeded_anyway": len(to_faulted_succeeded_directly),
                "successful": len(successful),
                "successful_direct_no_fault_triggered": len(successful_direct),
                "successful_failovers": len(successful_failovers),
                "failed": len(failed),
            },
            "failure_cause_breakdown": {
                "primary_fault_trigger_reason": dict(primary_failure_breakdown),
                "failed_failover_outcome": dict(failed_failover_outcome_breakdown),
                "failover_target_error_reason": dict(failover_target_error_breakdown),
            },
            "records": records,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        return summary


def build_default_router() -> OptimizationRouter:
    """Same mock-mode stack as experiments/simulator.py's __main__ block."""
    models = list(UnifiedLLMClient.PRICING_PER_1M_TOKENS.keys())
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fires N requests through the router with one backend faulted, and reports failover behavior."
    )
    parser.add_argument("--backend", default="llama3.2:1b", help="Backend to fault.")
    parser.add_argument("--n-requests", type=int, default=200)
    parser.add_argument(
        "--probability",
        type=float,
        default=1.0,
        help="Per-request failure probability for the faulted backend (ignored if --hard-down).",
    )
    parser.add_argument(
        "--hard-down",
        action="store_true",
        help="Backend fails on every call for the whole run, instead of probabilistically.",
    )
    parser.add_argument(
        "--error-types",
        default=None,
        help="Comma-separated subset of connection_refused,timeout,server_error. Default: all three.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="experiments/results/fault_injection_results.json",
        help="Where to write the results file (JSON, not git-ignored).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    error_types = [e.strip() for e in args.error_types.split(",")] if args.error_types else None

    router = build_default_router()
    experiment = FaultInjectionExperiment(
        router=router,
        faulted_backend=args.backend,
        n_requests=args.n_requests,
        fault_probability=args.probability,
        hard_down=args.hard_down,
        error_types=error_types,
        seed=args.seed,
    )
    summary = experiment.run()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    totals = summary["totals"]
    print(f"Faulted backend:        {args.backend} (hard_down={args.hard_down}, probability={args.probability})")
    print(f"Total requests:         {totals['total_requests']}")
    print(
        f"Successful:             {totals['successful']} "
        f"({totals['successful_direct_no_fault_triggered']} direct, "
        f"{totals['successful_failovers']} via failover)"
    )
    print(f"Failed:                 {totals['failed']}")
    print("Failure cause breakdown:")
    print(json.dumps(summary["failure_cause_breakdown"], indent=2))
    print(f"\nFull per-request results written to: {args.output}")


if __name__ == "__main__":
    main()

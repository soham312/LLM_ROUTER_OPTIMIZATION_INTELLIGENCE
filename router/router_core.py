import logging
from typing import Dict, Any, List, Optional
from router.client import UnifiedLLMClient, LLMResponse, backend_error_code, classify_backend_error
from router.embeddings import ContextEmbedder
from router.bandit import LinUCBRouter
from judge.judge import LLMJudge

from observability.logger import StructuredLogger

logger = logging.getLogger(__name__)


class RoutingFailoverError(RuntimeError):
    """
    Raised when route_and_execute's backend failover is exhausted: the
    initially selected backend raised, and either bandit.best_known_model()
    had no healthy alternate to offer, or that alternate raised too.

    Both the primary and failover failure reasons are preserved as
    attributes - never collapsed into just the last exception seen - so
    callers and logs can tell "the backend was down and there was nothing
    else to try" apart from "we tried a second backend and it failed for
    a different reason."
    """

    def __init__(
        self,
        primary_model: str,
        primary_exc: Exception,
        failover_model: Optional[str],
        failover_exc: Optional[Exception] = None,
    ):
        self.primary_model = primary_model
        self.primary_reason = classify_backend_error(primary_exc)
        self.primary_code = backend_error_code(primary_exc)
        self.failover_model = failover_model

        if failover_model is None:
            self.failover_reason = "no_alternate_backend_available"
            self.failover_code = "no_alternate_backend_available"
            message = (
                f"Backend '{primary_model}' failed ({self.primary_reason}) and no "
                f"healthy alternate backend was available to fail over to."
            )
        else:
            self.failover_reason = classify_backend_error(failover_exc)
            self.failover_code = backend_error_code(failover_exc)
            message = (
                f"Backend '{primary_model}' failed ({self.primary_reason}); "
                f"failover to '{failover_model}' also failed ({self.failover_reason})."
            )

        super().__init__(message)


class OptimizationRouter:
    """
    The Core Routing Intelligence (Phase 2).
    
    Ties together the client, embedding, contextual bandit, and LLM judge.
    Features:
    - Embedding-based context vectors (Stage 1/2)
    - LinUCB Contextual Bandit routing (Stage 3)
    - Non-stationary shock adaptation via bandit decay (Stage 4)
    - Fallback & escalation logic for low confidence/low quality (Stage 5)
    - Structured telemetry logging (Stage 8a)
    """
    
    def __init__(self, 
                 client: UnifiedLLMClient, 
                 embedder: ContextEmbedder, 
                 bandit: LinUCBRouter, 
                 judge: LLMJudge,
                 fallback_model: str = "mistral",
                 structured_logger: Optional[StructuredLogger] = None):
        self.client = client
        self.embedder = embedder
        self.bandit = bandit
        self.judge = judge
        self.structured_logger = structured_logger
        
        # The model to use if the bandit is unconfident or the initial model fails the judge
        self.fallback_model = fallback_model
        
        # Thresholds for escalation
        self.CONFIDENCE_THRESHOLD = 0.3  # Minimum expected reward to trust the bandit
        self.JUDGE_SCORE_THRESHOLD = 0.6 # Minimum score to accept a response

    def route_and_execute(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Routes the query, generates a response, evaluates it, and updates the bandit.
        """
        # 1. Prepare context
        history = conversation_history or []
        history_with_query = history + [{"role": "user", "content": query}]
        context_vector = self.embedder.get_embedding(history_with_query)
        
        # 2. Select model via Bandit
        selected_model, expected_reward, uncertainty, is_forced_exploration = self.bandit.select_model(context_vector)
        bandit_selected_model = selected_model  # the raw pick, preserved even if later overridden below

        escalation_triggered = False
        escalation_reason = None

        # 3. Fallback Logic: Low Confidence
        # `is_forced_exploration` picks are exempt: a genuinely untried arm
        # always starts at expected_reward == 0.0, so without this exemption
        # no non-fallback arm could ever clear this gate to earn its first
        # real trial (see LinUCBRouter's warm-start docstring).
        #
        # The escalation target is the bandit's own best-currently-known arm,
        # not a hardcoded fallback name - `fallback_model` is only used as a
        # tie-break preference for the (common, early-on) case where nothing
        # yet differentiates the candidates. This is what lets the system
        # route away from `fallback_model` itself if its own track record
        # has degraded (see README Section 11's second finding).
        if expected_reward < self.CONFIDENCE_THRESHOLD and not is_forced_exploration:
            escalation_target, _ = self.bandit.best_known_model(
                context_vector, exclude=selected_model, prefer=self.fallback_model
            )
            if escalation_target is not None:
                logger.warning(f"Escalation: Bandit confidence low (Expected Reward: {expected_reward:.2f}). Escalating to {escalation_target}.")
                selected_model = escalation_target
                escalation_triggered = True
                escalation_reason = "low_confidence"

        # 4. Generate Response, with a single backend-failover retry.
        #
        # This is a different failure mode from the confidence/judge-score
        # escalation above and below: those retry because the *quality* of
        # an answer looks bad, once a response already exists to judge.
        # This retries because client.generate() itself raised - a
        # connection refused, timeout, or 5xx (see router/client.py's
        # fault injection, and OllamaUnavailableError for the real-Ollama
        # equivalent) - so there's no response at all yet for the judge to
        # see. Exactly one failover attempt is made, against the bandit's
        # own best-known *other* arm (the same call the escalation paths
        # below use) - not a hardcoded fallback name, for the same reason
        # documented on best_known_model(). If that also fails, we raise
        # rather than silently giving up: a caller that swallowed this
        # would have no way to distinguish "the fault was transient and we
        # recovered" from "both the primary and its replacement are down,"
        # which is exactly the distinction fault-injection testing needs.
        logger.info(f"Routing query to: {selected_model}")
        attempted_backend = selected_model  # whichever model is actually dialed first, post confidence-escalation
        backend_failed_over = False
        backend_failure_reason = None
        backend_failure_code = None
        try:
            response = self.client.generate(selected_model, query)
        except Exception as primary_exc:
            backend_failure_reason = classify_backend_error(primary_exc)
            backend_failure_code = backend_error_code(primary_exc)
            logger.warning(
                f"Backend '{selected_model}' failed: {backend_failure_reason}. Attempting failover."
            )

            failover_target, _ = self.bandit.best_known_model(
                context_vector, exclude=selected_model, prefer=self.fallback_model
            )
            if failover_target is None:
                raise RoutingFailoverError(
                    primary_model=selected_model,
                    primary_exc=primary_exc,
                    failover_model=None,
                ) from primary_exc

            try:
                response = self.client.generate(failover_target, query)
            except Exception as failover_exc:
                logger.error(
                    f"Failover backend '{failover_target}' also failed for query "
                    f"{query!r}: {classify_backend_error(failover_exc)}"
                )
                raise RoutingFailoverError(
                    primary_model=selected_model,
                    primary_exc=primary_exc,
                    failover_model=failover_target,
                    failover_exc=failover_exc,
                ) from failover_exc

            logger.warning(f"Backend failover succeeded: '{selected_model}' -> '{failover_target}'.")
            backend_failed_over = True
            selected_model = failover_target

        # 5. Evaluate Response
        score, judge_metadata = self.judge.evaluate(query, response.response_text, selected_model)

        # We need a composite reward. It should balance quality (score) and cost.
        # A simple reward function: Quality - Penalty for Cost
        # Since scores are [0,1], we can normalize cost to a similar scale or penalize slightly.
        # e.g., reward = score - (cost_per_1M * 0.1)
        # Tracks every (model -> reward) actually observed this round, so the
        # bandit learns from every arm that really ran - not only whichever
        # one's response is ultimately served. Without this, an arm escalated
        # away from on a low judge score would never have its own outcome
        # recorded, and would stay stuck at its prior belief forever.
        observed_rewards = {selected_model: max(0.0, score - response.simulated_cost * 0.1)}

        # 6. Fallback Logic: Low Judge Score
        # Same dynamic-target reasoning as step 3: retry with the bandit's
        # own best-known *other* arm rather than a hardcoded fallback name.
        if score < self.JUDGE_SCORE_THRESHOLD and not escalation_triggered:
            retry_target, _ = self.bandit.best_known_model(
                context_vector, exclude=selected_model, prefer=self.fallback_model
            )
            if retry_target is not None:
                logger.warning(f"Escalation: Judge score too low ({score:.2f}). Retrying with {retry_target}.")
                # retry_target is just "the bandit's best other arm at this
                # context" - nothing stops it from being a backend that's
                # currently down (e.g. it could even be the very backend
                # step 4 already failed over away from, since that only
                # excludes whichever model is selected_model *now*). A
                # response already exists at this point (from the primary
                # attempt or step 4's own failover), so if this best-effort
                # quality retry hits a fault, we log it and keep that
                # existing response rather than letting an unhandled
                # exception crash a request that already succeeded.
                try:
                    retry_response = self.client.generate(retry_target, query)
                except Exception as retry_exc:
                    logger.warning(
                        f"Judge-score retry backend '{retry_target}' also failed "
                        f"({classify_backend_error(retry_exc)}); keeping the current response."
                    )
                else:
                    retry_score, retry_metadata = self.judge.evaluate(query, retry_response.response_text, retry_target)
                    observed_rewards[retry_target] = max(0.0, retry_score - retry_response.simulated_cost * 0.1)

                    # If the retry did better, use it. Otherwise stick with the
                    # original to not waste more time.
                    if retry_score > score:
                        response = retry_response
                        score = retry_score
                        judge_metadata = retry_metadata
                        selected_model = retry_target
                        escalation_triggered = True
                        escalation_reason = "low_judge_score"
                    else:
                        logger.warning(f"Retry didn't improve score. Sticking with original model {selected_model}.")

        # 7. Update Bandit - once per distinct model actually executed this
        # round, each with its own observed reward.
        final_reward = observed_rewards[selected_model]
        for model, reward in observed_rewards.items():
            self.bandit.update(model, context_vector, reward)
        
        # 8. Return comprehensive payload logged separately from bandit internals
        result = {
            "query": query,
            "response": response,
            "model_used": selected_model,
            "bandit_selected_model": bandit_selected_model,
            "attempted_backend": attempted_backend,
            "judge_score": score,
            "judge_metadata": judge_metadata,
            "final_reward": final_reward,
            "escalated": escalation_triggered,
            "escalation_reason": escalation_reason,
            "bandit_expected_reward": expected_reward,
            "bandit_uncertainty": uncertainty,
            "backend_failed_over": backend_failed_over,
            "backend_failure_reason": backend_failure_reason,
            "backend_failure_code": backend_failure_code,
        }
        
        # Stage 8a: Write structured telemetry log
        if self.structured_logger:
            self.structured_logger.log_decision(result)
            
        return result

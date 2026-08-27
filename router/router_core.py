import logging
from typing import Dict, Any, List, Optional
from router.client import UnifiedLLMClient, LLMResponse
from router.embeddings import ContextEmbedder
from router.bandit import LinUCBRouter
from judge.judge import LLMJudge

logger = logging.getLogger(__name__)

class OptimizationRouter:
    """
    The Core Routing Intelligence (Phase 2).
    
    Ties together the client, embedding, contextual bandit, and LLM judge.
    Features:
    - Embedding-based context vectors (Stage 1/2)
    - LinUCB Contextual Bandit routing (Stage 3)
    - Non-stationary shock adaptation via bandit decay (Stage 4)
    - Fallback & escalation logic for low confidence/low quality (Stage 5)
    """
    
    def __init__(self, 
                 client: UnifiedLLMClient, 
                 embedder: ContextEmbedder, 
                 bandit: LinUCBRouter, 
                 judge: LLMJudge,
                 fallback_model: str = "mistral"):
        self.client = client
        self.embedder = embedder
        self.bandit = bandit
        self.judge = judge
        
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
        selected_model, expected_reward, uncertainty = self.bandit.select_model(context_vector)
        
        escalation_triggered = False
        escalation_reason = None
        
        # 3. Fallback Logic: Low Confidence
        if expected_reward < self.CONFIDENCE_THRESHOLD and selected_model != self.fallback_model:
            logger.warning(f"Escalation: Bandit confidence low (Expected Reward: {expected_reward:.2f}). Escalating to {self.fallback_model}.")
            selected_model = self.fallback_model
            escalation_triggered = True
            escalation_reason = "low_confidence"
            
        # 4. Generate Response
        logger.info(f"Routing query to: {selected_model}")
        response = self.client.generate(selected_model, query)
        
        # 5. Evaluate Response
        score, judge_metadata = self.judge.evaluate(query, response.response_text, selected_model)
        
        # 6. Fallback Logic: Low Judge Score
        if score < self.JUDGE_SCORE_THRESHOLD and not escalation_triggered and selected_model != self.fallback_model:
            logger.warning(f"Escalation: Judge score too low ({score:.2f}). Retrying with {self.fallback_model}.")
            # Retry with fallback
            fallback_response = self.client.generate(self.fallback_model, query)
            fallback_score, fallback_metadata = self.judge.evaluate(query, fallback_response.response_text, self.fallback_model)
            
            # If fallback did better, use it. Otherwise stick with original to not waste more time.
            if fallback_score > score:
                response = fallback_response
                score = fallback_score
                judge_metadata = fallback_metadata
                selected_model = self.fallback_model
                escalation_triggered = True
                escalation_reason = "low_judge_score"
            else:
                logger.warning(f"Fallback didn't improve score. Sticking with original model {selected_model}.")
        
        # 7. Update Bandit
        # Only update if we didn't escalate due to low confidence (as that means we bypassed the bandit's choice).
        # Alternatively, we CAN update the bandit with the score of whatever model we eventually used.
        # It's usually better to update for the model that actually ran, so it learns about the fallback model too.
        
        # We need a composite reward. It should balance quality (score) and cost.
        # A simple reward function: Quality - Penalty for Cost
        # Since scores are [0,1], we can normalize cost to a similar scale or penalize slightly.
        # e.g., reward = score - (cost_per_1M * 0.1)
        cost_penalty = response.simulated_cost * 0.1 
        final_reward = max(0.0, score - cost_penalty)
        
        self.bandit.update(selected_model, context_vector, final_reward)
        
        # 8. Return comprehensive payload logged separately from bandit internals
        return {
            "query": query,
            "response": response,
            "model_used": selected_model,
            "judge_score": score,
            "judge_metadata": judge_metadata,
            "final_reward": final_reward,
            "escalated": escalation_triggered,
            "escalation_reason": escalation_reason,
            "bandit_expected_reward": expected_reward,
            "bandit_uncertainty": uncertainty
        }

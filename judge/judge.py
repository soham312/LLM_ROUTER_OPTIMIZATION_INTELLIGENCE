import re
import logging
from typing import Dict, Any, Tuple
from router.client import UnifiedLLMClient

logger = logging.getLogger(__name__)

class LLMJudge:
    """
    LLM-as-a-Judge for evaluating response quality.
    
    Why use an LLM as a judge?
    In a router setup, we need an automated way to compute 'reward' for the bandit.
    Static datasets don't work for open-ended generation. Using a capable local 
    model (like mistral or llama3.2:3b) to score responses allows the bandit to 
    learn continuously.
    
    Length-Bias Mitigation:
    LLMs inherently favor longer responses (verbosity bias). We mitigate this by:
    1. Explicit prompt instructions to penalize unnecessary verbosity.
    2. A programmatic penalty if the response is excessively long compared to the prompt.
    """
    
    def __init__(self, client: UnifiedLLMClient, judge_model: str = "mistral"):
        self.client = client
        self.judge_model = judge_model
        
        # State to simulate mid-run shock (e.g. model quality degradation)
        # We can artificially lower the score for a specific model.
        self._shock_state = {}
        
    def set_shock(self, target_model: str, score_penalty: float):
        """
        STAGE 4: Deterministic shock simulation.
        By calling this, we simulate that 'target_model' suddenly degraded in quality.
        The bandit should detect this (via lower rewards) and route traffic away from it.
        """
        self._shock_state[target_model] = score_penalty
        logger.warning(f"SHOCK TRIGGERED: {target_model} will now receive a penalty of -{score_penalty} to its scores.")
        
    def clear_shock(self, target_model: str):
        if target_model in self._shock_state:
            del self._shock_state[target_model]
            logger.info(f"SHOCK CLEARED for {target_model}.")

    def evaluate(self, prompt: str, response: str, model_used: str) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluates a response and returns a score between 0.0 and 1.0, 
        along with metadata (reasoning, length penalty applied).
        """
        # If in mock mode, generate a fast deterministic mock score
        if self.client.mock_mode:
            # Deterministic pseudo-random score based on model and prompt length
            base_score = 0.8 if model_used == "mistral" else \
                         0.7 if model_used == "phi3" else \
                         0.6 if model_used == "llama3.2:3b" else 0.4
            
            # Fluctuate deterministically
            hash_val = abs(hash(prompt + response)) % 100
            score = base_score + (hash_val / 100.0) * 0.2
            
            # Apply shock if active
            penalty = self._shock_state.get(model_used, 0.0)
            score = max(0.0, score - penalty)
            
            return score, {"reasoning": "Mock evaluation", "length_penalty": 0.0, "shock_penalty": penalty}
            
        # Real evaluation prompt
        eval_prompt = f"""
        You are an impartial judge evaluating an AI assistant's response.
        
        Task: Rate the assistant's response on a scale of 1 to 10.
        Criteria:
        - Accuracy and helpfulness.
        - Conciseness. Penalize responses that are unnecessarily long or repetitive.
        
        User Prompt: {prompt}
        
        Assistant Response: {response}
        
        Provide a brief reasoning, then the final score in the exact format:
        SCORE: <number>
        """
        
        try:
            judge_response = self.client.generate(self.judge_model, eval_prompt)
            eval_text = judge_response.response_text
            
            # Parse score
            match = re.search(r"SCORE:\s*([0-9.]+)", eval_text)
            if match:
                raw_score = float(match.group(1)) / 10.0 # Normalize to 0-1
            else:
                logger.warning(f"Failed to parse score from judge. Using default 0.5. Eval text: {eval_text}")
                raw_score = 0.5
                
            # Programmatic length-bias mitigation (heuristic)
            prompt_len = len(prompt.split())
            resp_len = len(response.split())
            length_penalty = 0.0
            
            # If response is > 5x the prompt length and > 100 words, apply small penalty
            if resp_len > prompt_len * 5 and resp_len > 100:
                length_penalty = 0.1
                
            final_score = max(0.0, min(1.0, raw_score - length_penalty))
            
            # Apply shock if active
            shock_penalty = self._shock_state.get(model_used, 0.0)
            final_score = max(0.0, final_score - shock_penalty)
            
            metadata = {
                "reasoning": eval_text,
                "raw_score": raw_score,
                "length_penalty": length_penalty,
                "shock_penalty": shock_penalty
            }
            return final_score, metadata
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return 0.5, {"error": str(e)}

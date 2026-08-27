import time
import random
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import uuid

# Attempt to import ollama, but allow mock mode even if not installed
try:
    import ollama
except ImportError:
    ollama = None

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    id: str
    model: str
    response_text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    simulated_cost: float
    is_mock: bool

class UnifiedLLMClient:
    """
    Unified client for interacting with multiple LLMs.
    Supports real execution via Ollama and a mock mode for fast iterations.
    
    Why use a unified client?
    In a router setup, the routing logic shouldn't care about the intricacies 
    of each model's API. This wrapper abstracts away the execution details, 
    enforcing a uniform response format that includes critical metadata 
    (latency, cost) needed by the contextual bandit to calculate rewards.
    """
    
    # Simulated pricing per 1 million tokens (combining prompt & completion for simplicity)
    # These prices are proxy figures representing the relative cost of different model sizes
    # in a real hosted environment, mapping our zero-cost local models to production realities.
    PRICING_PER_1M_TOKENS = {
        "llama3.2:1b": 0.10,  # Proxy for cheap tier (e.g., GPT-4o-mini / Haiku class)
        "llama3.2:3b": 0.20,  # Proxy for mid-cheap tier
        "phi3": 0.50,         # Proxy for mid tier
        "mistral": 1.00,      # Proxy for expensive tier (e.g., GPT-4o / Opus class proxy)
    }
    
    def __init__(self, mock_mode: bool = False):
        self.mock_mode = mock_mode
        if not self.mock_mode and ollama is None:
            logger.warning("Ollama not installed. Forcing mock mode. Install with: pip install ollama")
            self.mock_mode = True
            
    def _calculate_cost(self, model: str, total_tokens: int) -> float:
        """Calculates the simulated cost for a request based on proxy pricing."""
        # Default to $0.50 if model is unknown
        price_per_1m = self.PRICING_PER_1M_TOKENS.get(model, 0.50)
        return (total_tokens / 1_000_000.0) * price_per_1m

    def _mock_generate(self, model: str, prompt: str) -> LLMResponse:
        """
        Simulates an LLM call without doing actual compute.
        Useful for running thousands of episodes to train the bandit 
        without waiting for actual inference times, enabling rapid experiments.
        """
        # Simulate varying latencies based on model "size" (larger models take longer)
        base_latency = 50 if model == "llama3.2:1b" else \
                       100 if model == "llama3.2:3b" else \
                       150 if model == "phi3" else \
                       250 # mistral
                       
        time.sleep(random.uniform(0.01, 0.05)) # Tiny sleep to yield thread
        
        latency_ms = base_latency + random.uniform(10, 50)
        
        # Estimate tokens (rough heuristic: 1 word ~ 1.3 tokens)
        word_count = len(prompt.split())
        prompt_tokens = int(word_count * 1.3)
        
        # Simulate completion length
        completion_tokens = random.randint(10, 150)
        total_tokens = prompt_tokens + completion_tokens
        
        response_text = f"[MOCK {model}] Simulated response to: {prompt[:30]}..."
        cost = self._calculate_cost(model, total_tokens)
        
        return LLMResponse(
            id=str(uuid.uuid4()),
            model=model,
            response_text=response_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            simulated_cost=cost,
            is_mock=True
        )

    def generate(self, model: str, prompt: str) -> LLMResponse:
        """
        Generates a response using the requested model.
        Returns a standardized LLMResponse containing the text, real latency, and simulated cost.
        """
        if self.mock_mode:
            return self._mock_generate(model, prompt)
            
        start_time = time.time()
        
        try:
            # Use Ollama python client. Assumes local ollama server is running and models are pulled.
            response = ollama.generate(model=model, prompt=prompt)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract token counts provided by Ollama
            prompt_tokens = response.get('prompt_eval_count', 0)
            completion_tokens = response.get('eval_count', 0)
            total_tokens = prompt_tokens + completion_tokens
            
            # Fallback heuristic if Ollama doesn't return counts for some reason
            if total_tokens == 0:
                prompt_tokens = int(len(prompt.split()) * 1.3)
                completion_tokens = int(len(response.get('response', '').split()) * 1.3)
                total_tokens = prompt_tokens + completion_tokens
            
            cost = self._calculate_cost(model, total_tokens)
            
            return LLMResponse(
                id=str(uuid.uuid4()),
                model=model,
                response_text=response.get('response', ''),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                simulated_cost=cost,
                is_mock=False
            )
        except Exception as e:
            logger.error(f"Error calling Ollama model {model}: {e}")
            raise

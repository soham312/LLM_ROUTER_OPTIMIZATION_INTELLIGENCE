import time
import random
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import uuid

# Attempt to import ollama, but allow mock mode even if not installed
try:
    import ollama
except ImportError:
    ollama = None

logger = logging.getLogger(__name__)


class OllamaUnavailableError(RuntimeError):
    """
    Raised when a real (non-mock) generate() call can't reach the Ollama
    server or the requested model isn't pulled. Callers must handle this
    explicitly - the client never silently falls back to mock data on a
    backend failure, since that would corrupt latency/cost telemetry the
    bandit and SLA tracker rely on being real.
    """


# --- Fault injection --------------------------------------------------
#
# These simulate realistic backend failure modes (not a generic
# Exception) so fault-injection experiments exercise the same exception
# shapes a real deployment would throw. Each subclasses the builtin
# exception a real client would actually raise for that failure, so
# calling code that already does `except ConnectionError` or
# `except TimeoutError` behaves the same against a simulated fault as
# against a real one.

class SimulatedConnectionRefusedError(ConnectionError):
    """The backend process/port is unreachable - server down, wrong port,
    container crashed. Analogous to errno ECONNREFUSED."""


class SimulatedTimeoutError(TimeoutError):
    """The backend is reachable but did not respond in time - overloaded,
    hung request, or a network partition dropping packets silently."""


class SimulatedServerError(RuntimeError):
    """The backend responded, but with a server-side error (5xx) - e.g. it
    crashed handling this specific request, or a reverse proxy in front of
    it returned bad gateway / service unavailable / gateway timeout."""

    def __init__(self, model: str, status_code: int):
        self.model = model
        self.status_code = status_code
        super().__init__(
            f"Backend '{model}' returned HTTP {status_code} (simulated fault injection)"
        )


_VALID_FAULT_ERROR_TYPES = ("connection_refused", "timeout", "server_error")


@dataclass
class FaultConfig:
    """Configuration for a single faulted backend.

    :param probability: Per-request probability in [0, 1] that a call to
        this backend fails. Ignored (treated as 1.0) when hard_down=True.
    :param hard_down: If True, every call to this backend fails for as
        long as the fault is armed - simulates the backend being
        completely down for the run, rather than merely flaky.
    :param error_types: Which failure shapes to draw from when the fault
        triggers. Defaults to all three so repeated triggers look like a
        real mixed-failure backend rather than one canned error.
    """

    probability: float = 1.0
    hard_down: bool = False
    error_types: Tuple[str, ...] = _VALID_FAULT_ERROR_TYPES

    def __post_init__(self):
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be in [0.0, 1.0], got {self.probability}")
        if not self.error_types:
            raise ValueError("error_types must be non-empty")
        unknown = set(self.error_types) - set(_VALID_FAULT_ERROR_TYPES)
        if unknown:
            raise ValueError(
                f"Unknown error_types {sorted(unknown)}; must be a subset of {_VALID_FAULT_ERROR_TYPES}"
            )


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

def classify_backend_error(exc: Exception) -> str:
    """
    Full, human-readable reason a backend call failed: which known
    failure mode it was, plus the original message. Anything not
    recognized is still reported - tagged 'unclassified_error' with its
    real exception class name attached - never folded into a generic
    bucket. Shared by router_core's failover handling and by
    experiments/fault_injection.py's reporting, so both describe the
    same failure the same way.
    """
    if isinstance(exc, SimulatedConnectionRefusedError):
        return f"connection_refused: {exc}"
    if isinstance(exc, SimulatedTimeoutError):
        return f"timeout: {exc}"
    if isinstance(exc, SimulatedServerError):
        return f"http_{exc.status_code}: {exc}"
    if isinstance(exc, OllamaUnavailableError):
        return f"ollama_unavailable: {exc}"
    return f"unclassified_error[{type(exc).__name__}]: {exc}"


def backend_error_code(exc: Exception) -> str:
    """Stable, low-cardinality bucket key for aggregate breakdowns - the
    full detail (exact message, random timeout duration, HTTP status,
    etc.) lives in classify_backend_error(), not here."""
    if isinstance(exc, SimulatedConnectionRefusedError):
        return "connection_refused"
    if isinstance(exc, SimulatedTimeoutError):
        return "timeout"
    if isinstance(exc, SimulatedServerError):
        return f"http_{exc.status_code}"
    if isinstance(exc, OllamaUnavailableError):
        return "ollama_unavailable"
    return f"unclassified_error[{type(exc).__name__}]"


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

        # model -> FaultConfig, for fault-injection experiments.
        self._faults: Dict[str, FaultConfig] = {}

    def inject_fault(
        self,
        model: str,
        probability: float = 1.0,
        hard_down: bool = False,
        error_types: Optional[Tuple[str, ...]] = None,
    ) -> None:
        """
        Arms a fault on `model`: subsequent generate() calls to it will
        raise a simulated backend error instead of producing a response,
        either probabilistically (`probability`) or unconditionally for
        the rest of the run (`hard_down=True`).
        """
        config = FaultConfig(
            probability=probability,
            hard_down=hard_down,
            error_types=tuple(error_types) if error_types else _VALID_FAULT_ERROR_TYPES,
        )
        self._faults[model] = config
        logger.warning(
            f"Fault injection ARMED for '{model}': hard_down={config.hard_down}, "
            f"probability={config.probability}, error_types={config.error_types}"
        )

    def clear_fault(self, model: Optional[str] = None) -> None:
        """Disarms the fault on `model`, or every armed fault if model is None."""
        if model is None:
            self._faults.clear()
            logger.info("Fault injection cleared for all backends.")
        else:
            self._faults.pop(model, None)
            logger.info(f"Fault injection cleared for '{model}'.")

    def is_faulted(self, model: str) -> bool:
        return model in self._faults

    def _maybe_raise_fault(self, model: str) -> None:
        """Raises a simulated backend error if `model` has an armed fault
        that triggers this call. Never swallows anything - either returns
        silently (no fault configured, or this call got a lucky roll under
        a probabilistic fault) or raises a specific, logged exception."""
        config = self._faults.get(model)
        if config is None:
            return

        triggered = config.hard_down or (random.random() < config.probability)
        if not triggered:
            return

        error_type = random.choice(config.error_types)
        if error_type == "connection_refused":
            exc: Exception = SimulatedConnectionRefusedError(
                f"[Errno 61] Connection refused: backend '{model}' is unreachable (fault injection)"
            )
        elif error_type == "timeout":
            timeout_ms = random.randint(2000, 10000)
            exc = SimulatedTimeoutError(
                f"Request to backend '{model}' timed out after {timeout_ms}ms (fault injection)"
            )
        elif error_type == "server_error":
            status_code = random.choice([500, 502, 503, 504])
            exc = SimulatedServerError(model, status_code)
        else:
            # Unreachable given FaultConfig's validation, but never silently
            # ignore an unrecognized configuration either.
            raise ValueError(f"Unknown fault error_type '{error_type}' configured for '{model}'")

        logger.error(f"FAULT INJECTED on '{model}': {type(exc).__name__}: {exc}")
        raise exc

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
        self._maybe_raise_fault(model)

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
        except ConnectionError as e:
            # ollama-python raises the builtin ConnectionError when the
            # local server isn't reachable at all (connection refused).
            logger.error(f"Ollama server unreachable for model {model}: {e}")
            raise OllamaUnavailableError(
                f"Could not reach the Ollama server while generating with "
                f"'{model}'. Is it running? Start it with `ollama serve` "
                f"(or `brew services start ollama`)."
            ) from e
        except ollama.ResponseError as e:
            # Raised for server-side failures, e.g. the model isn't pulled
            # (404) - distinguished from "server is down" so the operator
            # knows to `ollama pull` rather than start the server.
            logger.error(f"Ollama rejected request for model {model}: {e}")
            raise OllamaUnavailableError(
                f"Ollama rejected the request for model '{model}' "
                f"(status {e.status_code}: {e.error}). Is it pulled? Run "
                f"`ollama pull {model}`."
            ) from e

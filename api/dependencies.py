import os
from functools import lru_cache

from judge.judge import LLMJudge
from observability.logger import StructuredLogger
from router.bandit import LinUCBRouter
from router.client import UnifiedLLMClient
from router.embeddings import ContextEmbedder
from router.router_core import OptimizationRouter


@lru_cache
def get_router() -> OptimizationRouter:
    """
    Builds the single OptimizationRouter instance shared by every request in
    this process. @lru_cache with no arguments makes this a lazy singleton -
    built once, on first request, rather than at import time (so importing
    this module for testing doesn't eagerly spin up a real client/embedder).

    Why mock_mode=True by default? Mirrors the project's zero-cost
    constraint: the API is runnable and demoable without a local Ollama
    server. Set ROUTER_MOCK_MODE=false (with Ollama running and models
    pulled) to serve real inference instead.

    Embeddings are controlled by a separate ROUTER_MOCK_EMBEDDINGS flag
    (defaulting to whatever ROUTER_MOCK_MODE resolves to, so the common
    case needs no extra configuration) rather than being forced to follow
    the LLM client - real embeddings need sentence-transformers/torch
    installed, a meaningfully heavier dependency than real LLM inference
    alone, so it should be possible to smoke-test real Ollama calls with
    mocked embeddings without that install being a hard requirement.
    """
    mock_mode = os.environ.get("ROUTER_MOCK_MODE", "true").lower() != "false"
    embeddings_mock_mode = os.environ.get(
        "ROUTER_MOCK_EMBEDDINGS", "true" if mock_mode else "false"
    ).lower() != "false"
    models = list(UnifiedLLMClient.PRICING_PER_1M_TOKENS.keys())

    client = UnifiedLLMClient(mock_mode=mock_mode)
    embedder = ContextEmbedder(mock_mode=embeddings_mock_mode)
    bandit = LinUCBRouter(models=models, embedding_dim=embedder.embedding_dim)
    judge = LLMJudge(client)
    structured_logger = StructuredLogger()

    return OptimizationRouter(
        client=client,
        embedder=embedder,
        bandit=bandit,
        judge=judge,
        fallback_model="mistral",
        structured_logger=structured_logger,
    )

from unittest.mock import MagicMock

import pytest

import router.embeddings as embeddings_module
from api.dependencies import get_router


@pytest.fixture(autouse=True)
def clear_router_cache():
    """get_router is an lru_cache singleton - clear it so each test builds fresh."""
    get_router.cache_clear()
    yield
    get_router.cache_clear()


def test_get_router_defaults_to_full_mock_mode(monkeypatch):
    monkeypatch.delenv("ROUTER_MOCK_MODE", raising=False)
    monkeypatch.delenv("ROUTER_MOCK_EMBEDDINGS", raising=False)

    router = get_router()

    assert router.client.mock_mode is True
    assert router.embedder.mock_mode is True


def test_get_router_embeddings_follow_mock_mode_by_default(monkeypatch):
    # Real-mode ContextEmbedder construction needs SentenceTransformer to
    # exist; patch it so this test doesn't require the real dependency
    # installed to prove the *resolution logic* is correct.
    monkeypatch.setattr(embeddings_module, "SentenceTransformer", MagicMock())
    monkeypatch.setenv("ROUTER_MOCK_MODE", "false")
    monkeypatch.delenv("ROUTER_MOCK_EMBEDDINGS", raising=False)

    router = get_router()

    assert router.client.mock_mode is False
    assert router.embedder.mock_mode is False


def test_get_router_embeddings_can_be_mocked_independently_of_the_client(monkeypatch):
    """
    Regression test for a real-Ollama smoke-test finding: real LLM
    inference needs only a running Ollama server; real embeddings
    additionally need sentence-transformers/torch installed - a
    meaningfully heavier dependency. ROUTER_MOCK_EMBEDDINGS lets the two be
    configured independently instead of forcing both to follow
    ROUTER_MOCK_MODE, so real inference can be exercised without that
    heavier install being a hard requirement.
    """
    monkeypatch.setenv("ROUTER_MOCK_MODE", "false")
    monkeypatch.setenv("ROUTER_MOCK_EMBEDDINGS", "true")

    router = get_router()

    assert router.client.mock_mode is False
    assert router.embedder.mock_mode is True

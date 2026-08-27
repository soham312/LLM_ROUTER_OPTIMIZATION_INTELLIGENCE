import pytest
import numpy as np
from router.embeddings import ContextEmbedder

def test_mock_embedding():
    embedder = ContextEmbedder(mock_mode=True)
    
    query = "What is the capital of France?"
    vec1 = embedder.get_embedding(query)
    
    assert isinstance(vec1, np.ndarray)
    assert vec1.shape == (384,)
    
    # Determinism check in mock mode
    vec2 = embedder.get_embedding(query)
    np.testing.assert_array_equal(vec1, vec2)

def test_empty_embedding():
    embedder = ContextEmbedder(mock_mode=True)
    vec = embedder.get_embedding("")
    
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert np.all(vec == 0)

def test_conversation_embedding():
    embedder = ContextEmbedder(mock_mode=True)
    conversation = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "Write a python script"}
    ]
    
    vec = embedder.get_embedding(conversation)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)

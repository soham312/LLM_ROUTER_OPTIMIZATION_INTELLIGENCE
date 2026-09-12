import pytest
from router.client import (
    OllamaUnavailableError,
    SimulatedConnectionRefusedError,
    SimulatedServerError,
    SimulatedTimeoutError,
    UnifiedLLMClient,
    backend_error_code,
    classify_backend_error,
)

def test_mock_client_generation():
    client = UnifiedLLMClient(mock_mode=True)
    response = client.generate("llama3.2:1b", "Hello world")
    
    assert response.is_mock is True
    assert response.model == "llama3.2:1b"
    assert response.prompt_tokens > 0
    assert response.completion_tokens > 0
    assert response.total_tokens == response.prompt_tokens + response.completion_tokens
    assert response.simulated_cost > 0
    assert response.latency_ms > 0

def test_cost_calculation():
    client = UnifiedLLMClient(mock_mode=True)
    
    # 1 million tokens for llama3.2:1b should be $0.10
    cost = client._calculate_cost("llama3.2:1b", 1_000_000)
    assert cost == 0.10
    
    # 1 million tokens for mistral should be $1.00
    cost = client._calculate_cost("mistral", 1_000_000)
    assert cost == 1.00


# --- Fault injection ---------------------------------------------------

def test_hard_down_always_fails():
    client = UnifiedLLMClient(mock_mode=True)
    client.inject_fault("mistral", hard_down=True)
    for _ in range(20):
        with pytest.raises((SimulatedConnectionRefusedError, SimulatedTimeoutError, SimulatedServerError)):
            client.generate("mistral", "hello")

def test_probability_zero_never_triggers():
    client = UnifiedLLMClient(mock_mode=True)
    client.inject_fault("mistral", probability=0.0)
    for _ in range(20):
        response = client.generate("mistral", "hello")
        assert response.model == "mistral"

def test_probability_one_always_triggers():
    client = UnifiedLLMClient(mock_mode=True)
    client.inject_fault("mistral", probability=1.0)
    for _ in range(20):
        with pytest.raises((SimulatedConnectionRefusedError, SimulatedTimeoutError, SimulatedServerError)):
            client.generate("mistral", "hello")

def test_fault_only_affects_targeted_backend():
    client = UnifiedLLMClient(mock_mode=True)
    client.inject_fault("mistral", hard_down=True)
    response = client.generate("phi3", "hello")
    assert response.model == "phi3"

def test_clear_fault_restores_normal_behavior():
    client = UnifiedLLMClient(mock_mode=True)
    client.inject_fault("mistral", hard_down=True)
    with pytest.raises(Exception):
        client.generate("mistral", "hello")

    client.clear_fault("mistral")
    response = client.generate("mistral", "hello")
    assert response.model == "mistral"

def test_clear_fault_with_no_model_clears_everything():
    client = UnifiedLLMClient(mock_mode=True)
    client.inject_fault("mistral", hard_down=True)
    client.inject_fault("phi3", hard_down=True)

    client.clear_fault()

    assert not client.is_faulted("mistral")
    assert not client.is_faulted("phi3")
    client.generate("mistral", "hello")
    client.generate("phi3", "hello")

def test_restricted_error_types_only_raises_requested_type():
    client = UnifiedLLMClient(mock_mode=True)
    client.inject_fault("mistral", hard_down=True, error_types=["timeout"])
    with pytest.raises(SimulatedTimeoutError):
        client.generate("mistral", "hello")

def test_server_error_carries_a_realistic_5xx_status_code():
    client = UnifiedLLMClient(mock_mode=True)
    client.inject_fault("mistral", hard_down=True, error_types=["server_error"])
    with pytest.raises(SimulatedServerError) as exc_info:
        client.generate("mistral", "hello")
    assert exc_info.value.status_code in (500, 502, 503, 504)

def test_is_faulted_reflects_current_state():
    client = UnifiedLLMClient(mock_mode=True)
    assert not client.is_faulted("mistral")
    client.inject_fault("mistral", hard_down=True)
    assert client.is_faulted("mistral")
    client.clear_fault("mistral")
    assert not client.is_faulted("mistral")

def test_invalid_probability_rejected():
    client = UnifiedLLMClient(mock_mode=True)
    with pytest.raises(ValueError):
        client.inject_fault("mistral", probability=1.5)

def test_unknown_error_type_rejected():
    client = UnifiedLLMClient(mock_mode=True)
    with pytest.raises(ValueError):
        client.inject_fault("mistral", error_types=["502_bad_gateway"])


# --- classify_backend_error / backend_error_code ------------------------

def test_classify_and_code_agree_and_are_specific_per_error_type():
    client = UnifiedLLMClient(mock_mode=True)

    for error_type, expected_code in [
        ("connection_refused", "connection_refused"),
        ("timeout", "timeout"),
    ]:
        client.inject_fault("mistral", hard_down=True, error_types=[error_type])
        try:
            client.generate("mistral", "hi")
            pytest.fail("expected the injected fault to raise")
        except Exception as e:
            assert backend_error_code(e) == expected_code
            assert classify_backend_error(e).startswith(f"{expected_code}:")
        client.clear_fault("mistral")

def test_server_error_code_includes_status():
    client = UnifiedLLMClient(mock_mode=True)
    client.inject_fault("mistral", hard_down=True, error_types=["server_error"])
    try:
        client.generate("mistral", "hi")
        pytest.fail("expected the injected fault to raise")
    except Exception as e:
        code = backend_error_code(e)
        assert code in ("http_500", "http_502", "http_503", "http_504")
        assert classify_backend_error(e).startswith(f"{code}:")

def test_ollama_unavailable_error_is_classified_not_left_generic():
    exc = OllamaUnavailableError("Could not reach the Ollama server while generating with 'mistral'.")
    assert backend_error_code(exc) == "ollama_unavailable"
    assert classify_backend_error(exc).startswith("ollama_unavailable:")

def test_unrecognized_exception_is_never_hidden_in_a_generic_bucket():
    class WeirdBackendError(Exception):
        pass

    exc = WeirdBackendError("something the harness has never seen before")
    assert backend_error_code(exc) == "unclassified_error[WeirdBackendError]"
    assert "WeirdBackendError" in classify_backend_error(exc)
    assert "something the harness has never seen before" in classify_backend_error(exc)

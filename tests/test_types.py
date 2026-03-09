"""
Tests for Pydantic type models.
"""

import pytest
from pydantic import ValidationError

from stratus_sdk.types import (
    AnthropicContentBlock,
    AnthropicRequest,
    AnthropicResponse,
    CreditPackage,
    CreditPurchaseResponse,
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
    LLMKeySetRequest,
    LLMKeySetResponse,
    LLMKeyStatus,
    Message,
    ModelInfo,
    StratusMetadata,
    Usage,
)
from stratus_sdk.exceptions import StratusErrorType


# --- StratusErrorType ---

def test_error_type_enum_values():
    assert StratusErrorType.authentication_error == "authentication_error"
    assert StratusErrorType.insufficient_credits == "insufficient_credits"
    assert StratusErrorType.rate_limit == "rate_limit"
    assert StratusErrorType.invalid_model == "invalid_model"
    assert StratusErrorType.model_not_loaded == "model_not_loaded"
    assert StratusErrorType.llm_provider_not_configured == "llm_provider_not_configured"
    assert StratusErrorType.llm_provider_error == "llm_provider_error"
    assert StratusErrorType.planning_failed == "planning_failed"
    assert StratusErrorType.validation_error == "validation_error"
    assert StratusErrorType.internal_error == "internal_error"


# --- AnthropicRequest ---

def test_anthropic_request_valid():
    req = AnthropicRequest(
        model="stratus-x1-ac",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=100,
    )
    assert req.model == "stratus-x1-ac"
    assert req.max_tokens == 100
    assert req.stream is False


def test_anthropic_request_with_system():
    req = AnthropicRequest(
        model="stratus-x1-ac",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=100,
        system="You are a helpful assistant.",
    )
    assert req.system == "You are a helpful assistant."


# --- AnthropicResponse ---

def test_anthropic_response_valid():
    resp = AnthropicResponse(
        id="msg-1",
        type="message",
        role="assistant",
        content=[{"type": "text", "text": "Hello!"}],
        model="stratus-x1-ac",
    )
    assert resp.id == "msg-1"
    assert len(resp.content) == 1
    assert resp.content[0].text == "Hello!"


# --- EmbeddingRequest / EmbeddingResponse ---

def test_embedding_request_string():
    req = EmbeddingRequest(model="stratus-embed", input="hello world")
    assert req.input == "hello world"


def test_embedding_request_list():
    req = EmbeddingRequest(model="stratus-embed", input=["a", "b", "c"])
    assert len(req.input) == 3  # type: ignore[arg-type]


def test_embedding_response_valid():
    resp = EmbeddingResponse(
        object="list",
        data=[EmbeddingObject(object="embedding", embedding=[0.1, 0.2, 0.3], index=0)],
        model="stratus-embed",
    )
    assert resp.data[0].embedding == [0.1, 0.2, 0.3]


# --- ModelInfo ---

def test_model_info_minimal():
    m = ModelInfo(id="stratus-x1-ac")
    assert m.id == "stratus-x1-ac"
    assert m.object == "model"


def test_model_info_full():
    m = ModelInfo(
        id="stratus-x1-ac",
        object="model",
        created=1700000000,
        owned_by="formation",
        description="Stratus X1 action-conditioned model",
    )
    assert m.owned_by == "formation"


# --- LLMKey types ---

def test_llm_key_set_request_partial():
    req = LLMKeySetRequest(openai_key="sk-openai")
    assert req.openai_key == "sk-openai"
    assert req.anthropic_key is None
    assert req.openrouter_key is None


def test_llm_key_status_defaults_false():
    status = LLMKeyStatus()
    assert status.openai is False
    assert status.anthropic is False
    assert status.openrouter is False


def test_llm_key_set_response():
    resp = LLMKeySetResponse(
        success=True,
        configured=LLMKeyStatus(openai=True, anthropic=False, openrouter=False),
    )
    assert resp.success is True
    assert resp.configured.openai is True


# --- CreditPackage ---

def test_credit_package_valid():
    pkg = CreditPackage(name="starter", credits=100.0, price_usd=9.99)
    assert pkg.name == "starter"
    assert pkg.credits == 100.0
    assert pkg.price_usd == 9.99


def test_credit_purchase_response():
    resp = CreditPurchaseResponse(success=True, credits_added=100.0, new_balance=150.0)
    assert resp.success is True
    assert resp.new_balance == 150.0


# --- StratusMetadata ---

def test_stratus_metadata_defaults_none():
    meta = StratusMetadata()
    assert meta.brain_signal is None
    assert meta.action_sequence is None
    assert meta.planning_time_ms is None


def test_stratus_metadata_full():
    meta = StratusMetadata(
        brain_signal=0.85,
        action_sequence=["step1", "step2"],
        planning_time_ms=120,
        validation_score=0.92,
        mode="plan",
    )
    assert meta.brain_signal == 0.85
    assert len(meta.action_sequence) == 2  # type: ignore[arg-type]
    assert meta.mode == "plan"

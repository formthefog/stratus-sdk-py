"""
Type definitions for Stratus SDK.

Pydantic models matching the M-JEPA-G API specification.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message (OpenAI-compatible format)."""

    role: Literal["system", "user", "assistant"]
    content: str


class Usage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    """Single chat completion choice."""

    index: int
    message: Message
    finish_reason: Literal["stop", "length", "error"]


class ChatCompletionResponse(BaseModel):
    """Chat completion response (matches API)."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class ChatCompletionChunk(BaseModel):
    """Streaming chat completion chunk."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[dict]


class Action(BaseModel):
    """Action with confidence score."""

    action_id: str
    action_text: str
    confidence: float = Field(ge=0.0, le=1.0)


class StatePrediction(BaseModel):
    """State prediction with action and metadata."""

    step: int
    predicted_state: str
    action: Action
    state_change: float


class RolloutSummary(BaseModel):
    """Rollout summary."""

    total_steps: int
    outcome: str
    final_state: str


class RolloutResponse(BaseModel):
    """Rollout response (trajectory prediction)."""

    predictions: List[StatePrediction]
    summary: RolloutSummary
    usage: Usage


class TrajectoryResult(BaseModel):
    """Trajectory result with quality metrics."""

    predictions: List[StatePrediction]
    summary: dict
    usage: Usage


class ModelMetrics(BaseModel):
    """Model performance metrics."""

    model: str
    embedding_quality: Optional[float] = None
    response_quality: Optional[float] = None
    attribution_accuracy: Optional[str] = None
    latency_p50: Optional[int] = None
    latency_p95: Optional[int] = None
    throughput: Optional[float] = None
    cost_per_1m_tokens: Optional[float] = None
    compression_ratio: Optional[float] = None
    error: Optional[str] = None


class ComparisonResult(BaseModel):
    """Model comparison result."""

    results: List[ModelMetrics]
    winner: dict
    timestamp: str


# --- Anthropic-compatible types ---

class AnthropicContentBlock(BaseModel):
    """Content block in Anthropic message format."""

    type: Literal["text", "tool_use", "tool_result"] = "text"
    text: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None


class AnthropicRequest(BaseModel):
    """Anthropic messages API request."""

    model: str
    messages: List[Dict[str, Any]]
    max_tokens: int
    system: Optional[str] = None
    stream: bool = False


class AnthropicResponse(BaseModel):
    """Anthropic messages API response."""

    id: str
    type: str = "message"
    role: str = "assistant"
    content: List[AnthropicContentBlock]
    model: str
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


# --- Embedding types ---

class EmbeddingObject(BaseModel):
    """Single embedding result."""

    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingRequest(BaseModel):
    """Embedding request."""

    model: str
    input: Union[str, List[str]]


class EmbeddingResponse(BaseModel):
    """Embedding response."""

    object: str = "list"
    data: List[EmbeddingObject]
    model: str
    usage: Optional[Dict[str, int]] = None


# --- Model listing ---

class ModelInfo(BaseModel):
    """Info about a single model."""

    id: str
    object: str = "model"
    created: Optional[int] = None
    owned_by: Optional[str] = None
    description: Optional[str] = None


class ModelsResponse(BaseModel):
    """Response from list models endpoint."""

    object: str = "list"
    data: List[ModelInfo]


# --- LLM key management ---

class LLMKeySetRequest(BaseModel):
    """Request to set LLM provider keys."""

    openai_key: Optional[str] = None
    anthropic_key: Optional[str] = None
    openrouter_key: Optional[str] = None


class LLMKeyStatus(BaseModel):
    """Status of configured LLM keys."""

    openai: bool = False
    anthropic: bool = False
    openrouter: bool = False


class LLMKeySetResponse(BaseModel):
    """Response after setting LLM keys."""

    success: bool
    configured: LLMKeyStatus


# --- Credits ---

class CreditPackage(BaseModel):
    """Available credit purchase package."""

    name: str
    credits: float
    price_usd: float
    description: Optional[str] = None


class CreditPurchaseResponse(BaseModel):
    """Response after purchasing credits."""

    success: bool
    credits_added: float
    new_balance: float
    transaction_id: Optional[str] = None


# --- Stratus metadata ---

class StratusMetadata(BaseModel):
    """Stratus-specific metadata returned with completions."""

    brain_signal: Optional[float] = None
    action_sequence: Optional[List[str]] = None
    planning_time_ms: Optional[int] = None
    validation_score: Optional[float] = None
    mode: Optional[str] = None

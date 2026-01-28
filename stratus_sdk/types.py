"""
Type definitions for Stratus SDK.

Pydantic models matching the M-JEPA-G API specification.
"""

from typing import List, Literal, Optional
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

"""
Stratus SDK - Official Python SDK for Stratus X1 M-JEPA-G world model.

Example:
    >>> from stratus_sdk import StratusClient, TrajectoryPredictor
    >>>
    >>> client = StratusClient(api_key="sk-stratus-...")
    >>>
    >>> # Chat completion (OpenAI-compatible)
    >>> response = await client.chat.completions.create(
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ...     model="stratus-x1-ac"
    ... )
    >>>
    >>> # Trajectory prediction
    >>> predictor = TrajectoryPredictor(client)
    >>> result = await predictor.predict(
    ...     initial_state="current state",
    ...     goal="desired goal",
    ...     max_steps=5
    ... )
"""

__version__ = "0.0.2"

# Client (primary name + legacy alias)
from .client import MJepaGClient, StratusClient

# Trajectory prediction
from .trajectory import TrajectoryPredictor

# Model comparison
from .comparison import ModelComparison, compare_models

# Production helpers
from .helpers import (
    CreditMonitor,
    HealthChecker,
    RateLimiter,
    SimpleCache,
    generate_cache_key,
    retry_with_backoff,
)

# Compression profiles
from .profiles import (
    MJEPA_512_BALANCED,
    MJEPA_512_HIGH_COMPRESSION,
    MJEPA_512_HIGH_QUALITY,
    MJEPA_512_ULTRA_COMPRESSION,
    MJEPA_768_BALANCED,
    MJEPA_768_HIGH_COMPRESSION,
    MJEPA_768_HIGH_QUALITY,
    MJEPA_768_ULTRA_COMPRESSION,
    CompressionLevel,
    MJepaProfile,
    detect_mjepa,
    get_mjepa_profile,
    is_mjepa_embedding,
)

# Types
from .types import (
    Action,
    AnthropicContentBlock,
    AnthropicRequest,
    AnthropicResponse,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ComparisonResult,
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
    ModelMetrics,
    ModelsResponse,
    RolloutResponse,
    StatePrediction,
    StratusMetadata,
    TrajectoryResult,
    Usage,
)

# Exceptions
from .exceptions import (
    APIError,
    AuthenticationError,
    InsufficientCreditsError,
    RateLimitError,
    StratusAPIError,
    StratusError,
    StratusErrorType,
    TimeoutError,
    ValidationError,
)

__all__ = [
    # Client
    "StratusClient",
    "MJepaGClient",
    # Trajectory
    "TrajectoryPredictor",
    # Comparison
    "ModelComparison",
    "compare_models",
    # Helpers
    "SimpleCache",
    "RateLimiter",
    "HealthChecker",
    "CreditMonitor",
    "retry_with_backoff",
    "generate_cache_key",
    # Profiles
    "CompressionLevel",
    "MJepaProfile",
    "get_mjepa_profile",
    "is_mjepa_embedding",
    "detect_mjepa",
    "MJEPA_768_HIGH_QUALITY",
    "MJEPA_768_BALANCED",
    "MJEPA_768_HIGH_COMPRESSION",
    "MJEPA_768_ULTRA_COMPRESSION",
    "MJEPA_512_HIGH_QUALITY",
    "MJEPA_512_BALANCED",
    "MJEPA_512_HIGH_COMPRESSION",
    "MJEPA_512_ULTRA_COMPRESSION",
    # Types
    "Message",
    "Usage",
    "ChatCompletionResponse",
    "ChatCompletionChunk",
    "Action",
    "StatePrediction",
    "RolloutResponse",
    "TrajectoryResult",
    "ModelMetrics",
    "ComparisonResult",
    "AnthropicRequest",
    "AnthropicResponse",
    "AnthropicContentBlock",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingObject",
    "ModelInfo",
    "ModelsResponse",
    "LLMKeySetRequest",
    "LLMKeySetResponse",
    "LLMKeyStatus",
    "CreditPackage",
    "CreditPurchaseResponse",
    "StratusMetadata",
    # Exceptions
    "StratusError",
    "StratusAPIError",
    "StratusErrorType",
    "AuthenticationError",
    "RateLimitError",
    "InsufficientCreditsError",
    "APIError",
    "TimeoutError",
    "ValidationError",
]

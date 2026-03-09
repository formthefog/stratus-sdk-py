"""
Exceptions for Stratus SDK.
"""

from enum import Enum
from typing import Optional


class StratusErrorType(str, Enum):
    """Typed error types matching TS SDK."""

    authentication_error = "authentication_error"
    insufficient_credits = "insufficient_credits"
    rate_limit = "rate_limit"
    invalid_model = "invalid_model"
    model_not_loaded = "model_not_loaded"
    llm_provider_not_configured = "llm_provider_not_configured"
    llm_provider_error = "llm_provider_error"
    planning_failed = "planning_failed"
    validation_error = "validation_error"
    internal_error = "internal_error"


class StratusError(Exception):
    """Base exception for Stratus SDK."""

    pass


class StratusAPIError(StratusError):
    """API error with typed error type (primary name, matches TS SDK)."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        error_type: Optional[StratusErrorType] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type


class AuthenticationError(StratusAPIError):
    """Invalid API key or authentication failure."""

    def __init__(self, message: str = "Invalid API key"):
        super().__init__(message, status_code=401, error_type=StratusErrorType.authentication_error)


class RateLimitError(StratusAPIError):
    """Rate limit exceeded."""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429, error_type=StratusErrorType.rate_limit)


class InsufficientCreditsError(StratusAPIError):
    """Insufficient credits to complete request."""

    def __init__(self, message: str = "Insufficient credits"):
        super().__init__(message, status_code=402, error_type=StratusErrorType.insufficient_credits)


class APIError(StratusAPIError):
    """General API error (alias for StratusAPIError for backwards compat)."""

    pass


class TimeoutError(StratusError):
    """Request timeout."""

    pass


class ValidationError(StratusError):
    """Request validation error."""

    pass

"""
Exceptions for Stratus SDK.
"""


class StratusError(Exception):
    """Base exception for Stratus SDK."""

    pass


class AuthenticationError(StratusError):
    """Invalid API key or authentication failure."""

    pass


class RateLimitError(StratusError):
    """Rate limit exceeded."""

    pass


class APIError(StratusError):
    """General API error."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class TimeoutError(StratusError):
    """Request timeout."""

    pass


class ValidationError(StratusError):
    """Request validation error."""

    pass


from typing import Optional

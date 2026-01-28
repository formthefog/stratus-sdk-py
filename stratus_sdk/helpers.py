"""
Production helper utilities.

Tools for caching, rate limiting, health monitoring, and retry logic.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, TypeVar

from .client import MJepaGClient

T = TypeVar("T")


class SimpleCache:
    """Simple in-memory cache with TTL."""

    def __init__(self, ttl_seconds: int = 300):
        """
        Initialize cache.

        Args:
            ttl_seconds: Time-to-live in seconds
        """
        self.ttl = ttl_seconds
        self._cache: Dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[Any]:
        """
        Get cached value.

        Args:
            key: Cache key

        Returns:
            Cached value or None if expired/missing
        """
        if key not in self._cache:
            return None

        value, timestamp = self._cache[key]

        # Check TTL
        if time.time() - timestamp > self.ttl:
            del self._cache[key]
            return None

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set cached value.

        Args:
            key: Cache key
            value: Value to cache
        """
        self._cache[key] = (value, time.time())

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def size(self) -> int:
        """Get number of cached items."""
        return len(self._cache)


class RateLimiter:
    """Rate limiter using token bucket algorithm."""

    def __init__(self, max_requests_per_second: float = 10.0):
        """
        Initialize rate limiter.

        Args:
            max_requests_per_second: Maximum requests per second
        """
        self.max_tokens = max_requests_per_second
        self.tokens = max_requests_per_second
        self.refill_rate = max_requests_per_second
        self.last_refill = time.time()

    async def acquire(self) -> bool:
        """
        Attempt to acquire a token.

        Returns:
            True if acquired, False if rate limited
        """
        self._refill()

        if self.tokens >= 1:
            self.tokens -= 1
            return True

        return False

    async def wait(self) -> None:
        """Wait until a token is available."""
        while not await self.acquire():
            await asyncio.sleep(0.1)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate

        self.tokens = min(self.tokens + tokens_to_add, self.max_tokens)
        self.last_refill = now

    def reset(self) -> None:
        """Reset rate limiter."""
        self.tokens = self.max_tokens
        self.last_refill = time.time()


class HealthChecker:
    """API health monitor."""

    def __init__(
        self,
        client: MJepaGClient,
        check_interval_seconds: int = 60,
        on_unhealthy: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize health checker.

        Args:
            client: M-JEPA-G client
            check_interval_seconds: Check interval
            on_unhealthy: Callback for unhealthy status
        """
        self.client = client
        self.check_interval = check_interval_seconds
        self.on_unhealthy = on_unhealthy
        self._monitoring_task: Optional[asyncio.Task] = None

    async def check(self) -> Dict[str, Any]:
        """
        Check API health.

        Returns:
            Health status dict
        """
        try:
            health = await self.client.health()

            return {
                "healthy": health.get("status") == "healthy",
                "model_loaded": health.get("model_loaded", False),
                "error": None,
            }
        except Exception as e:
            if self.on_unhealthy:
                self.on_unhealthy()

            return {"healthy": False, "model_loaded": False, "error": str(e)}

    async def start_monitoring(self) -> None:
        """Start periodic health checks."""
        if self._monitoring_task is not None:
            return

        async def monitor():
            while True:
                await self.check()
                await asyncio.sleep(self.check_interval)

        self._monitoring_task = asyncio.create_task(monitor())

    async def stop_monitoring(self) -> None:
        """Stop health checks."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None


async def retry_with_backoff(
    fn: Callable[[], T],
    max_retries: int = 3,
    initial_delay_ms: int = 1000,
    max_delay_ms: int = 10000,
    backoff_multiplier: float = 2.0,
) -> T:
    """
    Retry function with exponential backoff.

    Args:
        fn: Async function to retry
        max_retries: Maximum retry attempts
        initial_delay_ms: Initial delay in milliseconds
        max_delay_ms: Maximum delay in milliseconds
        backoff_multiplier: Delay multiplier on each retry

    Returns:
        Function result

    Raises:
        Last exception if all retries fail

    Example:
        >>> result = await retry_with_backoff(
        ...     lambda: client.rollout(...),
        ...     max_retries=3
        ... )
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            return await fn()
        except Exception as e:
            last_error = e

            # Don't retry on certain errors
            error_msg = str(e).lower()
            if any(x in error_msg for x in ["401", "400", "authentication", "invalid"]):
                raise

            if attempt < max_retries - 1:
                delay = min(
                    initial_delay_ms * (backoff_multiplier**attempt), max_delay_ms
                )
                await asyncio.sleep(delay / 1000)

    raise last_error or Exception("Retry failed")


def generate_cache_key(params: Dict[str, Any]) -> str:
    """
    Generate cache key from parameters.

    Args:
        params: Parameter dict

    Returns:
        Cache key string
    """
    import json

    sorted_params = dict(sorted(params.items()))
    return json.dumps(sorted_params, sort_keys=True)

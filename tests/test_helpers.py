"""
Tests for production helper utilities.
"""

import asyncio
import time
import pytest
import respx
import httpx

from stratus_sdk import StratusClient
from stratus_sdk.helpers import SimpleCache, RateLimiter, HealthChecker, CreditMonitor, retry_with_backoff
from stratus_sdk.exceptions import AuthenticationError, APIError


BASE_URL = "https://api.stratus.run"


# --- SimpleCache ---

def test_simple_cache_set_and_get():
    cache = SimpleCache(ttl_seconds=60)
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"


def test_simple_cache_miss():
    cache = SimpleCache(ttl_seconds=60)
    assert cache.get("nonexistent") is None


def test_simple_cache_ttl_expiry():
    cache = SimpleCache(ttl_seconds=0)  # Expires immediately
    cache.set("key1", "value1")
    time.sleep(0.01)
    assert cache.get("key1") is None


def test_simple_cache_clear():
    cache = SimpleCache()
    cache.set("k1", "v1")
    cache.set("k2", "v2")
    assert cache.size() == 2
    cache.clear()
    assert cache.size() == 0
    assert cache.get("k1") is None


def test_simple_cache_size():
    cache = SimpleCache()
    assert cache.size() == 0
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.size() == 2


# --- RateLimiter ---

@pytest.mark.asyncio
async def test_rate_limiter_acquire():
    limiter = RateLimiter(max_requests_per_second=10.0)
    result = await limiter.acquire()
    assert result is True


@pytest.mark.asyncio
async def test_rate_limiter_exhaustion():
    limiter = RateLimiter(max_requests_per_second=2.0)
    # Drain all tokens
    for _ in range(2):
        await limiter.acquire()
    # Next acquire should fail (no tokens left)
    result = await limiter.acquire()
    assert result is False


@pytest.mark.asyncio
async def test_rate_limiter_wait():
    limiter = RateLimiter(max_requests_per_second=100.0)
    # Should complete quickly with enough tokens
    start = time.time()
    await limiter.wait()
    elapsed = time.time() - start
    assert elapsed < 1.0


def test_rate_limiter_reset():
    limiter = RateLimiter(max_requests_per_second=5.0)
    limiter.tokens = 0
    limiter.reset()
    assert limiter.tokens == pytest.approx(5.0)


# --- HealthChecker ---

@pytest.mark.asyncio
@respx.mock
async def test_health_checker_check_success():
    respx.get(f"{BASE_URL}/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy", "model_loaded": True})
    )
    async with StratusClient(api_key="sk-test") as client:
        checker = HealthChecker(client)
        result = await checker.check()

    assert result["healthy"] is True
    assert result["model_loaded"] is True
    assert result["error"] is None


@pytest.mark.asyncio
@respx.mock
async def test_health_checker_check_failure():
    respx.get(f"{BASE_URL}/health").mock(side_effect=httpx.ConnectError("refused"))
    callback_called = []

    async with StratusClient(api_key="sk-test") as client:
        checker = HealthChecker(client, on_unhealthy=lambda: callback_called.append(True))
        result = await checker.check()

    assert result["healthy"] is False
    assert result["error"] is not None
    assert len(callback_called) == 1


@pytest.mark.asyncio
@respx.mock
async def test_health_checker_monitoring_start_stop():
    respx.get(f"{BASE_URL}/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy", "model_loaded": True})
    )
    async with StratusClient(api_key="sk-test") as client:
        checker = HealthChecker(client, check_interval_seconds=1)
        await checker.start_monitoring()
        assert checker._monitoring_task is not None
        await asyncio.sleep(0.01)
        await checker.stop_monitoring()
        assert checker._monitoring_task is None


# --- CreditMonitor ---

@pytest.mark.asyncio
@respx.mock
async def test_credit_monitor_warning_threshold():
    respx.get(f"{BASE_URL}/v1/credits/balance").mock(
        return_value=httpx.Response(200, json={"balance": 5.0})
    )
    warning_values = []
    async with StratusClient(api_key="sk-test") as client:
        monitor = CreditMonitor(
            client,
            warning_threshold=10.0,
            critical_threshold=1.0,
            on_warning=lambda b: warning_values.append(b),
        )
        balance = await monitor.check()

    assert balance == 5.0
    assert warning_values == [5.0]
    assert monitor.last_balance == 5.0


@pytest.mark.asyncio
@respx.mock
async def test_credit_monitor_critical_threshold():
    respx.get(f"{BASE_URL}/v1/credits/balance").mock(
        return_value=httpx.Response(200, json={"balance": 0.5})
    )
    critical_values = []
    async with StratusClient(api_key="sk-test") as client:
        monitor = CreditMonitor(
            client,
            warning_threshold=10.0,
            critical_threshold=1.0,
            on_critical=lambda b: critical_values.append(b),
        )
        balance = await monitor.check()

    assert balance == 0.5
    assert critical_values == [0.5]


@pytest.mark.asyncio
@respx.mock
async def test_credit_monitor_no_callback_above_threshold():
    respx.get(f"{BASE_URL}/v1/credits/balance").mock(
        return_value=httpx.Response(200, json={"balance": 100.0})
    )
    warning_called = []
    critical_called = []
    async with StratusClient(api_key="sk-test") as client:
        monitor = CreditMonitor(
            client,
            warning_threshold=10.0,
            critical_threshold=1.0,
            on_warning=lambda b: warning_called.append(b),
            on_critical=lambda b: critical_called.append(b),
        )
        balance = await monitor.check()

    assert balance == 100.0
    assert len(warning_called) == 0
    assert len(critical_called) == 0


# --- retry_with_backoff ---

@pytest.mark.asyncio
async def test_retry_with_backoff_succeeds():
    call_count = 0

    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise APIError("server error", 500)
        return "success"

    result = await retry_with_backoff(flaky, max_retries=3, initial_delay_ms=1, max_delay_ms=10)
    assert result == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_with_backoff_exhausted():
    call_count = 0

    async def always_fails():
        nonlocal call_count
        call_count += 1
        raise APIError("server error", 500)

    with pytest.raises(APIError):
        await retry_with_backoff(always_fails, max_retries=3, initial_delay_ms=1, max_delay_ms=10)

    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_with_backoff_no_retry_on_auth():
    call_count = 0

    async def auth_error():
        nonlocal call_count
        call_count += 1
        raise AuthenticationError("bad key")

    with pytest.raises(AuthenticationError):
        await retry_with_backoff(auth_error, max_retries=3, initial_delay_ms=1, max_delay_ms=10)

    assert call_count == 1

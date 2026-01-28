"""
Tests for M-JEPA-G client.
"""

import pytest

from stratus_sdk import MJepaGClient, CompressionLevel
from stratus_sdk.exceptions import AuthenticationError


@pytest.mark.asyncio
async def test_client_initialization():
    """Test client initialization."""
    client = MJepaGClient(api_key="test-key")

    assert client.api_key == "test-key"
    assert client.timeout == 30.0
    assert client.retries == 3

    await client.close()


@pytest.mark.asyncio
async def test_compression_profile():
    """Test compression profile setting."""
    client = MJepaGClient(
        api_key="test-key", compression_profile=CompressionLevel.HIGH
    )

    assert client.compression_profile == CompressionLevel.HIGH
    assert "x" in client.get_compression_ratio()
    assert client.get_quality_score() > 99.0

    await client.close()


@pytest.mark.asyncio
async def test_context_manager():
    """Test async context manager."""
    async with MJepaGClient(api_key="test-key") as client:
        assert client.api_key == "test-key"

    # Client should be closed after exiting context

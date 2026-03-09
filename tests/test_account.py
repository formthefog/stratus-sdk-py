"""
Tests for account LLM key management endpoints.
"""

import pytest
import respx
import httpx

from stratus_sdk import StratusClient
from stratus_sdk.exceptions import AuthenticationError


BASE_URL = "https://api.stratus.run"


@pytest.mark.asyncio
@respx.mock
async def test_llm_keys_set_success():
    respx.post(f"{BASE_URL}/v1/account/llm-keys").mock(
        return_value=httpx.Response(
            200,
            json={
                "success": True,
                "configured": {"openai": True, "anthropic": False, "openrouter": False},
            },
        )
    )
    async with StratusClient(api_key="sk-test") as client:
        result = await client.account.llm_keys.set(openai_key="sk-openai-xxx")

    assert result.success is True
    assert result.configured.openai is True
    assert result.configured.anthropic is False


@pytest.mark.asyncio
@respx.mock
async def test_llm_keys_set_multiple_providers():
    respx.post(f"{BASE_URL}/v1/account/llm-keys").mock(
        return_value=httpx.Response(
            200,
            json={
                "success": True,
                "configured": {"openai": True, "anthropic": True, "openrouter": True},
            },
        )
    )
    async with StratusClient(api_key="sk-test") as client:
        result = await client.account.llm_keys.set(
            openai_key="sk-openai",
            anthropic_key="sk-ant-xxx",
            openrouter_key="sk-or-xxx",
        )

    assert result.configured.openai is True
    assert result.configured.anthropic is True
    assert result.configured.openrouter is True


@pytest.mark.asyncio
@respx.mock
async def test_llm_keys_set_invalid_key():
    respx.post(f"{BASE_URL}/v1/account/llm-keys").mock(
        return_value=httpx.Response(
            401,
            json={"error": {"message": "Invalid API key"}},
        )
    )
    async with StratusClient(api_key="bad") as client:
        with pytest.raises(AuthenticationError):
            await client.account.llm_keys.set(openai_key="invalid")


@pytest.mark.asyncio
@respx.mock
async def test_llm_keys_get_returns_status_flags():
    respx.get(f"{BASE_URL}/v1/account/llm-keys").mock(
        return_value=httpx.Response(
            200,
            json={"openai": True, "anthropic": False, "openrouter": False},
        )
    )
    async with StratusClient(api_key="sk-test") as client:
        status = await client.account.llm_keys.get()

    assert status.openai is True
    assert status.anthropic is False
    assert status.openrouter is False


@pytest.mark.asyncio
@respx.mock
async def test_llm_keys_get_all_configured():
    respx.get(f"{BASE_URL}/v1/account/llm-keys").mock(
        return_value=httpx.Response(
            200,
            json={"openai": True, "anthropic": True, "openrouter": True},
        )
    )
    async with StratusClient(api_key="sk-test") as client:
        status = await client.account.llm_keys.get()

    assert status.openai is True
    assert status.anthropic is True
    assert status.openrouter is True


@pytest.mark.asyncio
@respx.mock
async def test_llm_keys_delete_specific_provider():
    respx.delete(f"{BASE_URL}/v1/account/llm-keys").mock(
        return_value=httpx.Response(200, json={"success": True, "deleted": "openai"})
    )
    async with StratusClient(api_key="sk-test") as client:
        result = await client.account.llm_keys.delete(provider="openai")

    assert result["success"] is True


@pytest.mark.asyncio
@respx.mock
async def test_llm_keys_delete_all():
    respx.delete(f"{BASE_URL}/v1/account/llm-keys").mock(
        return_value=httpx.Response(200, json={"success": True, "deleted": "all"})
    )
    async with StratusClient(api_key="sk-test") as client:
        result = await client.account.llm_keys.delete()

    assert result["success"] is True

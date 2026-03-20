"""
Tests for Stratus client.
"""

import pytest
import respx
import httpx

from stratus_sdk import StratusClient, MJepaGClient, CompressionLevel
from stratus_sdk.exceptions import AuthenticationError, RateLimitError, APIError, TimeoutError


BASE_URL = "https://api.stratus.run"


def make_chat_response(content: str = "Hello there!") -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "stratus-x1-ac",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


# --- Initialization ---

@pytest.mark.asyncio
async def test_client_initialization_defaults():
    client = StratusClient(api_key="sk-test")
    assert client.api_key == "sk-test"
    assert client.api_url == "https://api.stratus.run"
    assert client.timeout == 30.0
    assert client.retries == 3
    await client.close()


@pytest.mark.asyncio
async def test_client_initialization_custom():
    client = StratusClient(
        api_key="sk-test",
        api_url="https://custom.example.com/",
        timeout=60.0,
        retries=5,
    )
    assert client.api_url == "https://custom.example.com"
    assert client.timeout == 60.0
    assert client.retries == 5
    await client.close()


@pytest.mark.asyncio
async def test_client_headers_contain_both_auth_forms():
    client = StratusClient(api_key="my-key")
    headers = client._client.headers
    assert headers.get("authorization") == "Bearer my-key"
    assert headers.get("x-api-key") == "my-key"
    await client.close()


@pytest.mark.asyncio
async def test_mjepa_client_alias():
    client = MJepaGClient(api_key="sk-test")
    assert isinstance(client, StratusClient)
    await client.close()


# --- Compression profile ---

@pytest.mark.asyncio
async def test_compression_profile():
    client = StratusClient(api_key="sk-test", compression_profile=CompressionLevel.HIGH)
    assert client.compression_profile == CompressionLevel.HIGH
    assert "x" in client.get_compression_ratio()
    assert client.get_quality_score() > 99.0
    await client.close()


# --- Context manager ---

@pytest.mark.asyncio
async def test_context_manager():
    async with StratusClient(api_key="sk-test") as client:
        assert client.api_key == "sk-test"


# --- Health ---

@pytest.mark.asyncio
@respx.mock
async def test_health_success():
    respx.get(f"{BASE_URL}/health").mock(
        return_value=httpx.Response(200, json={"status": "healthy", "model_loaded": True})
    )
    async with StratusClient(api_key="sk-test") as client:
        result = await client.health()
    assert result["status"] == "healthy"
    assert result["model_loaded"] is True


@pytest.mark.asyncio
@respx.mock
async def test_health_server_error():
    respx.get(f"{BASE_URL}/health").mock(
        return_value=httpx.Response(503, json={"status": "unhealthy"})
    )
    async with StratusClient(api_key="sk-test") as client:
        result = await client.health()
    assert result["status"] == "unhealthy"


# --- Chat completions ---

@pytest.mark.asyncio
@respx.mock
async def test_chat_completions_create_success():
    respx.post(f"{BASE_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=make_chat_response())
    )
    async with StratusClient(api_key="sk-test") as client:
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
            model="stratus-x1-ac",
        )
    assert response.model == "stratus-x1-ac"
    assert response.choices[0].message.content == "Hello there!"


@pytest.mark.asyncio
@respx.mock
async def test_chat_completions_create_with_tools():
    respx.post(f"{BASE_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=make_chat_response())
    )
    async with StratusClient(api_key="sk-test") as client:
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
            model="stratus-x1-ac",
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            tool_choice="auto",
            stratus={"mode": "plan", "validation_threshold": 0.8},
        )
    assert response.id == "chatcmpl-test"


@pytest.mark.asyncio
@respx.mock
async def test_chat_completions_401():
    respx.post(f"{BASE_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": "Unauthorized"})
    )
    async with StratusClient(api_key="bad-key") as client:
        with pytest.raises(AuthenticationError):
            await client.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}],
                model="stratus-x1-ac",
            )


@pytest.mark.asyncio
@respx.mock
async def test_chat_completions_429():
    respx.post(f"{BASE_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": "Rate limit"})
    )
    async with StratusClient(api_key="sk-test") as client:
        with pytest.raises(RateLimitError):
            await client.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}],
                model="stratus-x1-ac",
            )


@pytest.mark.asyncio
@respx.mock
async def test_chat_completions_500_raises_api_error():
    # 500 retries; mock to always return 500 to exhaust retries fast
    respx.post(f"{BASE_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": "Internal Server Error"})
    )
    async with StratusClient(api_key="sk-test", retries=1) as client:
        with pytest.raises(APIError):
            await client.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}],
                model="stratus-x1-ac",
            )


@pytest.mark.asyncio
@respx.mock
async def test_chat_completions_invalid_json_response():
    respx.post(f"{BASE_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=b"not json")
    )
    async with StratusClient(api_key="sk-test") as client:
        with pytest.raises(Exception):
            await client.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}],
                model="stratus-x1-ac",
            )


# --- Streaming ---

@pytest.mark.asyncio
@respx.mock
async def test_chat_completions_stream():
    sse_data = (
        "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\","
        "\"created\":1700000000,\"model\":\"stratus-x1-ac\","
        "\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n"
        "data: [DONE]\n\n"
    )
    respx.post(f"{BASE_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=sse_data)
    )
    chunks = []
    async with StratusClient(api_key="sk-test") as client:
        async for chunk in client.chat.completions.stream(
            messages=[{"role": "user", "content": "Hi"}],
            model="stratus-x1-ac",
        ):
            chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0].id == "c1"


@pytest.mark.asyncio
@respx.mock
async def test_chat_completions_stream_skips_malformed():
    sse_data = (
        "data: not-valid-json\n\n"
        "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\","
        "\"created\":1700000000,\"model\":\"m\",\"choices\":[]}\n\n"
        "data: [DONE]\n\n"
    )
    respx.post(f"{BASE_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=sse_data)
    )
    chunks = []
    async with StratusClient(api_key="sk-test") as client:
        async for chunk in client.chat.completions.stream(
            messages=[{"role": "user", "content": "Hi"}], model="m"
        ):
            chunks.append(chunk)
    assert len(chunks) == 1


# --- Embeddings ---

@pytest.mark.asyncio
@respx.mock
async def test_embeddings_single_string():
    respx.post(f"{BASE_URL}/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
                "model": "stratus-embed",
            },
        )
    )
    async with StratusClient(api_key="sk-test") as client:
        result = await client.embeddings(model="stratus-embed", input="hello")
    assert len(result.data) == 1
    assert result.data[0].embedding == [0.1, 0.2]


@pytest.mark.asyncio
@respx.mock
async def test_embeddings_list_of_strings():
    respx.post(f"{BASE_URL}/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "embedding": [0.1], "index": 0},
                    {"object": "embedding", "embedding": [0.2], "index": 1},
                ],
                "model": "stratus-embed",
            },
        )
    )
    async with StratusClient(api_key="sk-test") as client:
        result = await client.embeddings(model="stratus-embed", input=["a", "b"])
    assert len(result.data) == 2


# --- List models ---

@pytest.mark.asyncio
@respx.mock
async def test_list_models():
    respx.get(f"{BASE_URL}/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": "stratus-x1-ac", "object": "model"},
                    {"id": "stratus-embed", "object": "model"},
                ],
            },
        )
    )
    async with StratusClient(api_key="sk-test") as client:
        models = await client.list_models()
    assert len(models) == 2
    assert models[0].id == "stratus-x1-ac"


# --- Rollout ---

@pytest.mark.asyncio
@respx.mock
async def test_rollout_success():
    respx.post(f"{BASE_URL}/v1/rollout").mock(
        return_value=httpx.Response(
            200,
            json={
                "predictions": [
                    {
                        "step": 1,
                        "predicted_state": {"step": 1, "magnitude": 0.8, "confidence": "high"},
                        "action": {"action_id": 1, "action_name": "do thing"},
                        "state_change": 0.5,
                        "brain_confidence": 0.9,
                    }
                ],
                "summary": {"total_steps": 1, "outcome": "success", "final_magnitude": 0.8},
                "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
            },
        )
    )
    async with StratusClient(api_key="sk-test") as client:
        result = await client.rollout(goal="reach goal", initial_state="start")
    assert len(result.predictions) == 1
    assert result.summary.outcome == "success"


@pytest.mark.asyncio
@respx.mock
async def test_rollout_invalid_response():
    respx.post(f"{BASE_URL}/v1/rollout").mock(
        return_value=httpx.Response(200, json={"unexpected": "data"})
    )
    async with StratusClient(api_key="sk-test") as client:
        with pytest.raises(APIError):
            await client.rollout(goal="g", initial_state="s")


# --- Retry logic ---

@pytest.mark.asyncio
@respx.mock
async def test_retry_on_500_then_success():
    call_count = 0

    def side_effect(request):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return httpx.Response(500)
        return httpx.Response(200, json=make_chat_response())

    respx.post(f"{BASE_URL}/v1/chat/completions").mock(side_effect=side_effect)
    async with StratusClient(api_key="sk-test", retries=3) as client:
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
            model="stratus-x1-ac",
        )
    assert response.id == "chatcmpl-test"
    assert call_count == 3


@pytest.mark.asyncio
@respx.mock
async def test_no_retry_on_401():
    call_count = 0

    def side_effect(request):
        nonlocal call_count
        call_count += 1
        return httpx.Response(401)

    respx.post(f"{BASE_URL}/v1/chat/completions").mock(side_effect=side_effect)
    async with StratusClient(api_key="bad-key", retries=3) as client:
        with pytest.raises(AuthenticationError):
            await client.chat.completions.create(
                messages=[{"role": "user", "content": "Hi"}],
                model="stratus-x1-ac",
            )
    assert call_count == 1


# --- Anthropic messages ---

@pytest.mark.asyncio
@respx.mock
async def test_messages_success():
    respx.post(f"{BASE_URL}/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello!"}],
                "model": "stratus-x1-ac",
                "stop_reason": "end_turn",
            },
        )
    )
    async with StratusClient(api_key="sk-test") as client:
        result = await client.messages(
            model="stratus-x1-ac",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100,
        )
    assert result.id == "msg-test"
    assert result.content[0].text == "Hello!"

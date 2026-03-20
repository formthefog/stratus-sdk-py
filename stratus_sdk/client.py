"""
Stratus API Client.

Type-safe async client for the Stratus X1 M-JEPA-G API.
"""

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx
from pydantic import ValidationError as PydanticValidationError

from .exceptions import APIError, AuthenticationError, RateLimitError, TimeoutError
from .profiles import CompressionLevel, get_mjepa_profile
from .types import (
    AnthropicResponse,
    ChatCompletionChunk,
    ChatCompletionResponse,
    CreditPackage,
    CreditPurchaseResponse,
    EmbeddingResponse,
    LLMKeySetResponse,
    LLMKeyStatus,
    ModelInfo,
    RolloutResponse,
)


class StratusClient:
    """
    Stratus API Client.

    Provides type-safe access to Stratus X1 endpoints:
    - Chat completions (OpenAI-compatible)
    - Anthropic-compatible messages
    - Embeddings
    - State rollout (trajectory prediction)
    - LLM key management
    - Credits
    - Streaming support
    - Built-in error handling and retries

    Example:
        >>> client = StratusClient(api_key="sk-stratus-...")
        >>> response = await client.chat.completions.create(
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     model="stratus-x1-ac"
        ... )
    """

    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api.stratus.run",
        timeout: float = 30.0,
        retries: int = 3,
        compression_profile: CompressionLevel = CompressionLevel.MEDIUM,
    ):
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.compression_profile = compression_profile

        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "x-api-key": api_key,
                "Content-Type": "application/json",
            },
        )

        self.chat = _ChatNamespace(self)
        self.account = _AccountNamespace(self)
        self.credits = _CreditsNamespace(self)

    async def __aenter__(self) -> "StratusClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def _request(
        self, method: str, endpoint: str, **kwargs: Any
    ) -> httpx.Response:
        url = f"{self.api_url}{endpoint}"
        last_error: Optional[Exception] = None

        for attempt in range(self.retries):
            try:
                response = await self._client.request(method, url, **kwargs)

                if response.status_code == 200:
                    return response
                elif response.status_code == 401:
                    raise AuthenticationError("Invalid API key")
                elif response.status_code == 429:
                    raise RateLimitError("Rate limit exceeded")
                elif response.status_code >= 500:
                    if attempt < self.retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    raise APIError(f"Server error: {response.status_code}", response.status_code)
                else:
                    try:
                        error_body = response.json()
                        msg = error_body.get("error", {}).get("message", "Unknown error")
                    except Exception:
                        msg = f"HTTP {response.status_code}"
                    raise APIError(msg, response.status_code)

            except (AuthenticationError, RateLimitError):
                raise
            except httpx.TimeoutException as e:
                last_error = TimeoutError(f"Request timeout: {e}")
                if attempt < self.retries - 1:
                    continue
                raise last_error
            except httpx.RequestError as e:
                last_error = APIError(f"Request failed: {e}")
                if attempt < self.retries - 1:
                    continue
                raise last_error

        raise last_error or APIError("Request failed after retries")

    async def rollout(
        self,
        goal: str,
        initial_state: str,
        max_steps: int = 10,
        return_intermediate: bool = True,
    ) -> RolloutResponse:
        """Predict state trajectory (rollout)."""
        response = await self._request(
            "POST",
            "/v1/rollout",
            json={
                "goal": goal,
                "initial_state": initial_state,
                "max_steps": max_steps,
                "return_intermediate": return_intermediate,
            },
        )
        try:
            return RolloutResponse(**response.json())
        except PydanticValidationError as e:
            raise APIError(f"Invalid response format: {e}")

    async def messages(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        system: Optional[str] = None,
        stream: bool = False,
    ) -> AnthropicResponse:
        """
        Anthropic-compatible messages endpoint.

        Args:
            model: Model name
            messages: List of message dicts
            max_tokens: Max tokens to generate
            system: Optional system prompt
            stream: Enable streaming (not yet supported here)
        """
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if system is not None:
            body["system"] = system

        response = await self._request("POST", "/v1/messages", json=body)
        try:
            return AnthropicResponse(**response.json())
        except PydanticValidationError as e:
            raise APIError(f"Invalid response format: {e}")

    async def embeddings(
        self,
        model: str,
        input: Union[str, List[str]],
    ) -> EmbeddingResponse:
        """
        Create embeddings.

        Args:
            model: Model name
            input: String or list of strings to embed
        """
        response = await self._request(
            "POST",
            "/v1/embeddings",
            json={"model": model, "input": input},
        )
        try:
            return EmbeddingResponse(**response.json())
        except PydanticValidationError as e:
            raise APIError(f"Invalid response format: {e}")

    async def list_models(self) -> List[ModelInfo]:
        """List available models."""
        response = await self._request("GET", "/v1/models")
        data = response.json()
        models_data = data.get("data", data) if isinstance(data, dict) else data
        return [ModelInfo(**m) for m in models_data]

    async def health(self) -> Dict[str, Any]:
        """Check API health."""
        response = await self._client.get(f"{self.api_url}/health", timeout=5.0)
        return response.json()  # type: ignore[no-any-return]

    def get_compression_ratio(self) -> str:
        """Get estimated compression ratio for current profile."""
        profile = get_mjepa_profile(self.compression_profile)
        return f"{profile.ratio}x"

    def get_quality_score(self) -> float:
        """Get estimated quality score for current profile."""
        profile = get_mjepa_profile(self.compression_profile)
        return profile.quality


# Alias for backwards compatibility
MJepaGClient = StratusClient


class _ChatCompletions:
    """Chat completions namespace (OpenAI-compatible)."""

    def __init__(self, client: StratusClient):
        self._client = client

    async def create(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        stratus: Optional[Dict[str, Any]] = None,
        openai_key: Optional[str] = None,
        anthropic_key: Optional[str] = None,
        openrouter_key: Optional[str] = None,
    ) -> ChatCompletionResponse:
        """Create chat completion."""
        if stream:
            raise NotImplementedError("Use stream() method for streaming")

        body: Dict[str, Any] = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if stratus is not None:
            body["stratus"] = stratus
        if openai_key is not None:
            body["openai_key"] = openai_key
        if anthropic_key is not None:
            body["anthropic_key"] = anthropic_key
        if openrouter_key is not None:
            body["openrouter_key"] = openrouter_key

        response = await self._client._request("POST", "/v1/chat/completions", json=body)
        try:
            return ChatCompletionResponse(**response.json())
        except PydanticValidationError as e:
            raise APIError(f"Invalid response format: {e}")

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Stream chat completion."""
        async with self._client._client.stream(
            "POST",
            f"{self._client.api_url}/v1/chat/completions",
            json={
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            },
        ) as response:
            async for line in response.aiter_lines():
                if not line.strip() or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                    try:
                        chunk_data = json.loads(data)
                        yield ChatCompletionChunk(**chunk_data)
                    except (json.JSONDecodeError, PydanticValidationError):
                        continue


class _ChatNamespace:
    """Chat namespace (client.chat)."""

    def __init__(self, client: StratusClient):
        self.completions = _ChatCompletions(client)


class _LLMKeys:
    """LLM key management namespace (client.account.llm_keys)."""

    def __init__(self, client: StratusClient):
        self._client = client

    async def set(
        self,
        openai_key: Optional[str] = None,
        anthropic_key: Optional[str] = None,
        openrouter_key: Optional[str] = None,
    ) -> LLMKeySetResponse:
        """Set LLM provider API keys."""
        body: Dict[str, Any] = {}
        if openai_key is not None:
            body["openai_key"] = openai_key
        if anthropic_key is not None:
            body["anthropic_key"] = anthropic_key
        if openrouter_key is not None:
            body["openrouter_key"] = openrouter_key

        response = await self._client._request("POST", "/v1/account/llm-keys", json=body)
        try:
            return LLMKeySetResponse(**response.json())
        except PydanticValidationError as e:
            raise APIError(f"Invalid response format: {e}")

    async def get(self) -> LLMKeyStatus:
        """Get configured LLM key status."""
        response = await self._client._request("GET", "/v1/account/llm-keys")
        try:
            return LLMKeyStatus(**response.json())
        except PydanticValidationError as e:
            raise APIError(f"Invalid response format: {e}")

    async def delete(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Delete LLM provider key(s).

        Args:
            provider: Provider name to delete ('openai', 'anthropic', 'openrouter').
                      If None, deletes all configured keys.
        """
        body: Dict[str, Any] = {}
        if provider is not None:
            body["provider"] = provider

        response = await self._client._request("DELETE", "/v1/account/llm-keys", json=body)
        return response.json()  # type: ignore[no-any-return]


class _AccountNamespace:
    """Account namespace (client.account)."""

    def __init__(self, client: StratusClient):
        self.llm_keys = _LLMKeys(client)


class _CreditsNamespace:
    """Credits namespace (client.credits)."""

    def __init__(self, client: StratusClient):
        self._client = client

    async def packages(self) -> List[CreditPackage]:
        """List available credit packages."""
        response = await self._client._request("GET", "/v1/credits/packages")
        data = response.json()
        packages_data = data.get("packages", data.get("data", data)) if isinstance(data, dict) else data
        if packages_data is None:
            packages_data = []
        return [CreditPackage(**p) for p in packages_data]

    async def purchase(
        self,
        package_name: str,
        payment_header: str,
    ) -> CreditPurchaseResponse:
        """
        Purchase a credit package.

        Args:
            package_name: Name of the package to purchase
            payment_header: Payment authorization header value
        """
        response = await self._client._request(
            "POST",
            "/v1/credits/purchase",
            json={"package_name": package_name},
            headers={"X-Payment": payment_header},
        )
        try:
            return CreditPurchaseResponse(**response.json())
        except PydanticValidationError as e:
            raise APIError(f"Invalid response format: {e}")


# Legacy name for ChatCompletions (used in older code)
ChatCompletions = _ChatCompletions

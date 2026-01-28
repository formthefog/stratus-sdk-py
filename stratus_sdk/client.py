"""
M-JEPA-G API Client.

Type-safe async client for the Stratus X1 M-JEPA-G API.
"""

import json
from typing import AsyncIterator, Dict, List, Optional

import httpx
from pydantic import ValidationError as PydanticValidationError

from .exceptions import APIError, AuthenticationError, RateLimitError, TimeoutError
from .profiles import CompressionLevel, get_mjepa_profile
from .types import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    Message,
    RolloutResponse,
)


class MJepaGClient:
    """
    M-JEPA-G API Client.

    Provides type-safe access to M-JEPA-G endpoints:
    - Chat completions (OpenAI-compatible)
    - State rollout (trajectory prediction)
    - Streaming support
    - Built-in error handling and retries

    Example:
        >>> client = MJepaGClient(api_key="sk-stratus-...")
        >>> response = await client.chat.completions.create(
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     model="stratus-x1-ac"
        ... )
    """

    def __init__(
        self,
        api_key: str,
        api_url: str = "http://212.115.124.137:8000",
        timeout: float = 30.0,
        retries: int = 3,
        compression_profile: CompressionLevel = CompressionLevel.MEDIUM,
    ):
        """
        Initialize M-JEPA-G client.

        Args:
            api_key: Stratus API key
            api_url: API base URL (default: deployed server)
            timeout: Request timeout in seconds
            retries: Number of retries on failure
            compression_profile: Compression quality level
        """
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.compression_profile = compression_profile

        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        # Chat completions namespace (OpenAI-compatible)
        self.chat = ChatCompletions(self)

    async def __aenter__(self) -> "MJepaGClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def _request(
        self, method: str, endpoint: str, **kwargs
    ) -> httpx.Response:
        """
        Make HTTP request with retries.

        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request arguments

        Returns:
            HTTP response

        Raises:
            AuthenticationError: Invalid API key
            RateLimitError: Rate limit exceeded
            TimeoutError: Request timeout
            APIError: Other API errors
        """
        url = f"{self.api_url}{endpoint}"
        last_error = None

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
                    # Server error - retry
                    if attempt < self.retries - 1:
                        await httpx.AsyncClient().get("https://httpbin.org/delay/1")
                        continue
                    raise APIError(f"Server error: {response.status_code}")
                else:
                    error_body = response.json()
                    raise APIError(
                        error_body.get("error", {}).get("message", "Unknown error"),
                        response.status_code,
                    )

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
        """
        Predict state trajectory (rollout).

        Args:
            goal: Goal to achieve
            initial_state: Initial state description
            max_steps: Maximum prediction steps
            return_intermediate: Return intermediate states

        Returns:
            RolloutResponse with predictions and summary

        Example:
            >>> result = await client.rollout(
            ...     goal="Increase stability to >80%",
            ...     initial_state="stability: 45%",
            ...     max_steps=5
            ... )
        """
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

    async def health(self) -> Dict[str, bool]:
        """
        Check API health.

        Returns:
            Health status dict

        Example:
            >>> status = await client.health()
            >>> print(status)  # {"status": "healthy", "model_loaded": True}
        """
        response = await self._client.get(f"{self.api_url}/health", timeout=5.0)
        return response.json()

    def get_compression_ratio(self) -> str:
        """Get estimated compression ratio for current profile."""
        profile = get_mjepa_profile(self.compression_profile)
        return f"{profile.ratio}x"

    def get_quality_score(self) -> float:
        """Get estimated quality score for current profile."""
        profile = get_mjepa_profile(self.compression_profile)
        return profile.quality


class ChatCompletions:
    """Chat completions namespace (OpenAI-compatible)."""

    def __init__(self, client: MJepaGClient):
        self._client = client

    async def create(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> ChatCompletionResponse:
        """
        Create chat completion.

        Args:
            messages: List of messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            stream: Enable streaming

        Returns:
            ChatCompletionResponse

        Example:
            >>> response = await client.chat.completions.create(
            ...     messages=[{"role": "user", "content": "Hello!"}],
            ...     model="stratus-x1-ac"
            ... )
        """
        if stream:
            raise NotImplementedError("Use stream() method for streaming")

        response = await self._client._request(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
        )

        try:
            return ChatCompletionResponse(**response.json())
        except PydanticValidationError as e:
            raise APIError(f"Invalid response format: {e}")

    async def stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """
        Stream chat completion.

        Args:
            messages: List of messages
            model: Model name
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Yields:
            ChatCompletionChunk objects

        Example:
            >>> async for chunk in client.chat.completions.stream(...):
            ...     print(chunk.choices[0].delta.get("content", ""), end="")
        """
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
                    data = line[6:]  # Remove "data: " prefix
                    try:
                        chunk_data = json.loads(data)
                        yield ChatCompletionChunk(**chunk_data)
                    except (json.JSONDecodeError, PydanticValidationError):
                        continue

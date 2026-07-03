from __future__ import annotations

import httpx
from loguru import logger

from .base import BaseLLMProvider, post_json_with_retry
from .security import is_local_url, validate_api_key, validate_transport_security


class AnthropicProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        model: str = "claude-3-5-sonnet-latest",
        max_tokens: int = 32768,
        allow_insecure_http: bool = False,
        anthropic_version: str = "2023-06-01",
    ):
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.default_model = model
        self.default_max_tokens = max_tokens
        self.anthropic_version = anthropic_version
        self._validate_api_key()
        validate_transport_security(
            self.base_url,
            env_key="LLM_BASE_URL",
            allow_insecure_http=allow_insecure_http,
        )

        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self.anthropic_version,
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        self._client = httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(600.0, connect=30.0))

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        model = model or self.default_model
        max_tokens = max_tokens or self.default_max_tokens
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        logger.debug(f"Anthropic Messages request: model={model}, prompt_len={len(prompt)}")
        data = await post_json_with_retry(
            self._client,
            self._endpoint(),
            payload,
            provider="Anthropic Messages",
        )
        content = self._extract_text(data)
        logger.debug(f"Anthropic Messages response: {len(content)} chars")
        return content

    async def close(self) -> None:
        await self._client.aclose()

    def _endpoint(self) -> str:
        if self.base_url.endswith("/messages"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/messages"
        return f"{self.base_url}/v1/messages"

    @staticmethod
    def _extract_text(data: dict) -> str:
        parts: list[str] = []
        for block in data.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)
        raise KeyError("Anthropic Messages payload did not include text content blocks")

    def _validate_api_key(self) -> None:
        if self.api_key or not is_local_url(self.base_url):
            validate_api_key(self.api_key, env_key="LLM_API_KEY", provider="anthropic")

from __future__ import annotations

import httpx
from loguru import logger

from .base import BaseLLMProvider, chat_completion_with_retry


class KimiProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions",
        model: str = "kimi-k2.6",
        max_tokens: int = 32768,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = model
        self.default_max_tokens = max_tokens
        self._client = httpx.AsyncClient(
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            timeout=httpx.Timeout(600.0, connect=30.0),
        )

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
            "stream": False,
        }
        logger.debug(f"Kimi request: model={model}, prompt_len={len(prompt)}")
        return await chat_completion_with_retry(
            self._client, self.base_url, payload, provider="Kimi"
        )

    async def close(self) -> None:
        await self._client.aclose()

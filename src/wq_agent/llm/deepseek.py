from __future__ import annotations

import httpx
from loguru import logger

from .base import BaseLLMProvider, chat_completion_with_retry


class DeepSeekProvider(BaseLLMProvider):
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1/chat/completions"):
        self.api_key = api_key
        self.base_url = base_url
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
        max_tokens: int = 4096,
    ) -> str:
        model = model or "deepseek-chat"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        logger.debug(f"DeepSeek request: model={model}, prompt_len={len(prompt)}")
        return await chat_completion_with_retry(
            self._client, self.base_url, payload, provider="DeepSeek"
        )

    async def close(self) -> None:
        await self._client.aclose()

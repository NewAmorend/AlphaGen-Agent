from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import httpx
from loguru import logger

# 瞬时错误（读超时 / 连接错误等）可重试；4xx 业务错误（除 429）直接抛。
_RETRYABLE_EXC = (httpx.TimeoutException, httpx.TransportError)
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


async def chat_completion_with_retry(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    *,
    provider: str,
    max_attempts: int = 3,
) -> str:
    """POST 一个 OpenAI 兼容的 chat completion，带瞬时错误重试 + 指数退避。

    解决生产批跑时单次 LLM 读超时直接 abort 整轮的问题。返回 message content。
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                logger.debug(f"{provider} response: {len(content)} chars")
                return content
            if resp.status_code in _RETRYABLE_STATUS:
                last_exc = Exception(f"{provider} API error ({resp.status_code}): {resp.text[:200]}")
                logger.warning(f"{provider} transient {resp.status_code}, attempt {attempt}/{max_attempts}")
            else:
                raise Exception(f"{provider} API error ({resp.status_code}): {resp.text[:500]}")
        except _RETRYABLE_EXC as exc:
            last_exc = exc
            logger.warning(f"{provider} {type(exc).__name__}, attempt {attempt}/{max_attempts}")
        if attempt < max_attempts:
            await asyncio.sleep(2 ** attempt)  # 2s, 4s
    raise last_exc or Exception(f"{provider}: retries exhausted")


class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...

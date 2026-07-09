from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod

import httpx
from loguru import logger

# 瞬时错误（读超时 / 连接错误等）可重试；4xx 业务错误（除 429）直接抛。
_RETRYABLE_EXC = (httpx.TimeoutException, httpx.TransportError)
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_SENSITIVE_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,\"']+"),
    re.compile(r"(?i)(\"?(?:api[_-]?key|token|secret)\"?\s*[:=]\s*\"?)[^\"\s,}]+"),
)


def _sanitize_error_body(body: str) -> str:
    sanitized = body
    for pattern in _SENSITIVE_PATTERNS:
        sanitized = pattern.sub(r"\1[REDACTED]", sanitized)
    return sanitized[:500]


class LLMAPIError(Exception):
    def __init__(self, provider: str, status_code: int, body: str):
        self.provider = provider
        self.status_code = status_code
        self.body = _sanitize_error_body(body)
        super().__init__(f"{provider} API error ({status_code}): {self.body}")


async def post_json_with_retry(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    *,
    provider: str,
    max_attempts: int = 3,
) -> dict:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in _RETRYABLE_STATUS:
                last_exc = LLMAPIError(provider, resp.status_code, resp.text[:200])
                logger.warning(f"{provider} transient {resp.status_code}, attempt {attempt}/{max_attempts}")
            else:
                raise LLMAPIError(provider, resp.status_code, resp.text)
        except _RETRYABLE_EXC as exc:
            last_exc = exc
            logger.warning(f"{provider} {type(exc).__name__}, attempt {attempt}/{max_attempts}")
        if attempt < max_attempts:
            await asyncio.sleep(2 ** attempt)  # 2s, 4s
    raise last_exc or Exception(f"{provider}: retries exhausted")


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
    data = await post_json_with_retry(
        client, url, payload, provider=provider, max_attempts=max_attempts
    )
    content = data["choices"][0]["message"]["content"]
    logger.debug(f"{provider} response: {len(content)} chars")
    return content


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

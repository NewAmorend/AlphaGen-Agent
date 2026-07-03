from __future__ import annotations

import httpx
from loguru import logger

from .base import BaseLLMProvider, LLMAPIError, chat_completion_with_retry, post_json_with_retry
from .security import is_local_url, validate_api_key, validate_transport_security


_RESPONSES_FALLBACK_STATUS = {404, 405, 501}
_RESPONSES_FALLBACK_PHRASES = (
    "not found",
    "no route",
    "unknown endpoint",
    "unsupported endpoint",
    "unsupported url",
    "cannot post /v1/responses",
    "invalid endpoint",
)
_REASONING_EFFORTS = {"", "none", "minimal", "low", "medium", "high", "xhigh"}
_CHAT_TOKEN_PARAMS = {"max_tokens", "max_completion_tokens"}


class OpenAICompatibleProvider(BaseLLMProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-5.4",
        wire_api: str = "auto",
        reasoning_effort: str = "",
        store: bool = False,
        max_tokens: int = 32768,
        allow_insecure_http: bool = False,
        chat_token_param: str = "max_tokens",
        chat_reasoning_effort: bool = False,
    ):
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.default_model = model
        self.wire_api = self._normalize_wire_api(wire_api)
        self.reasoning_effort = self._normalize_reasoning_effort(reasoning_effort)
        self.store = store
        self.default_max_tokens = max_tokens
        self.chat_token_param = self._normalize_chat_token_param(chat_token_param)
        self.chat_reasoning_effort = chat_reasoning_effort
        self._validate_api_key()
        self._validate_transport_security(allow_insecure_http)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
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

        if self.wire_api == "responses":
            return await self._generate_responses(prompt, model, temperature, max_tokens)
        if self.wire_api == "chat_completions":
            return await self._generate_chat(prompt, model, temperature, max_tokens)

        try:
            return await self._generate_responses(prompt, model, temperature, max_tokens)
        except LLMAPIError as exc:
            if not self._should_fallback_to_chat(exc):
                raise
            logger.warning(
                f"OpenAI Responses API unavailable ({exc.status_code}); "
                "falling back to Chat Completions"
            )
            return await self._generate_chat(prompt, model, temperature, max_tokens)

    async def close(self) -> None:
        await self._client.aclose()

    async def _generate_responses(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload: dict = {
            "model": model,
            "input": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if self.store:
            payload["store"] = True
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}

        logger.debug(f"OpenAI Responses request: model={model}, prompt_len={len(prompt)}")
        data = await post_json_with_retry(
            self._client,
            self._endpoint("responses"),
            payload,
            provider="OpenAI Responses",
        )
        content = self._extract_responses_text(data)
        logger.debug(f"OpenAI Responses response: {len(content)} chars")
        return content

    async def _generate_chat(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": False,
        }
        if self.store:
            payload["store"] = True
        payload[self.chat_token_param] = max_tokens
        if self.reasoning_effort and self.chat_reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        logger.debug(f"OpenAI Chat Completions request: model={model}, prompt_len={len(prompt)}")
        return await chat_completion_with_retry(
            self._client,
            self._endpoint("chat/completions"),
            payload,
            provider="OpenAI Chat Completions",
        )

    def _endpoint(self, suffix: str) -> str:
        base = self.base_url
        if suffix == "responses":
            if base.endswith("/responses"):
                return base
            if base.endswith("/chat/completions"):
                return base.removesuffix("/chat/completions") + "/responses"
        elif suffix == "chat/completions":
            if base.endswith("/chat/completions"):
                return base
            if base.endswith("/responses"):
                return base.removesuffix("/responses") + "/chat/completions"

        if base.endswith("/v1"):
            return f"{base}/{suffix}"
        return f"{base}/v1/{suffix}"

    @staticmethod
    def _extract_responses_text(data: dict) -> str:
        output_text = data.get("output_text")
        if isinstance(output_text, str):
            return output_text

        parts: list[str] = []
        for item in data.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                text = content.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(text, dict) and isinstance(text.get("value"), str):
                    parts.append(text["value"])
        if parts:
            return "\n".join(parts)
        raise KeyError("OpenAI Responses payload did not include output_text or text output")

    @staticmethod
    def _normalize_wire_api(wire_api: str) -> str:
        normalized = wire_api.strip().lower().replace("-", "_")
        aliases = {
            "auto": "auto",
            "responses": "responses",
            "response": "responses",
            "chat": "chat_completions",
            "chat_completion": "chat_completions",
            "chat_completions": "chat_completions",
        }
        if normalized not in aliases:
            raise ValueError(
                "LLM_WIRE_API must be one of: auto, responses, chat_completions"
            )
        return aliases[normalized]

    @staticmethod
    def _normalize_reasoning_effort(reasoning_effort: str) -> str:
        normalized = reasoning_effort.strip().lower()
        if normalized not in _REASONING_EFFORTS:
            raise ValueError(
                "LLM_REASONING_EFFORT must be blank or one of: "
                "none, minimal, low, medium, high, xhigh"
            )
        return normalized

    @staticmethod
    def _normalize_chat_token_param(chat_token_param: str) -> str:
        normalized = chat_token_param.strip().lower()
        if normalized not in _CHAT_TOKEN_PARAMS:
            raise ValueError(
                "LLM_CHAT_TOKEN_PARAM must be one of: max_tokens, max_completion_tokens"
            )
        return normalized

    @staticmethod
    def _should_fallback_to_chat(exc: LLMAPIError) -> bool:
        if exc.status_code in _RESPONSES_FALLBACK_STATUS:
            return True
        if exc.status_code not in {400, 422}:
            return False
        body = exc.body.lower()
        return any(phrase in body for phrase in _RESPONSES_FALLBACK_PHRASES)

    def _validate_api_key(self) -> None:
        if self.api_key or not is_local_url(self.base_url):
            validate_api_key(
                self.api_key,
                env_key="LLM_API_KEY",
                provider="openai_compatible",
            )

    def _validate_transport_security(self, allow_insecure_http: bool) -> None:
        validate_transport_security(
            self.base_url,
            env_key="LLM_BASE_URL",
            allow_insecure_http=allow_insecure_http,
        )

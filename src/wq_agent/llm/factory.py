from __future__ import annotations

from ..config import Settings
from .anthropic import AnthropicProvider
from .base import BaseLLMProvider
from .openai_compatible import OpenAICompatibleProvider


PROTOCOL_PROVIDERS = ("openai_compatible", "anthropic")
PROTOCOL_PROVIDER_OPTIONS = PROTOCOL_PROVIDERS
OPENAI_COMPATIBLE_DEFAULT_MODEL = "gpt-5.4"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_DEFAULT_MODEL = "claude-3-5-sonnet-latest"


class LLMFactory:
    _providers: dict[str, type[BaseLLMProvider]] = {
        "openai_compatible": OpenAICompatibleProvider,
        "anthropic": AnthropicProvider,
    }

    @classmethod
    def register(cls, name: str, provider_cls: type[BaseLLMProvider]) -> None:
        cls._providers[name] = provider_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseLLMProvider:
        provider_name = name.strip().lower()
        provider_cls = cls._providers.get(provider_name)
        if not provider_cls:
            raise ValueError(
                f"Unknown LLM provider: {name}. Available: {sorted(cls._providers.keys())}"
            )
        return provider_cls(**kwargs)

    @classmethod
    def from_settings(cls, settings) -> BaseLLMProvider:
        requested_provider = _first_text(
            settings.LLM_PROVIDER,
            _settings_default("LLM_PROVIDER"),
        ).lower()
        max_tokens = _int_setting(settings, "LLM_MAX_TOKENS")
        if requested_provider == "anthropic":
            return AnthropicProvider(
                api_key=_first_text(settings.LLM_API_KEY),
                base_url=_anthropic_base_url(settings),
                model=_first_text(settings.LLM_MODEL, ANTHROPIC_DEFAULT_MODEL),
                max_tokens=max_tokens,
                allow_insecure_http=settings.LLM_ALLOW_INSECURE_HTTP,
                anthropic_version=_first_text(
                    settings.ANTHROPIC_VERSION,
                    _settings_default("ANTHROPIC_VERSION"),
                ),
            )
        if requested_provider == "openai_compatible":
            return OpenAICompatibleProvider(
                api_key=_first_text(settings.LLM_API_KEY),
                base_url=_first_text(settings.LLM_BASE_URL, _settings_default("LLM_BASE_URL")),
                model=_first_text(settings.LLM_MODEL, OPENAI_COMPATIBLE_DEFAULT_MODEL),
                wire_api=_first_text(settings.LLM_WIRE_API, _settings_default("LLM_WIRE_API")),
                reasoning_effort=_first_text(settings.LLM_REASONING_EFFORT),
                store=settings.LLM_STORE,
                max_tokens=max_tokens,
                allow_insecure_http=settings.LLM_ALLOW_INSECURE_HTTP,
                chat_token_param=_first_text(
                    settings.LLM_CHAT_TOKEN_PARAM,
                    _settings_default("LLM_CHAT_TOKEN_PARAM"),
                    "max_tokens",
                ),
                chat_reasoning_effort=settings.LLM_CHAT_REASONING_EFFORT,
            )
        raise ValueError(f"Unknown LLM provider in settings: {requested_provider}")


def _settings_default(key: str) -> str:
    field = Settings.model_fields[key]
    value = field.default
    if isinstance(value, bool):
        return "true" if value else "false"
    return "" if value is None else str(value)


def _first_text(*values) -> str:
    for value in values:
        text = "" if value is None else str(value).strip()
        if text:
            return text
    return ""


def _explicit_text(settings, key: str) -> str:
    if key not in getattr(settings, "model_fields_set", set()):
        return ""
    return _first_text(getattr(settings, key, ""))


def _int_setting(settings, key: str) -> int:
    value = _first_text(getattr(settings, key, ""), _settings_default(key))
    return int(value)


def _anthropic_base_url(settings) -> str:
    base_url = _explicit_text(settings, "LLM_BASE_URL")
    if not base_url or _normalize_base_url(base_url) == _normalize_base_url(
        _settings_default("LLM_BASE_URL")
    ):
        return ANTHROPIC_BASE_URL
    return base_url


def _normalize_base_url(value: str) -> str:
    return value.strip().rstrip("/").lower()

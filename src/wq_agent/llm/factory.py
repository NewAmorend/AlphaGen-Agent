from __future__ import annotations

from ..config import Settings
from .base import BaseLLMProvider
from .kimi import KimiProvider
from .deepseek import DeepSeekProvider
from .openai import OpenAIProvider


GLOBAL_MODEL_OPTIONS: dict[str, tuple[str, ...]] = {
    "openai": ("gpt-5.5", "gpt-5.4", "gpt-5.4-mini", "gpt-5.3-codex"),
    "kimi": ("kimi-k2.6",),
    "deepseek": ("deepseek-chat", "deepseek-reasoner"),
}


class LLMFactory:
    _providers: dict[str, type[BaseLLMProvider]] = {
        "kimi": KimiProvider,
        "deepseek": DeepSeekProvider,
        "openai": OpenAIProvider,
    }

    @classmethod
    def register(cls, name: str, provider_cls: type[BaseLLMProvider]) -> None:
        cls._providers[name] = provider_cls

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseLLMProvider:
        provider_cls = cls._providers.get(name)
        if not provider_cls:
            raise ValueError(f"Unknown LLM provider: {name}. Available: {list(cls._providers.keys())}")
        return provider_cls(**kwargs)

    @classmethod
    def from_settings(cls, settings) -> BaseLLMProvider:
        provider = _first_text(settings.LLM_PROVIDER, _settings_default("LLM_PROVIDER")).lower()
        model_override = _provider_model_override(provider, _first_text(getattr(settings, "LLM_MODEL", "")))
        max_tokens = _int_setting(settings, "LLM_MAX_TOKENS")
        if provider == "kimi":
            return KimiProvider(
                api_key=settings.KIMI_API_KEY,
                base_url=_first_text(settings.KIMI_BASE_URL, _settings_default("KIMI_BASE_URL")),
                model=_first_text(model_override, settings.KIMI_MODEL, _settings_default("KIMI_MODEL")),
                max_tokens=max_tokens,
            )
        elif provider == "deepseek":
            return DeepSeekProvider(
                api_key=settings.DEEPSEEK_API_KEY,
                base_url=_first_text(
                    settings.DEEPSEEK_BASE_URL, _settings_default("DEEPSEEK_BASE_URL")
                ),
                model=_first_text(
                    model_override, settings.DEEPSEEK_MODEL, _settings_default("DEEPSEEK_MODEL")
                ),
                max_tokens=max_tokens,
            )
        elif provider == "openai":
            return OpenAIProvider(
                api_key=settings.OPENAI_API_KEY,
                base_url=_first_text(settings.OPENAI_BASE_URL, _settings_default("OPENAI_BASE_URL")),
                model=_first_text(
                    model_override, settings.OPENAI_MODEL, _settings_default("OPENAI_MODEL")
                ),
                wire_api=_first_text(settings.OPENAI_WIRE_API, _settings_default("OPENAI_WIRE_API")),
                reasoning_effort=settings.OPENAI_REASONING_EFFORT,
                store=settings.OPENAI_STORE,
                max_tokens=max_tokens,
                allow_insecure_http=settings.OPENAI_ALLOW_INSECURE_HTTP,
                chat_token_param=_first_text(
                    settings.OPENAI_CHAT_TOKEN_PARAM, _settings_default("OPENAI_CHAT_TOKEN_PARAM")
                ),
                chat_reasoning_effort=settings.OPENAI_CHAT_REASONING_EFFORT,
            )
        raise ValueError(f"Unknown LLM provider in settings: {provider}")


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


def _int_setting(settings, key: str) -> int:
    value = _first_text(getattr(settings, key, ""), _settings_default(key))
    return int(value)


def _provider_model_override(provider: str, model: str) -> str:
    if not model:
        return ""
    if provider == "openai":
        return model
    allowed = GLOBAL_MODEL_OPTIONS.get(provider)
    if allowed and model not in allowed:
        raise ValueError(
            f"LLM_MODEL={model!r} is not valid for LLM_PROVIDER={provider!r}. "
            f"Use one of: {', '.join(allowed)}; or leave LLM_MODEL blank and set "
            f"{provider.upper()}_MODEL instead."
        )
    return model

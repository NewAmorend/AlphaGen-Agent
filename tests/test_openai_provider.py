from __future__ import annotations

import pytest

from wq_agent.config import Settings
from wq_agent.llm import AnthropicProvider, LLMFactory, OpenAICompatibleProvider
from wq_agent.llm.factory import PROTOCOL_PROVIDER_OPTIONS


class _Resp:
    def __init__(self, status: int, data: dict | None = None, text: str = "error-body"):
        self.status_code = status
        self._data = data or {}
        self.text = text

    def json(self):
        return self._data


class _Client:
    def __init__(self, responses: list[_Resp]):
        self.responses = responses
        self.calls: list[dict] = []

    async def post(self, url, json=None):
        self.calls.append({"url": url, "json": json})
        idx = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[idx]

    async def aclose(self):
        pass


def _provider(
    *,
    wire_api: str = "responses",
    base_url: str = "http://127.0.0.1:8080",
    api_key: str = "test-key",
):
    return OpenAICompatibleProvider(
        api_key=api_key,
        base_url=base_url,
        model="gpt-test",
        wire_api=wire_api,
        reasoning_effort="xhigh",
        store=False,
        max_tokens=123,
    )


def _settings(**values) -> Settings:
    return Settings(_env_file=None, **values)


def test_factory_creates_openai_compatible_provider_from_protocol_settings():
    provider = LLMFactory.from_settings(
        _settings(
            LLM_PROVIDER="openai_compatible",
            LLM_API_KEY="test-key",
            LLM_BASE_URL="http://127.0.0.1:8080",
            LLM_MODEL="my-private-model",
            LLM_WIRE_API="responses",
        )
    )

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.default_model == "my-private-model"
    assert provider.wire_api == "responses"


def test_factory_exposes_only_protocol_provider_options():
    assert PROTOCOL_PROVIDER_OPTIONS == ("openai_compatible", "anthropic")
    assert "openai" not in PROTOCOL_PROVIDER_OPTIONS
    assert "kimi" not in PROTOCOL_PROVIDER_OPTIONS
    assert "deepseek" not in PROTOCOL_PROVIDER_OPTIONS


@pytest.mark.parametrize("provider", ["openai", "kimi", "deepseek"])
def test_factory_rejects_brand_provider_names(provider: str):
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        LLMFactory.from_settings(_settings(LLM_PROVIDER=provider))

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        LLMFactory.create(provider)


def test_factory_passes_custom_global_model_without_allowlist_validation():
    provider = LLMFactory.from_settings(
        _settings(
            LLM_PROVIDER="openai_compatible",
            LLM_MODEL="private-proxy-model",
            LLM_API_KEY="test-key",
        )
    )

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.default_model == "private-proxy-model"


def test_openai_compatible_allows_localhost_without_api_key_and_omits_auth_header():
    provider = OpenAICompatibleProvider(api_key="", base_url="http://127.0.0.1:8080")

    assert "Authorization" not in provider._client.headers


def test_openai_compatible_rejects_remote_missing_or_placeholder_api_key():
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        OpenAICompatibleProvider(api_key="", base_url="https://api.openai.com/v1")

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        OpenAICompatibleProvider(api_key="your_openai_or_proxy_key")

    with pytest.raises(ValueError, match="LLM_API_KEY"):
        OpenAICompatibleProvider(api_key="change_me")


def test_openai_compatible_rejects_remote_insecure_http_but_allows_override():
    with pytest.raises(ValueError, match="insecure HTTP"):
        OpenAICompatibleProvider(api_key="test-key", base_url="http://example.com:8080")

    provider = OpenAICompatibleProvider(
        api_key="test-key",
        base_url="http://example.com:8080",
        allow_insecure_http=True,
    )
    assert provider.base_url == "http://example.com:8080"


@pytest.mark.asyncio
async def test_responses_payload_omits_store_by_default_and_parses_output_text():
    provider = _provider(wire_api="responses")
    client = _Client([_Resp(200, {"output_text": "alpha output"})])
    provider._client = client

    out = await provider.generate("make alpha", temperature=0.7, max_tokens=456)

    assert out == "alpha output"
    assert client.calls[0]["url"] == "http://127.0.0.1:8080/v1/responses"
    payload = client.calls[0]["json"]
    assert payload["model"] == "gpt-test"
    assert payload["input"] == [{"role": "user", "content": "make alpha"}]
    assert payload["temperature"] == 0.7
    assert payload["max_output_tokens"] == 456
    assert "store" not in payload
    assert payload["reasoning"] == {"effort": "xhigh"}


@pytest.mark.asyncio
async def test_responses_fallback_text_output_parsing():
    provider = _provider(wire_api="responses")
    client = _Client(
        [
            _Resp(
                200,
                {
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "alpha one"},
                                {"type": "output_text", "text": "alpha two"},
                            ],
                        }
                    ]
                },
            )
        ]
    )
    provider._client = client

    out = await provider.generate("make alpha")

    assert out == "alpha one\nalpha two"


@pytest.mark.asyncio
async def test_chat_completions_minimal_payload_omits_vendor_specific_fields():
    provider = _provider(wire_api="chat_completions", base_url="http://127.0.0.1:8080/v1")
    client = _Client([_Resp(200, {"choices": [{"message": {"content": "chat alpha"}}]})])
    provider._client = client

    out = await provider.generate("make alpha", temperature=0.2, max_tokens=99)

    assert out == "chat alpha"
    assert client.calls[0]["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    payload = client.calls[0]["json"]
    assert payload["model"] == "gpt-test"
    assert payload["messages"] == [{"role": "user", "content": "make alpha"}]
    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 99
    assert payload["stream"] is False
    assert "store" not in payload
    assert "reasoning_effort" not in payload


@pytest.mark.asyncio
async def test_chat_completions_optional_store_token_and_reasoning_parameters():
    provider = OpenAICompatibleProvider(
        api_key="test-key",
        base_url="https://proxy.example",
        model="gpt-test",
        wire_api="chat_completions",
        reasoning_effort="xhigh",
        store=True,
        chat_token_param="max_completion_tokens",
        chat_reasoning_effort=True,
    )
    provider._client = _Client([_Resp(200, {"choices": [{"message": {"content": "chat alpha"}}]})])

    await provider.generate("make alpha", max_tokens=99)

    payload = provider._client.calls[0]["json"]
    assert payload["store"] is True
    assert payload["max_completion_tokens"] == 99
    assert "max_tokens" not in payload
    assert payload["reasoning_effort"] == "xhigh"


def test_openai_compatible_endpoint_accepts_root_and_full_endpoint_urls():
    root = OpenAICompatibleProvider(api_key="test-key", base_url="https://proxy.example")
    assert root._endpoint("chat/completions") == "https://proxy.example/v1/chat/completions"
    assert root._endpoint("responses") == "https://proxy.example/v1/responses"

    chat = OpenAICompatibleProvider(
        api_key="test-key",
        base_url="https://proxy.example/custom/chat/completions",
    )
    assert chat._endpoint("chat/completions") == "https://proxy.example/custom/chat/completions"
    assert chat._endpoint("responses") == "https://proxy.example/custom/responses"

    responses = OpenAICompatibleProvider(
        api_key="test-key",
        base_url="https://proxy.example/custom/responses",
    )
    assert responses._endpoint("responses") == "https://proxy.example/custom/responses"
    assert responses._endpoint("chat/completions") == (
        "https://proxy.example/custom/chat/completions"
    )


@pytest.mark.asyncio
async def test_auto_mode_falls_back_to_chat_completions_when_responses_is_unavailable():
    provider = _provider(wire_api="auto")
    client = _Client(
        [
            _Resp(404, text="missing responses endpoint"),
            _Resp(200, {"choices": [{"message": {"content": "chat fallback"}}]}),
        ]
    )
    provider._client = client

    out = await provider.generate("make alpha")

    assert out == "chat fallback"
    assert client.calls[0]["url"] == "http://127.0.0.1:8080/v1/responses"
    assert client.calls[1]["url"] == "http://127.0.0.1:8080/v1/chat/completions"


@pytest.mark.asyncio
async def test_auto_mode_does_not_fall_back_for_responses_configuration_errors():
    provider = _provider(wire_api="auto")
    client = _Client([_Resp(400, text="invalid reasoning effort")])
    provider._client = client

    with pytest.raises(Exception, match="invalid reasoning effort"):
        await provider.generate("make alpha")

    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_anthropic_messages_payload_headers_and_text_parsing():
    provider = AnthropicProvider(
        api_key="test-key",
        base_url="https://api.anthropic.com",
        model="claude-test",
        max_tokens=123,
    )
    headers = provider._client.headers
    client = _Client(
        [
            _Resp(
                200,
                {"content": [{"type": "text", "text": "alpha one"}, {"type": "text", "text": "alpha two"}]},
            )
        ]
    )
    provider._client = client

    out = await provider.generate("make alpha", temperature=0.4, max_tokens=88)

    assert out == "alpha one\nalpha two"
    assert headers["x-api-key"] == "test-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert client.calls[0]["url"] == "https://api.anthropic.com/v1/messages"
    payload = client.calls[0]["json"]
    assert payload == {
        "model": "claude-test",
        "messages": [{"role": "user", "content": "make alpha"}],
        "temperature": 0.4,
        "max_tokens": 88,
    }


def test_factory_creates_anthropic_provider():
    provider = LLMFactory.from_settings(
        _settings(
            LLM_PROVIDER="anthropic",
            LLM_API_KEY="test-key",
            LLM_MODEL="claude-custom",
        )
    )

    assert isinstance(provider, AnthropicProvider)
    assert provider.base_url == "https://api.anthropic.com"
    assert provider.default_model == "claude-custom"


def test_factory_uses_anthropic_default_when_env_example_base_url_is_unchanged():
    provider = LLMFactory.from_settings(
        _settings(
            LLM_PROVIDER="anthropic",
            LLM_API_KEY="test-key",
            LLM_BASE_URL="https://api.openai.com/v1",
        )
    )

    assert isinstance(provider, AnthropicProvider)
    assert provider.base_url == "https://api.anthropic.com"


def test_factory_treats_openai_default_with_trailing_slash_as_anthropic_default():
    provider = LLMFactory.from_settings(
        _settings(
            LLM_PROVIDER="anthropic",
            LLM_API_KEY="test-key",
            LLM_BASE_URL="https://api.openai.com/v1/",
        )
    )

    assert isinstance(provider, AnthropicProvider)
    assert provider.base_url == "https://api.anthropic.com"


def test_factory_respects_explicit_anthropic_base_url():
    provider = LLMFactory.from_settings(
        _settings(
            LLM_PROVIDER="anthropic",
            LLM_API_KEY="test-key",
            LLM_BASE_URL="https://anthropic-proxy.example",
        )
    )

    assert isinstance(provider, AnthropicProvider)
    assert provider.base_url == "https://anthropic-proxy.example"

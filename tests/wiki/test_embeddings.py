from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from wq_agent.wiki.embeddings import (
    LocalEmbeddingProvider,
    NoOpEmbeddingProvider,
    make_embedding_provider,
)


@pytest.mark.asyncio
async def test_local_embedding_returns_lists_and_detects_dim():
    fake_model = MagicMock()
    fake_model.embed = MagicMock(return_value=iter([
        np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32),
    ]))
    fake_module = SimpleNamespace(TextEmbedding=MagicMock(return_value=fake_model))
    with patch.dict("sys.modules", {"fastembed": fake_module}):
        provider = LocalEmbeddingProvider(model_name="fake/model")
        out = await provider.embed(["hello", "world"])
        assert out == [[pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3), pytest.approx(0.4)],
                       [pytest.approx(0.5), pytest.approx(0.6), pytest.approx(0.7), pytest.approx(0.8)]]
        assert provider.dim == 4
        fake_module.TextEmbedding.assert_called_once_with(model_name="fake/model")


@pytest.mark.asyncio
async def test_local_embedding_lazy_loads_model_once():
    fake_model = MagicMock()
    fake_model.embed = MagicMock(side_effect=lambda texts: iter([
        np.zeros(3, dtype=np.float32) for _ in texts
    ]))
    fake_module = SimpleNamespace(TextEmbedding=MagicMock(return_value=fake_model))
    with patch.dict("sys.modules", {"fastembed": fake_module}):
        provider = LocalEmbeddingProvider(model_name="fake/model")
        await provider.embed(["a"])
        await provider.embed(["b"])
        await provider.embed(["c"])
        # TextEmbedding 只在首次构造一次
        assert fake_module.TextEmbedding.call_count == 1


@pytest.mark.asyncio
async def test_local_embedding_raises_useful_error_without_fastembed():
    import sys
    saved = sys.modules.pop("fastembed", None)
    try:
        with patch.dict("sys.modules", {"fastembed": None}):
            provider = LocalEmbeddingProvider(model_name="fake/model")
            with pytest.raises(RuntimeError, match="pip install fastembed"):
                await provider.embed(["text"])
    finally:
        if saved is not None:
            sys.modules["fastembed"] = saved


@pytest.mark.asyncio
async def test_local_embedding_empty_input_returns_empty():
    provider = LocalEmbeddingProvider(model_name="fake/model")
    assert await provider.embed([]) == []


def test_factory_returns_local_for_provider_local():
    settings = SimpleNamespace(
        EMBEDDING_PROVIDER="local",
        EMBEDDING_DIM=0,
        LOCAL_EMBEDDING_MODEL="BAAI/bge-small-zh-v1.5",
        EMBEDDING_API_KEY="",
        KIMI_API_KEY="",
        EMBEDDING_MODEL="",
        EMBEDDING_BASE_URL="",
    )
    p = make_embedding_provider(settings)
    assert isinstance(p, LocalEmbeddingProvider)
    assert p.model_name == "BAAI/bge-small-zh-v1.5"


def test_factory_returns_noop_for_provider_none():
    settings = SimpleNamespace(
        EMBEDDING_PROVIDER="none",
        EMBEDDING_DIM=512,
        LOCAL_EMBEDDING_MODEL="",
        EMBEDDING_API_KEY="",
        KIMI_API_KEY="",
        EMBEDDING_MODEL="",
        EMBEDDING_BASE_URL="",
    )
    p = make_embedding_provider(settings)
    assert isinstance(p, NoOpEmbeddingProvider)

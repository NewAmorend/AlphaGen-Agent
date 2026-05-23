from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Iterable

import httpx
from loguru import logger


class BaseEmbeddingProvider(ABC):
    dim: int

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...


class NoOpEmbeddingProvider(BaseEmbeddingProvider):
    """检索时跳过向量通道。"""

    def __init__(self, dim: int = 0):
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]

    async def close(self) -> None:
        return None


class _HTTPEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        dim: int,
    ):
        if not api_key:
            raise ValueError(f"Embedding provider {model} requires api_key")
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.dim = dim
        self._client = httpx.AsyncClient(
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            timeout=httpx.Timeout(60.0, connect=15.0),
        )

    async def close(self) -> None:
        await self._client.aclose()


class VolcengineEmbeddingProvider(_HTTPEmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.model, "input": texts}
        resp = await self._client.post(self.base_url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Volcengine embedding error ({resp.status_code}): {resp.text[:300]}"
            )
        data = resp.json()
        items = data.get("data", [])
        vectors = [item["embedding"] for item in sorted(items, key=lambda x: x.get("index", 0))]
        if vectors and len(vectors[0]) != self.dim:
            logger.warning(
                f"Embedding dim mismatch: provider returned {len(vectors[0])}, configured {self.dim}"
            )
        return vectors


class ZhipuEmbeddingProvider(_HTTPEmbeddingProvider):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for text in texts:
            payload = {"model": self.model, "input": text}
            resp = await self._client.post(self.base_url, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Zhipu embedding error ({resp.status_code}): {resp.text[:300]}"
                )
            data = resp.json()
            out.append(data["data"][0]["embedding"])
        return out


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """本地 ONNX 嵌入（fastembed）。首次调用会从 HuggingFace 下载模型，之后离线运行。"""

    def __init__(self, model_name: str, dim: int | None = None):
        self.model_name = model_name
        self.dim = dim or 0
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "Local embedding requires `pip install fastembed`. "
                "Then set EMBEDDING_PROVIDER=local."
            ) from exc
        logger.info(f"Loading local embedding model: {self.model_name}")
        self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        loop = asyncio.get_event_loop()
        # fastembed.embed() 是同步 CPU 工作，丢线程池避免阻塞事件循环
        vecs = await loop.run_in_executor(None, lambda: [v for v in model.embed(texts)])
        out = [v.tolist() for v in vecs]
        if out and self.dim == 0:
            self.dim = len(out[0])
            logger.info(f"Detected embedding dim: {self.dim}")
        return out

    async def close(self) -> None:
        self._model = None


def make_embedding_provider(settings) -> BaseEmbeddingProvider:
    name = (settings.EMBEDDING_PROVIDER or "none").lower()
    if name in ("", "none", "off", "disabled"):
        return NoOpEmbeddingProvider(dim=settings.EMBEDDING_DIM)
    if name == "local":
        return LocalEmbeddingProvider(
            model_name=settings.LOCAL_EMBEDDING_MODEL,
            dim=settings.EMBEDDING_DIM if settings.EMBEDDING_DIM > 0 else None,
        )
    api_key = settings.EMBEDDING_API_KEY or settings.KIMI_API_KEY
    if name == "volcengine":
        return VolcengineEmbeddingProvider(
            api_key=api_key,
            base_url=settings.EMBEDDING_BASE_URL,
            model=settings.EMBEDDING_MODEL,
            dim=settings.EMBEDDING_DIM,
        )
    if name == "zhipu" or name == "zhipuai":
        return ZhipuEmbeddingProvider(
            api_key=api_key,
            base_url=settings.EMBEDDING_BASE_URL,
            model=settings.EMBEDDING_MODEL,
            dim=settings.EMBEDDING_DIM,
        )
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.EMBEDDING_PROVIDER}")


def chunk(iterable: list, size: int) -> Iterable[list]:
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]

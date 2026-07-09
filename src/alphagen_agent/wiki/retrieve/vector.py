from __future__ import annotations

import math
from dataclasses import dataclass

from loguru import logger

from ..embeddings import BaseEmbeddingProvider, NoOpEmbeddingProvider
from ..schema import Page


@dataclass
class VectorHit:
    page: Page
    score: float


class VectorChannel:
    """sqlite-vec backed semantic search; falls back to in-memory cosine."""

    def __init__(
        self,
        pages: list[Page],
        embeddings: dict[str, list[float]],
        provider: BaseEmbeddingProvider,
    ):
        self.pages_by_path = {str(p.path): p for p in pages}
        self.embeddings = {k: v for k, v in embeddings.items() if v}
        self.provider = provider
        self.disabled = isinstance(provider, NoOpEmbeddingProvider) or not self.embeddings

    async def search(self, query: str, top_k: int = 10) -> list[VectorHit]:
        if self.disabled:
            return []
        try:
            qvecs = await self.provider.embed([query])
        except Exception as exc:
            logger.warning(f"Vector channel disabled this turn: embedding failed ({exc})")
            return []
        if not qvecs or not qvecs[0]:
            return []
        q = qvecs[0]
        qn = math.sqrt(sum(x * x for x in q)) or 1.0
        hits: list[VectorHit] = []
        for path, vec in self.embeddings.items():
            if len(vec) != len(q):
                continue
            dot = sum(a * b for a, b in zip(q, vec))
            vn = math.sqrt(sum(x * x for x in vec)) or 1.0
            score = dot / (qn * vn)
            page = self.pages_by_path.get(path)
            if not page:
                continue
            hits.append(VectorHit(page=page, score=score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

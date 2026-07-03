from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from ..db import Database
from .embeddings import BaseEmbeddingProvider, NoOpEmbeddingProvider, chunk
from .schema import Page
from .store import WikiStore
from .tokenize import Tokenizer
from .retrieve.graph import GraphChannel
from .retrieve.grep import GrepChannel
from .retrieve.hybrid import HybridRetriever
from .retrieve.vector import VectorChannel


@dataclass
class IndexStats:
    pages: int
    embeddings: int
    embedded_now: int
    nodes: int
    edges: int
    communities: int
    broken_links: int
    errors: int


class WikiIndex:
    """构建/刷新索引 + 暴露 HybridRetriever。"""

    def __init__(
        self,
        store: WikiStore,
        db: Database,
        embedder: BaseEmbeddingProvider,
        grep_weight: int = 7,
        vector_weight: int = 3,
    ):
        self.store = store
        self.db = db
        self.embedder = embedder
        self.grep_weight = grep_weight
        self.vector_weight = vector_weight
        self._pages: list[Page] = []
        self._retriever: HybridRetriever | None = None

    @property
    def pages(self) -> list[Page]:
        return self._pages

    @property
    def retriever(self) -> HybridRetriever | None:
        return self._retriever

    async def build(
        self,
        *,
        incremental: bool = False,
        extra_dict_terms: list[str] | None = None,
    ) -> IndexStats:
        pages, errors = self.store.load_pages()
        self._pages = pages

        keep = {str(p.path) for p in pages}
        await self.db.delete_wiki_pages(keep)
        prior_hashes = await self.db.get_wiki_hashes() if incremental else {}

        rows = [
            {
                "path": str(p.path),
                "title": p.title,
                "type_": p.type.value,
                "tags": p.tags,
                "sources": p.sources,
                "content_hash": p.content_hash,
            }
            for p in pages
        ]
        await self.db.bulk_upsert_wiki_pages(rows)

        to_embed: list[Page] = [
            p for p in pages
            if not incremental or prior_hashes.get(str(p.path)) != p.content_hash
        ]

        embedded_now = await self._embed_pages(to_embed)
        embeddings = await self.db.load_wiki_embeddings()

        tokenizer = Tokenizer.from_paths(
            base_dict=self.store.dictionary_path(),
            auto_dict=self.store.auto_dictionary_path(),
            synonyms_yaml=self.store.synonyms_path(),
            extra_terms=extra_dict_terms,
        )
        grep = GrepChannel(pages=pages, tokenizer=tokenizer)
        vector = VectorChannel(pages=pages, embeddings=embeddings, provider=self.embedder)
        graph = GraphChannel(pages=pages)
        graph.save(self.store.graph_index_path())

        self._retriever = HybridRetriever(
            pages=pages,
            grep=grep,
            vector=vector,
            graph=graph,
            grep_weight=self.grep_weight,
            vector_weight=self.vector_weight,
        )

        broken = self.store.find_broken_links(pages)
        if broken:
            logger.warning(f"{len(broken)} pages have broken wikilinks")

        return IndexStats(
            pages=len(pages),
            embeddings=len(embeddings),
            embedded_now=embedded_now,
            nodes=graph.graph.number_of_nodes(),
            edges=graph.graph.number_of_edges(),
            communities=len(set(graph.communities.values())) if graph.communities else 0,
            broken_links=sum(len(b[1]) for b in broken),
            errors=len(errors),
        )

    async def _embed_pages(self, pages: list[Page]) -> int:
        if isinstance(self.embedder, NoOpEmbeddingProvider) or not pages:
            return 0
        texts = [self._embed_text(p) for p in pages]
        count = 0
        for batch_pages, batch_texts in zip(chunk(pages, 16), chunk(texts, 16)):
            try:
                vecs = await self.embedder.embed(batch_texts)
            except Exception as exc:
                logger.warning(f"Embedding batch failed: {exc}")
                continue
            for p, v in zip(batch_pages, vecs):
                if v:
                    await self.db.upsert_wiki_embedding(str(p.path), v)
                    count += 1
        return count

    @staticmethod
    def _embed_text(page: Page) -> str:
        head = f"{page.title}\n类型: {page.type.value}\n标签: {', '.join(page.tags)}\n\n"
        return head + page.body[:2000]

    def write_auto_dictionary(self, terms: list[str]) -> None:
        path = self.store.auto_dictionary_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        uniq = sorted({t.strip() for t in terms if t and t.strip()})
        path.write_text("\n".join(uniq) + "\n", encoding="utf-8")

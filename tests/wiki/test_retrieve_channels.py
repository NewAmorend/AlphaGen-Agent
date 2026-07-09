from __future__ import annotations

import math
from pathlib import Path

import pytest

from alphagen_agent.wiki.embeddings import BaseEmbeddingProvider
from alphagen_agent.wiki.retrieve.graph import GraphChannel
from alphagen_agent.wiki.retrieve.grep import GrepChannel
from alphagen_agent.wiki.retrieve.hybrid import HybridRetriever
from alphagen_agent.wiki.retrieve.vector import VectorChannel
from alphagen_agent.wiki.store import WikiStore
from alphagen_agent.wiki.tokenize import Tokenizer


def _tokenizer(root: Path) -> Tokenizer:
    return Tokenizer.from_paths(
        base_dict=root / "dictionary/base.txt",
        synonyms_yaml=root / "dictionary/synonyms.yaml",
    )


def test_fmm_tokenize_chinese_and_ascii(wiki_root: Path):
    tk = _tokenizer(wiki_root)
    tokens = tk.tokenize("动量 反转 ts_delta")
    assert "动量" in tokens
    assert "反转" in tokens
    assert "ts_delta" in tokens


def test_synonym_expansion_assigns_lower_weight(wiki_root: Path):
    tk = _tokenizer(wiki_root)
    weighted, originals = tk.expand_synonyms(["动量"])
    weights = dict(weighted)
    assert weights["动量"] == 1.0
    assert weights["momentum"] == 0.5
    assert originals == ["动量"]


def test_grep_priority_when_all_originals_matched(wiki_root: Path):
    store = WikiStore(wiki_root)
    pages, _ = store.load_pages()
    tk = _tokenizer(wiki_root)
    grep = GrepChannel(pages=pages, tokenizer=tk)
    hits = grep.search("动量")
    assert hits
    assert hits[0].matched_all_originals
    assert "动量" in hits[0].page.title or "动量" in hits[0].page.body


def test_grep_search_is_fast_on_large_wiki(tmp_path: Path):
    """Regression: 旧版 per-page subprocess ripgrep 在 7700 页上要 13 分钟。
    现在纯内存 str.count，1500 页 + 10 token query 应在 1 秒内完成。"""
    import time
    from datetime import date

    from alphagen_agent.wiki.schema import Page, PageType

    pages: list[Page] = []
    for i in range(1500):
        pages.append(Page(
            path=Path(f"fields/f_{i}.md"),
            title=f"field_{i}",
            type=PageType.FIELD,
            tags=["field", "fundamental"],
            created=date(2026, 5, 22),
            body=f"description of field {i} mentioning momentum reversal volatility liquidity quality",
            wikilinks=[],
        ))
    tk = _tokenizer(tmp_path) if (tmp_path / "dictionary/base.txt").exists() else Tokenizer.from_paths()

    g = GrepChannel(pages=pages, tokenizer=tk)
    start = time.monotonic()
    hits = g.search("momentum reversal volatility liquidity quality field fundamental", top_k=5)
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, f"Grep too slow: {elapsed:.2f}s"
    assert len(hits) == 5


def test_graph_handles_large_boilerplate_wiki_without_blowup(tmp_path: Path):
    """Regression: 7000+ pages 几乎全部共享相同的 boilerplate tag/source 时，旧版 O(n²) 会 OOM。"""
    import time
    from datetime import date

    from alphagen_agent.wiki.schema import Page, PageType

    pages: list[Page] = []
    for i in range(1500):
        pages.append(Page(
            path=Path(f"fields/field_{i}.md"),
            title=f"field_{i}",
            type=PageType.FIELD,
            tags=["field", "fundamental6", "usa", "matrix"],  # 全员共享 → 应被过滤
            sources=["worldquantbrain-api"],                  # 全员共享 → 应被过滤
            created=date(2026, 5, 22),
            body=f"placeholder body for field {i}",
            wikilinks=[],
        ))

    start = time.monotonic()
    g = GraphChannel(pages=pages)
    elapsed = time.monotonic() - start

    assert elapsed < 5.0, f"Graph build too slow: {elapsed:.2f}s"
    assert g.graph.number_of_edges() < 1000, \
        f"Edges should be bounded by noise filter, got {g.graph.number_of_edges()}"


def test_graph_expand_follows_wikilinks(wiki_root: Path):
    store = WikiStore(wiki_root)
    pages, _ = store.load_pages()
    # 加几个页让 graph 通过最小节点阈值
    extra_dir = wiki_root / "concepts"
    for i in range(10):
        path = extra_dir / f"filler_{i}.md"
        path.write_text(
            f"---\ntitle: filler{i}\ntype: concept\ntags: [filler, momentum]\ncreated: 2026-05-22\n---\n\n看 [[动量]]\n",
            encoding="utf-8",
        )
    pages, _ = store.load_pages()
    g = GraphChannel(pages=pages)
    expanded = g.expand(["momentum"], k=2)
    # momentum 不是 slug，验证用真实 slug
    target_slug = next((p.slug for p in pages if p.title == "动量"), None)
    assert target_slug is not None
    expanded = g.expand([target_slug], k=4)
    titles = [h.page.title for h in expanded]
    # ts_delta 与 ts_decay_linear 都被 momentum 页 wikilink，应至少跳到其中之一
    assert any(t in {"ts_delta", "ts_decay_linear"} for t in titles)


class _DeterministicEmbed(BaseEmbeddingProvider):
    """Map each page title to its own one-hot dim, so cosine has a unique max per title."""

    _TITLES = ["动量", "反转", "ts_decay_linear", "ts_delta"]

    def __init__(self):
        self.dim = len(self._TITLES)

    def _vec_for(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        head = text.split("\n", 1)[0]
        for i, title in enumerate(self._TITLES):
            if head == title or head.startswith(title + " "):
                v[i] = 1.0
                return v
        # query path: pick first matching title found in text
        for i, title in enumerate(self._TITLES):
            if title in text:
                v[i] = 1.0
                return v
        return v

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            v = self._vec_for(t)
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_vector_channel_ranks_by_cosine(wiki_root: Path):
    store = WikiStore(wiki_root)
    pages, _ = store.load_pages()
    embedder = _DeterministicEmbed()
    text_for = {str(p.path): f"{p.title}\n标签: {','.join(p.tags)}\n{p.body[:300]}" for p in pages}
    vectors = await embedder.embed(list(text_for.values()))
    embeddings = {path: v for path, v in zip(text_for.keys(), vectors)}
    vc = VectorChannel(pages=pages, embeddings=embeddings, provider=embedder)
    hits = await vc.search("动量")
    assert hits
    assert hits[0].page.title == "动量"


@pytest.mark.asyncio
async def test_hybrid_priority_then_rrf_then_graph(wiki_root: Path):
    store = WikiStore(wiki_root)
    pages, _ = store.load_pages()
    tk = _tokenizer(wiki_root)
    grep = GrepChannel(pages=pages, tokenizer=tk)
    embedder = _DeterministicEmbed()
    vectors = await embedder.embed([f"{p.title}\n{p.body[:300]}" for p in pages])
    embeddings = {str(p.path): v for p, v in zip(pages, vectors)}
    vc = VectorChannel(pages=pages, embeddings=embeddings, provider=embedder)
    g = GraphChannel(pages=pages)
    hyb = HybridRetriever(pages=pages, grep=grep, vector=vc, graph=g)
    hits = await hyb.search("动量", top_k=5)
    titles = [h.page.title for h in hits]
    assert titles[0] == "动量"  # priority grep 置顶
    assert any("priority" in h.sources for h in hits if h.page.title == "动量")

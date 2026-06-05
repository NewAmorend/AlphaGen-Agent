from __future__ import annotations

from pathlib import Path

import pytest

from wq_agent.wiki.schema import PageType, parse_page
from wq_agent.wiki.store import CompositeWikiStore, WikiStore


def test_parse_page_extracts_frontmatter_and_links(wiki_root: Path):
    page = parse_page(wiki_root / "concepts/momentum.md")
    assert page.title == "动量"
    assert page.type is PageType.CONCEPT
    assert "momentum" in page.tags
    assert "ts_delta" in page.wikilinks
    assert "ts_decay_linear" in page.wikilinks
    assert page.summary(40).endswith("…") or len(page.summary(40)) <= 40


def test_parse_page_rejects_missing_frontmatter(tmp_path: Path):
    p = tmp_path / "bad.md"
    p.write_text("no frontmatter here\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_page(p)


def test_store_iterates_real_pages_only(wiki_root: Path):
    store = WikiStore(wiki_root)
    pages, errors = store.load_pages()
    assert errors == []
    assert len(pages) == 4  # 2 concepts + 2 operators


def test_store_finds_broken_links(wiki_root: Path):
    (wiki_root / "concepts/orphan_link.md").write_text(
        "---\ntitle: 孤儿\ntype: concept\ntags: [test]\ncreated: 2026-05-22\n---\n\n看 [[不存在的页]]。\n",
        encoding="utf-8",
    )
    store = WikiStore(wiki_root)
    pages, _ = store.load_pages()
    broken = store.find_broken_links(pages)
    assert any("不存在的页" in misses for _, misses in broken)


def test_store_resolves_path_style_wikilinks(wiki_root: Path):
    (wiki_root / "concepts/path_link.md").write_text(
        "---\ntitle: path link\ntype: concept\ntags: [test]\ncreated: 2026-05-22\n---\n\n看 [[operators/ts_delta]]。\n",
        encoding="utf-8",
    )
    store = WikiStore(wiki_root)
    pages, _ = store.load_pages()
    broken = store.find_broken_links(pages)
    assert not any("operators/ts_delta" in misses for _, misses in broken)


def test_composite_store_reads_public_and_private_pages(wiki_root: Path, tmp_path: Path):
    private_root = tmp_path / "private_wiki"
    (private_root / "entries").mkdir(parents=True)
    (private_root / "entries/alpha-1.md").write_text(
        "---\ntitle: Private Alpha 1\ntype: entry\ntags: [entry, private]\ncreated: 2026-06-05\n---\n\n"
        "私有记录引用 [[operators/ts_delta]] 和 [[entries/alpha-1]]。\n",
        encoding="utf-8",
    )

    store = CompositeWikiStore(wiki_root, [private_root])
    pages, errors = store.load_pages()

    assert errors == []
    assert len(pages) == 5
    assert any(p.title == "Private Alpha 1" for p in pages)
    assert store.dictionary_path() == wiki_root / "dictionary" / "base.txt"
    assert not any("operators/ts_delta" in misses for _, misses in store.find_broken_links(pages))
    assert not any("entries/alpha-1" in misses for _, misses in store.find_broken_links(pages))


def test_composite_store_exists_when_only_private_root_exists(tmp_path: Path):
    public_root = tmp_path / "wiki"
    private_root = tmp_path / "private_wiki"
    private_root.mkdir()

    store = CompositeWikiStore(public_root, [private_root])

    assert store.exists()
    assert list(store.iter_page_paths()) == []

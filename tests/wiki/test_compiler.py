from __future__ import annotations

import json

from alphagen_agent.wiki.compiler import MANAGED_MARKER, WikiCompiler
from alphagen_agent.wiki.schema import parse_page
from alphagen_agent.wiki.store import WikiStore


def test_wiki_compiler_generates_hubs_and_typed_edges(wiki_root):
    compiler = WikiCompiler(WikiStore(wiki_root))

    stats = compiler.compile(max_hubs=5, min_pages=2, pages_per_hub=5)

    assert stats.source_pages == 4
    assert stats.hubs_written >= 1
    assert stats.typed_edges >= 4

    hub_path = wiki_root / "hubs" / "momentum.md"
    assert hub_path.exists()
    hub = parse_page(hub_path)
    assert hub.title == "Hub: momentum"
    assert "hub" in hub.tags
    assert "concepts/momentum" in hub.body
    assert "operators/ts_delta" in hub.body
    assert MANAGED_MARKER in hub.body
    store = WikiStore(wiki_root)
    pages, _ = store.load_pages()
    assert not [
        (page.path, misses)
        for page, misses in store.find_broken_links(pages)
        if "hubs" in page.path.parts
    ]

    payload = json.loads((wiki_root / "typed_edges.json").read_text(encoding="utf-8"))
    edges = payload["edges"]
    assert any(e["relation"] == "has_topic" and e["target"] == "hubs/momentum" for e in edges)
    assert any(
        e["relation"] == "references"
        and e["source"] == "concepts/momentum"
        and e["target"] == "operators/ts_delta"
        for e in edges
    )


def test_wiki_compiler_does_not_overwrite_manual_hub(wiki_root):
    manual = wiki_root / "hubs" / "momentum.md"
    manual.parent.mkdir(parents=True, exist_ok=True)
    manual.write_text(
        "---\ntitle: Manual Momentum\ntype: concept\ntags: [hub, momentum]\n---\n\nmanual",
        encoding="utf-8",
    )

    stats = WikiCompiler(WikiStore(wiki_root)).compile(max_hubs=5, min_pages=2)

    assert stats.hubs_skipped >= 1
    assert "manual" in manual.read_text(encoding="utf-8")

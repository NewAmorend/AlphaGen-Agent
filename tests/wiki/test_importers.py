from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from alphagen_agent.models import WQOperator
from alphagen_agent.wiki.importers.papers import (
    PaperImporter,
    PaperRecord,
    fetch_arxiv,
    parse_url,
    slugify,
)
from alphagen_agent.wiki.importers.wq import WQDocImporter
from alphagen_agent.wiki.importers.wq_tutorials import WQTutorialImporter, html_to_markdown
from alphagen_agent.wiki.schema import parse_page


_ARXIV_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678v1</id>
    <title>A Test Paper About Momentum</title>
    <summary>We document a striking pattern.</summary>
    <published>2024-01-15T00:00:00Z</published>
    <author><name>Alice Researcher</name></author>
    <author><name>Bob Quant</name></author>
  </entry>
</feed>
"""


def test_slugify_basic():
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("中文 mixed text") == "mixed-text" or "中文" in slugify("中文 mixed text")


def test_parse_url_recognizes_sources():
    assert parse_url("https://arxiv.org/abs/2401.12345") == ("arxiv", "2401.12345")
    assert parse_url("https://arxiv.org/pdf/2401.12345v2.pdf") == ("arxiv", "2401.12345")
    assert parse_url("https://papers.ssrn.com/sol3/papers.cfm?abstract_id=987654")[0] == "ssrn"
    assert parse_url("https://example.com/random") is None


@pytest.mark.asyncio
async def test_fetch_arxiv_parses_atom(monkeypatch):
    class _Resp:
        status_code = 200
        text = _ARXIV_RESPONSE

    async def _get(url):
        return _Resp()

    client = httpx.AsyncClient()
    client.get = _get  # type: ignore[method-assign]
    record = await fetch_arxiv("1234.5678", client=client)
    await client.aclose()
    assert record.title == "A Test Paper About Momentum"
    assert record.authors == ["Alice Researcher", "Bob Quant"]
    assert record.year == 2024
    assert record.source == "arxiv"
    assert "striking pattern" in record.abstract


def test_paper_record_to_page_roundtrip(tmp_path: Path):
    rec = PaperRecord(
        title="Test Paper",
        authors=["Jane Doe"],
        year=2024,
        abstract="A short abstract.",
        url="https://arxiv.org/abs/2401.99999",
        source="arxiv",
        identifier="2401.99999",
    )
    importer = PaperImporter(tmp_path)
    path = importer._write(rec, tags=["momentum"])
    assert path.exists()
    page = parse_page(path)
    assert page.title == "Test Paper"
    assert "paper" in page.tags
    assert "momentum" in page.tags


def test_paper_import_skips_manually_edited(tmp_path: Path):
    rec = PaperRecord(
        title="X", authors=[], year=None, abstract="", url="",
        source="manual", identifier="",
    )
    importer = PaperImporter(tmp_path)
    path = importer._write(rec, tags=None)
    # 模拟用户手动改了页（移除 managed marker）
    path.write_text("---\ntitle: Edited\ntype: concept\ntags: [test]\ncreated: 2026-05-22\n---\n\nuser content\n", encoding="utf-8")
    importer._write(rec, tags=None)  # 不应覆盖
    assert "user content" in path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_wq_importer_operators_renders_one_page_per_operator(tmp_path: Path):
    client = AsyncMock()
    client.get_operators = AsyncMock(return_value=[
        WQOperator(name="ts_mean", category="Time Series", type="SCALAR",
                   description="Time series mean."),
        WQOperator(name="rank", category="Cross Sectional", type="SCALAR",
                   description="Cross-sectional rank."),
    ])
    importer = WQDocImporter(wiki_root=tmp_path, client=client)
    n = await importer.import_operators()
    assert n == 2
    page_paths = sorted((tmp_path / "operators").glob("*.md"))
    assert {p.name for p in page_paths} == {"ts_mean.md", "rank.md"}
    page = parse_page(tmp_path / "operators" / "ts_mean.md")
    assert page.type.value == "operator"
    assert "Time series mean" in page.body
    assert "managed by alphagen-agent" in page.body  # marker present


def test_wq_importer_accepts_legacy_managed_marker(tmp_path: Path):
    from alphagen_agent.wiki.importers.wq import _write_if_new_or_managed

    target = tmp_path / "legacy.md"
    target.write_text(
        "<!-- managed by wq-agent wiki import-wq; manual edits below -->\nold",
        encoding="utf-8",
    )

    assert _write_if_new_or_managed(target, "new") is True
    assert target.read_text(encoding="utf-8") == "new"


@pytest.mark.asyncio
async def test_wq_importer_handles_nested_dict_fields(tmp_path: Path):
    """WQ /data-sets 与 /data-fields 经常把 category/region/universe/type 返回成 {id,name} dict。"""
    client = AsyncMock()
    client.get_operators = AsyncMock(return_value=[])
    client.get_all_datasets = AsyncMock(return_value=[
        {
            "id": "fundamental6",
            "name": "Fundamental 6",
            "category": {"id": "fundamental", "name": "Fundamental"},
            "region": {"id": "USA", "name": "USA"},
            "universe": {"id": "TOP3000"},
            "delay": 1,
            "description": "Quarterly fundamentals",
            "fieldCount": 123,
        },
    ])
    client.get_all_data_fields_paged = AsyncMock(return_value=[
        {
            "id": "fnd6_test_field",
            "description": {"id": "wrapped", "name": "wrapped desc"},
            "type": {"id": "MATRIX"},
            "dataset": {"id": "fundamental6"},
            "coverage": [{"region": "USA", "value": 0.95}],
        },
    ])
    importer = WQDocImporter(wiki_root=tmp_path, client=client)
    stats = await importer.import_all(region="USA", universe="TOP3000", delay=1, include_fields=True)
    assert stats.datasets == 1
    assert stats.fields == 1

    ds_page = parse_page(tmp_path / "datasets" / "fundamental6.md")
    assert "fundamental" in ds_page.tags
    assert "usa" in ds_page.tags
    assert "Fundamental 6" in ds_page.body
    # 字段清单已内嵌进 dataset 页
    assert "fnd6_test_field" in ds_page.body
    assert "字段清单" in ds_page.body

    field_page = parse_page(tmp_path / "fields" / "fnd6_test_field.md")
    assert "fundamental6" in field_page.tags
    assert "matrix" in field_page.tags


@pytest.mark.asyncio
async def test_wq_importer_default_skips_per_field_pages_but_writes_dict(tmp_path: Path):
    """include_fields=False（默认）：字段只内嵌到 dataset 页 + 写到 dictionary，不生成 7000 个 md。"""
    client = AsyncMock()
    client.get_operators = AsyncMock(return_value=[])
    client.get_all_datasets = AsyncMock(return_value=[
        {"id": "pv1", "name": "Price Volume", "category": "pv", "delay": 1, "region": "USA"},
    ])
    client.get_all_data_fields_paged = AsyncMock(return_value=[
        {"id": "close", "type": "MATRIX", "dataset": {"id": "pv1"}, "description": "Closing price"},
        {"id": "volume", "type": "MATRIX", "dataset": {"id": "pv1"}, "description": "Daily volume"},
    ])
    importer = WQDocImporter(wiki_root=tmp_path, client=client)
    stats = await importer.import_all(region="USA", universe="TOP3000", delay=1)  # default include_fields=False
    assert stats.datasets == 1
    assert stats.fields == 0   # 没有写 per-field 页
    assert not (tmp_path / "fields").exists() or not any((tmp_path / "fields").iterdir())
    # 但 dataset 页含字段清单
    ds_body = (tmp_path / "datasets" / "pv1.md").read_text(encoding="utf-8")
    assert "close" in ds_body and "volume" in ds_body
    # 字段 ID 落到了 dictionary/auto.txt
    auto_txt = (tmp_path / "dictionary" / "auto.txt").read_text(encoding="utf-8")
    assert "close" in auto_txt and "volume" in auto_txt


@pytest.mark.asyncio
async def test_wq_importer_respects_user_edits(tmp_path: Path):
    client = AsyncMock()
    client.get_operators = AsyncMock(return_value=[
        WQOperator(name="ts_mean", category="Time Series", type="SCALAR",
                   description="Original description."),
    ])
    importer = WQDocImporter(wiki_root=tmp_path, client=client)
    await importer.import_operators()
    # 用户改了页
    path = tmp_path / "operators" / "ts_mean.md"
    path.write_text(
        "---\ntitle: ts_mean (user)\ntype: operator\ntags: [user]\ncreated: 2026-05-22\n---\n\nuser body\n",
        encoding="utf-8",
    )
    # 再跑 importer，不应覆盖
    n = await importer.import_operators()
    assert n == 0
    assert "user body" in path.read_text(encoding="utf-8")


def test_wq_tutorial_html_to_markdown_handles_rich_content():
    html = """
    <h2>Details</h2>
    <p>Use <b>rank</b> with <a href="https://example.test">docs</a>.</p>
    <ul><li>First</li><li><code>ts_mean(close, 20)</code></li></ul>
    <pre>alpha = rank(close)</pre>
    <table><tr><th>Name</th><th>Value</th></tr><tr><td>A</td><td>1</td></tr></table>
    <a href="$tutorialpage/create-alphas/first-alpha">Internal page</a>
    """
    markdown = html_to_markdown(html)
    assert "### Details" in markdown
    assert "**rank**" in markdown
    assert "[docs](https://example.test)" in markdown
    assert "- First" in markdown
    assert "`ts_mean(close, 20)`" in markdown
    assert "```\nalpha = rank(close)\n```" in markdown
    assert "| Name | Value |" in markdown
    assert (
        "[Internal page](https://platform.worldquantbrain.com/learn/documentation/"
        "create-alphas/first-alpha)" in markdown
    )


@pytest.mark.asyncio
async def test_wq_tutorial_importer_renders_blocks_and_skips_unchanged(tmp_path: Path):
    tutorial = {
        "id": "create-alphas",
        "title": "Create Alphas",
        "category": "Core",
        "sequence": 1,
        "duration": "PT10M",
        "lastModified": "2026-07-01T00:00:00Z",
        "pages": [
            {
                "id": "first-alpha",
                "title": "First Alpha",
                "duration": "PT3M",
                "lastModified": "2026-07-02T00:00:00Z",
            }
        ],
    }
    page = {
        "id": "first-alpha",
        "title": "First Alpha",
        "duration": "PT3M",
        "lastModified": "2026-07-02T00:00:00Z",
        "content": [
            {"type": "TEXT", "value": "<p>Start with <b>volume</b>.</p>"},
            {"type": "HEADING", "value": {"level": "1", "content": "Example"}},
            {"type": "IMAGE", "value": {"title": "chart.png", "url": "https://img.test/chart.png"}},
            {"type": "EQUATION", "value": "return = pnl / capital"},
            {
                "type": "SIMULATION_EXAMPLE",
                "value": {
                    "type": "REGULAR",
                    "regular": "rank(volume)",
                    "settings": {"region": "USA", "delay": 1},
                },
            },
            {
                "type": "TABLE",
                "value": {"data": [["Field", "Meaning"], ["volume", "Daily volume"]]},
            },
        ],
    }
    client = AsyncMock()
    client.get_all_tutorials = AsyncMock(return_value=[tutorial])
    client.get_tutorial_page = AsyncMock(return_value=page)
    importer = WQTutorialImporter(tmp_path, client)

    stats = await importer.import_all()
    assert stats.groups == 1
    assert stats.pages == 1
    assert stats.skipped == 0

    path = tmp_path / "worldquant-docs" / "create-alphas" / "first-alpha.md"
    parsed = parse_page(path)
    assert parsed.type.value == "concept"
    assert parsed.extra["source_last_modified"] == "2026-07-02T00:00:00Z"
    assert "**volume**" in parsed.body
    assert "rank(volume)" in parsed.body
    assert "| Field | Meaning |" in parsed.body
    assert "https://img.test/chart.png" in parsed.body

    index = (path.parent / "index.md").read_text(encoding="utf-8")
    assert "[[worldquant-docs/create-alphas/first-alpha|First Alpha]]" in index

    second = await importer.import_all()
    assert second.pages == 0
    assert second.skipped == 1
    client.get_tutorial_page.assert_awaited_once_with("first-alpha")

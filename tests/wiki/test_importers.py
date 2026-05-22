from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from wq_agent.models import WQOperator
from wq_agent.wiki.importers.papers import (
    PaperImporter,
    PaperRecord,
    fetch_arxiv,
    parse_url,
    slugify,
)
from wq_agent.wiki.importers.wq import WQDocImporter
from wq_agent.wiki.schema import parse_page


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
    assert "managed by wq-agent" in page.body  # marker present


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
    stats = await importer.import_all(region="USA", universe="TOP3000", delay=1)
    assert stats.datasets == 1
    assert stats.fields == 1

    ds_page = parse_page(tmp_path / "datasets" / "fundamental6.md")
    assert "fundamental" in ds_page.tags
    assert "usa" in ds_page.tags
    assert "Fundamental 6" in ds_page.body

    field_page = parse_page(tmp_path / "fields" / "fnd6_test_field.md")
    assert "fundamental6" in field_page.tags
    assert "matrix" in field_page.tags


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

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
import yaml
from loguru import logger


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9a-zA-Z\.\-_/]+?)(?:v\d+)?(?:\.pdf)?(?:\?.*)?$", re.I)
_SSRN_ID_RE = re.compile(r"ssrn(?:\.com)?/.+?(?:abstract_?id=|abstract=)(\d+)", re.I)
_MANAGED_MARKER = "<!-- managed by alphagen-agent wiki import-paper; manual edits below -->"
_LEGACY_MANAGED_MARKER = "<!-- managed by wq-agent wiki import-paper; manual edits below -->"


def slugify(text: str, max_len: int = 80) -> str:
    s = _SLUG_RE.sub("-", text.lower()).strip("-")
    return s[:max_len] or "paper"


@dataclass
class PaperRecord:
    title: str
    authors: list[str]
    year: int | None
    abstract: str
    url: str
    source: str  # arxiv | ssrn | manual | generic
    identifier: str = ""

    def to_page(self, tags: list[str] | None = None) -> str:
        tags = list(tags or [])
        tags.extend(["paper", self.source])
        front_data = {
            "title": self.title,
            "type": "concept",
            "tags": sorted({t.lower() for t in tags if t})[:10],
            "sources": [self.url],
            "created": date.today().isoformat(),
            "authors": self.authors,
            "year": self.year,
            "paper_source": self.source,
            "identifier": self.identifier,
        }
        front = "---\n" + yaml.safe_dump(
            front_data, sort_keys=False, allow_unicode=True
        ) + "---\n"
        author_line = ", ".join(self.authors) if self.authors else "—"
        year_line = self.year or "—"
        body = f"""
{_MANAGED_MARKER}

# {self.title}

**作者**：{author_line}　**年份**：{year_line}　**来源**：[{self.source}]({self.url})

## 摘要

{self.abstract.strip() or '（待补充）'}

## Key Takeaway（人工补充）

- TODO：核心结论用 1-3 句概括，便于检索

## WQ Brain 因子实现（人工补充）

```
TODO：把论文的因子定义翻译成 FASTEXPR
```

## 相关

- TODO：在 wiki/concepts/ 下补充相关因子家族页，用 `[[页名]]` 互链
"""
        return front + body

    def filename(self) -> str:
        if self.year and self.authors:
            stem = f"{self.year}-{slugify(self.authors[0].split(',')[0].split(' ')[-1])}-{slugify(self.title, 60)}"
        else:
            stem = slugify(self.title)
        return f"{stem[:120]}.md"


async def fetch_arxiv(arxiv_id: str, client: httpx.AsyncClient | None = None) -> PaperRecord:
    """Use arxiv export API; no auth required."""
    api = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    own = client is None
    cli = client or httpx.AsyncClient(timeout=30.0)
    try:
        resp = await cli.get(api)
    finally:
        if own:
            await cli.aclose()
    if resp.status_code != 200:
        raise RuntimeError(f"arxiv API error {resp.status_code}: {resp.text[:200]}")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    entry = root.find("a:entry", ns)
    if entry is None:
        raise RuntimeError(f"arxiv no entry for id {arxiv_id}")
    title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
    abstract = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
    published = entry.findtext("a:published", default="", namespaces=ns) or ""
    year = int(published[:4]) if published[:4].isdigit() else None
    authors = [
        (a.findtext("a:name", default="", namespaces=ns) or "").strip()
        for a in entry.findall("a:author", ns)
    ]
    authors = [a for a in authors if a]
    url = f"https://arxiv.org/abs/{arxiv_id}"
    return PaperRecord(
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
        url=url,
        source="arxiv",
        identifier=arxiv_id,
    )


async def fetch_ssrn(abstract_id: str, client: httpx.AsyncClient | None = None) -> PaperRecord:
    """Scrape SSRN abstract page (title + abstract from meta + DOM)."""
    url = f"https://papers.ssrn.com/sol3/papers.cfm?abstract_id={abstract_id}"
    own = client is None
    cli = client or httpx.AsyncClient(
        timeout=30.0,
        headers={"User-Agent": "Mozilla/5.0 (compatible; alphagen-agent paper importer)"},
        follow_redirects=True,
    )
    try:
        resp = await cli.get(url)
    finally:
        if own:
            await cli.aclose()
    if resp.status_code != 200:
        raise RuntimeError(f"SSRN error {resp.status_code} for id {abstract_id}")
    html = resp.text
    title = _meta(html, "citation_title") or _meta(html, "og:title") or f"SSRN {abstract_id}"
    abstract = _meta(html, "citation_abstract") or _meta(html, "description") or ""
    authors_raw = _meta_all(html, "citation_author")
    year = None
    date_meta = _meta(html, "citation_publication_date") or _meta(html, "citation_date") or ""
    if len(date_meta) >= 4 and date_meta[:4].isdigit():
        year = int(date_meta[:4])
    return PaperRecord(
        title=title.strip(),
        authors=[a.strip() for a in authors_raw if a.strip()],
        year=year,
        abstract=abstract.strip(),
        url=url,
        source="ssrn",
        identifier=abstract_id,
    )


def _meta(html: str, name: str) -> str:
    patterns = [
        rf'<meta\s+name=["\']{re.escape(name)}["\']\s+content=["\'](.*?)["\']',
        rf'<meta\s+property=["\']{re.escape(name)}["\']\s+content=["\'](.*?)["\']',
        rf'<meta\s+content=["\'](.*?)["\']\s+name=["\']{re.escape(name)}["\']',
    ]
    for p in patterns:
        m = re.search(p, html, re.I | re.S)
        if m:
            return m.group(1)
    return ""


def _meta_all(html: str, name: str) -> list[str]:
    out: list[str] = []
    pattern = rf'<meta\s+name=["\']{re.escape(name)}["\']\s+content=["\'](.*?)["\']'
    for m in re.finditer(pattern, html, re.I | re.S):
        out.append(m.group(1))
    return out


def parse_url(url: str) -> tuple[str, str] | None:
    """Returns (source, identifier) or None for unknown URLs."""
    arx = _ARXIV_ID_RE.search(url)
    if arx:
        return "arxiv", arx.group(1)
    ssrn = _SSRN_ID_RE.search(url)
    if ssrn:
        return "ssrn", ssrn.group(1)
    return None


class PaperImporter:
    def __init__(self, wiki_root: Path):
        self.wiki_root = Path(wiki_root)

    async def import_url(self, url: str, tags: list[str] | None = None) -> Path:
        parsed = parse_url(url)
        if parsed is None:
            raise ValueError(
                f"Unsupported URL: {url}. Use arxiv.org/abs/<id> or ssrn.com/...?abstract_id=<id>, "
                "or use import_manual()."
            )
        source, ident = parsed
        if source == "arxiv":
            record = await fetch_arxiv(ident)
        elif source == "ssrn":
            record = await fetch_ssrn(ident)
        else:
            raise ValueError(f"Unhandled source: {source}")
        return self._write(record, tags)

    def import_manual(
        self,
        title: str,
        authors: list[str],
        year: int | None,
        url: str,
        abstract: str = "",
        tags: list[str] | None = None,
    ) -> Path:
        record = PaperRecord(
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            url=url,
            source="manual",
        )
        return self._write(record, tags)

    def _write(self, record: PaperRecord, tags: list[str] | None) -> Path:
        out_dir = self.wiki_root / "papers"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / record.filename()
        content = record.to_page(tags)
        if path.exists():
            existing = path.read_text(encoding="utf-8", errors="replace")
            if not any(marker in existing for marker in (_MANAGED_MARKER, _LEGACY_MANAGED_MARKER)):
                logger.info(f"Refusing to overwrite manually-edited {path}")
                return path
        path.write_text(content, encoding="utf-8")
        logger.info(f"Wrote paper page {path.name}")
        return path

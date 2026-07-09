from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from ...wq.client import WQClient


_MANAGED_MARKER = "<!-- managed by alphagen-agent wiki import-wq; manual edits below -->"
_LEGACY_MANAGED_MARKER = "<!-- managed by wq-agent wiki import-wq; manual edits below -->"
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SPACE_RE = re.compile(r"\s+")


@dataclass
class TutorialImportStats:
    groups: int = 0
    pages: int = 0
    skipped: int = 0


def _slug(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip().lower())
    return text[:120] or "untitled"


def _normalize_href(value: str | None) -> str:
    href = str(value or "").strip()
    tutorial_prefix = "$tutorialpage/"
    if href.startswith(tutorial_prefix):
        suffix = href.removeprefix(tutorial_prefix).lstrip("/")
        return f"https://platform.worldquantbrain.com/learn/documentation/{suffix}"
    references = {
        "$reference/operators": "https://platform.worldquantbrain.com/learn/operators",
        "$reference/datasets": "https://platform.worldquantbrain.com/learn/data-and-operators/datasets",
    }
    return references.get(href, href)


def _escape_table_cell(value: Any) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip().replace("|", "\\|")


def _markdown_table(rows: list[list[Any]]) -> str:
    clean = [[_escape_table_cell(cell) for cell in row] for row in rows if row]
    if not clean:
        return ""
    width = max(len(row) for row in clean)
    clean = [row + [""] * (width - len(row)) for row in clean]
    lines = [
        "| " + " | ".join(clean[0]) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in clean[1:])
    return "\n".join(lines)


class _HTMLToMarkdown(HTMLParser):
    """Small, dependency-free converter for the tag set used by BRAIN tutorials."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.output: list[str] = []
        self.links: list[str] = []
        self.list_depth = 0
        self.pre_depth = 0
        self.ignored_depth = 0
        self.table_depth = 0
        self.table_rows: list[list[str]] = []
        self.table_row: list[str] | None = None
        self.table_cell: list[str] | None = None

    def _append(self, value: str) -> None:
        self.output.append(value)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "style":
            self.ignored_depth += 1
            return
        if self.ignored_depth:
            return

        if tag == "table":
            self.table_depth += 1
            if self.table_depth == 1:
                self.table_rows = []
            return
        if self.table_depth:
            if tag == "tr":
                self.table_row = []
            elif tag in {"td", "th"}:
                self.table_cell = []
            elif tag == "br" and self.table_cell is not None:
                self.table_cell.append(" ")
            return

        if tag in {"p", "div"}:
            self._append("\n\n")
        elif tag == "br":
            self._append("\n")
        elif tag in {"ul", "ol"}:
            self.list_depth += 1
            self._append("\n")
        elif tag == "li":
            self._append("\n" + "  " * max(0, self.list_depth - 1) + "- ")
        elif tag in {"b", "strong"}:
            self._append("**")
        elif tag in {"i", "em"}:
            self._append("_")
        elif tag == "code" and not self.pre_depth:
            self._append("`")
        elif tag == "pre":
            self.pre_depth += 1
            self._append("\n\n```\n")
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = min(6, int(tag[1]) + 1)
            self._append("\n\n" + "#" * level + " ")
        elif tag == "blockquote":
            self._append("\n\n> ")
        elif tag == "hr":
            self._append("\n\n---\n\n")
        elif tag == "a":
            self.links.append(_normalize_href(attrs_dict.get("href")))
            self._append("[")
        elif tag == "img":
            src = attrs_dict.get("src") or ""
            alt = attrs_dict.get("alt") or "image"
            if src:
                self._append(f"![{alt}]({src})")
        elif tag == "iframe":
            src = attrs_dict.get("src") or ""
            title = attrs_dict.get("title") or "Embedded content"
            if src:
                self._append(f"\n\n[{title}]({src})\n\n")
        elif tag == "sub":
            self._append("<sub>")

    def handle_endtag(self, tag: str) -> None:
        if tag == "style" and self.ignored_depth:
            self.ignored_depth -= 1
            return
        if self.ignored_depth:
            return

        if self.table_depth:
            if tag in {"td", "th"} and self.table_cell is not None:
                if self.table_row is not None:
                    self.table_row.append(_escape_table_cell("".join(self.table_cell)))
                self.table_cell = None
            elif tag == "tr" and self.table_row is not None:
                if self.table_row:
                    self.table_rows.append(self.table_row)
                self.table_row = None
            elif tag == "table":
                self.table_depth -= 1
                if self.table_depth == 0:
                    rendered = _markdown_table(self.table_rows)
                    if rendered:
                        self._append("\n\n" + rendered + "\n\n")
            return

        if tag in {"p", "div"}:
            self._append("\n\n")
        elif tag in {"ul", "ol"}:
            self.list_depth = max(0, self.list_depth - 1)
            self._append("\n")
        elif tag in {"b", "strong"}:
            self._append("**")
        elif tag in {"i", "em"}:
            self._append("_")
        elif tag == "code" and not self.pre_depth:
            self._append("`")
        elif tag == "pre":
            self._append("\n```\n\n")
            self.pre_depth = max(0, self.pre_depth - 1)
        elif tag == "a":
            href = self.links.pop() if self.links else ""
            self._append(f"]({href})" if href else "]")
        elif tag == "sub":
            self._append("</sub>")

    def handle_data(self, data: str) -> None:
        if self.ignored_depth:
            return
        if self.table_depth and self.table_cell is not None:
            self.table_cell.append(data)
            return
        if self.pre_depth:
            self._append(data)
            return
        value = _SPACE_RE.sub(" ", data)
        if value.strip() or (value and self.output):
            self._append(value)

    def markdown(self) -> str:
        text = "".join(self.output).replace("\xa0", " ")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def html_to_markdown(value: str) -> str:
    parser = _HTMLToMarkdown()
    parser.feed(value or "")
    parser.close()
    return parser.markdown()


def _render_simulation(value: dict[str, Any]) -> str:
    simulation_type = str(value.get("type") or "REGULAR")
    expressions: list[str] = []
    for key in ("regular", "selection", "combo"):
        expression = value.get(key)
        if expression:
            expressions.append(f"**{key.title()} expression**\n\n```fastexpr\n{expression}\n```")
    if not expressions:
        expressions.append(
            "```json\n"
            + json.dumps({k: v for k, v in value.items() if k != "settings"}, indent=2)
            + "\n```"
        )
    settings = value.get("settings") or {}
    setting_rows = [["Setting", "Value"], *[[key, val] for key, val in settings.items()]]
    parts = [f"**Simulation example ({simulation_type})**", *expressions]
    if settings:
        parts.extend(["**Settings**", _markdown_table(setting_rows)])
    return "\n\n".join(parts)


def render_content_blocks(blocks: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for block in blocks:
        kind = str(block.get("type") or "").upper()
        value = block.get("value")
        if kind == "TEXT":
            text = html_to_markdown(str(value or ""))
        elif kind == "HEADING" and isinstance(value, dict):
            level = min(6, max(2, int(value.get("level") or 1) + 1))
            text = "#" * level + " " + str(value.get("content") or "").strip()
        elif kind == "IMAGE" and isinstance(value, dict):
            url = str(value.get("url") or "")
            title = str(value.get("title") or "image").replace("]", "")
            text = f"![{title}]({url})" if url else ""
        elif kind == "EQUATION":
            text = f"**Equation**\n\n```text\n{value or ''}\n```"
        elif kind == "SIMULATION_EXAMPLE" and isinstance(value, dict):
            text = _render_simulation(value)
        elif kind == "TABLE" and isinstance(value, dict):
            text = _markdown_table(value.get("data") or [])
        else:
            text = (
                f"```json\n{json.dumps(value, ensure_ascii=False, indent=2, default=str)}\n```"
                if value is not None
                else ""
            )
        if text:
            rendered.append(text.strip())
    return "\n\n".join(rendered).strip()


def _frontmatter(title: str, tags: list[str], extra: dict[str, Any]) -> str:
    data = {
        "title": title,
        "type": "concept",
        "tags": sorted({tag for tag in tags if tag})[:10],
        "sources": ["worldquantbrain-api"],
        "created": date.today().isoformat(),
        **extra,
    }
    return "---\n" + yaml.safe_dump(data, sort_keys=False, allow_unicode=True) + "---\n"


def _is_current_managed_page(path: Path, source_last_modified: str) -> bool:
    if not path.exists():
        return False
    existing = path.read_text(encoding="utf-8", errors="replace")
    if not any(marker in existing for marker in (_MANAGED_MARKER, _LEGACY_MANAGED_MARKER)):
        return True
    match = _FRONTMATTER_RE.match(existing)
    if not match or not source_last_modified:
        return False
    metadata = yaml.safe_load(match.group(1)) or {}
    return str(metadata.get("source_last_modified") or "") == source_last_modified


def _write_managed(path: Path, content: str) -> bool:
    if path.exists():
        existing = path.read_text(encoding="utf-8", errors="replace")
        if not any(marker in existing for marker in (_MANAGED_MARKER, _LEGACY_MANAGED_MARKER)):
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


class WQTutorialImporter:
    def __init__(self, wiki_root: Path, client: WQClient):
        self.wiki_root = Path(wiki_root)
        self.client = client

    async def import_all(self, force: bool = False) -> TutorialImportStats:
        stats = TutorialImportStats()
        tutorials = await self.client.get_all_tutorials()
        root = self.wiki_root / "worldquant-docs"
        for tutorial in sorted(tutorials, key=lambda item: item.get("sequence", 0)):
            tutorial_id = str(tutorial.get("id") or "")
            if not tutorial_id:
                continue
            group_dir = root / _slug(tutorial_id)
            page_links: list[tuple[str, str]] = []
            for page_summary in tutorial.get("pages") or []:
                page_id = str(page_summary.get("id") or "")
                if not page_id:
                    continue
                path = group_dir / f"{_slug(page_id)}.md"
                last_modified = str(page_summary.get("lastModified") or "")
                title = str(page_summary.get("title") or page_id).strip()
                page_links.append((page_id, title))
                if not force and _is_current_managed_page(path, last_modified):
                    stats.skipped += 1
                    continue
                page = await self.client.get_tutorial_page(page_id)
                if not page:
                    logger.warning(f"Skipping unavailable tutorial page: {page_id}")
                    stats.skipped += 1
                    continue
                content = self._render_page(tutorial, page)
                if _write_managed(path, content):
                    stats.pages += 1
                else:
                    stats.skipped += 1
            index_path = group_dir / "index.md"
            if _write_managed(index_path, self._render_group(tutorial, page_links)):
                stats.groups += 1
        return stats

    @staticmethod
    def _render_page(tutorial: dict[str, Any], page: dict[str, Any]) -> str:
        tutorial_id = str(tutorial.get("id") or "")
        page_id = str(page.get("id") or "")
        title = str(page.get("title") or page_id).strip()
        category = str(tutorial.get("category") or page.get("category") or "")
        source_url = (
            "https://platform.worldquantbrain.com/learn/documentation/"
            f"{tutorial_id}/{page_id}"
        )
        front = _frontmatter(
            title,
            ["worldquant", "documentation", _slug(tutorial_id), _slug(category)],
            {
                "tutorial_id": tutorial_id,
                "page_id": page_id,
                "category": category,
                "duration": page.get("duration"),
                "source_last_modified": page.get("lastModified"),
                "source_url": source_url,
            },
        )
        body = render_content_blocks(page.get("content") or [])
        return (
            front
            + "\n"
            + _MANAGED_MARKER
            + f"\n\n# {title}\n\n"
            + f"[在 WorldQuant BRAIN 查看原文]({source_url})\n\n"
            + body
            + "\n"
        )

    @staticmethod
    def _render_group(
        tutorial: dict[str, Any],
        page_links: list[tuple[str, str]],
    ) -> str:
        tutorial_id = str(tutorial.get("id") or "")
        title = str(tutorial.get("title") or tutorial_id).strip()
        category = str(tutorial.get("category") or "")
        source_url = (
            "https://platform.worldquantbrain.com/learn/documentation/"
            f"{tutorial_id}"
        )
        front = _frontmatter(
            title,
            ["worldquant", "documentation", "documentation-index", _slug(category)],
            {
                "tutorial_id": tutorial_id,
                "category": category,
                "duration": tutorial.get("duration"),
                "source_last_modified": tutorial.get("lastModified"),
                "source_url": source_url,
            },
        )
        lines = [
            "",
            _MANAGED_MARKER,
            "",
            f"# {title}",
            "",
            f"**Category**：{category or 'N/A'}",
            "",
            "## Pages",
            "",
        ]
        lines.extend(
            f"- [[worldquant-docs/{_slug(tutorial_id)}/{_slug(page_id)}|{page_title}]]"
            for page_id, page_title in page_links
        )
        lines.append("")
        return front + "\n".join(lines)

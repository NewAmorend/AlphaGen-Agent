from __future__ import annotations

import enum
import hashlib
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml


class PageType(str, enum.Enum):
    CONCEPT = "concept"
    ENTITY = "entity"
    OPERATOR = "operator"
    FIELD = "field"
    PATTERN = "pattern"
    RECIPE = "recipe"
    LESSON = "lesson"
    ENTRY = "entry"


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_WIKILINK_RE = re.compile(r"\[\[([^\]\|]+?)(?:\|[^\]]+)?\]\]")


@dataclass
class Page:
    path: Path
    title: str
    type: PageType
    tags: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    created: date | None = None
    body: str = ""
    wikilinks: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return self.path.stem

    @property
    def rel_path(self) -> str:
        return str(self.path)

    @property
    def content_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.title.encode("utf-8"))
        h.update(b"\n")
        h.update(self.body.encode("utf-8"))
        return h.hexdigest()[:16]

    def summary(self, max_chars: int = 200) -> str:
        body = re.sub(r"\s+", " ", self.body).strip()
        if len(body) <= max_chars:
            return body
        return body[: max_chars - 1].rstrip() + "…"


def parse_page(path: Path, text: str | None = None) -> Page:
    raw = text if text is not None else path.read_text(encoding="utf-8")
    fm_match = _FRONTMATTER_RE.match(raw)
    if not fm_match:
        raise ValueError(f"Page {path} missing YAML frontmatter")
    fm_raw = fm_match.group(1)
    body = raw[fm_match.end():].strip()
    fm = yaml.safe_load(fm_raw) or {}

    title = fm.get("title")
    if not title:
        raise ValueError(f"Page {path} missing 'title'")
    type_str = fm.get("type")
    if not type_str:
        raise ValueError(f"Page {path} missing 'type'")
    try:
        page_type = PageType(type_str)
    except ValueError as exc:
        raise ValueError(f"Page {path} has unknown type '{type_str}'") from exc

    tags = fm.get("tags") or []
    if not isinstance(tags, list) or not tags:
        raise ValueError(f"Page {path} must have at least one tag")
    tags = [str(t) for t in tags]

    sources = fm.get("sources") or []
    if not isinstance(sources, list):
        sources = [str(sources)]
    sources = [str(s) for s in sources]

    created = fm.get("created")
    if isinstance(created, str):
        try:
            created = date.fromisoformat(created)
        except ValueError:
            created = None

    extra = {
        k: v
        for k, v in fm.items()
        if k not in {"title", "type", "tags", "sources", "created"}
    }

    wikilinks = sorted({m.group(1).strip() for m in _WIKILINK_RE.finditer(body)})

    return Page(
        path=path,
        title=str(title),
        type=page_type,
        tags=tags,
        sources=sources,
        created=created if isinstance(created, date) else None,
        body=body,
        wikilinks=wikilinks,
        extra=extra,
    )

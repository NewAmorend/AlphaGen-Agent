from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import yaml

from .schema import Page
from .store import WikiStore


MANAGED_MARKER = "<!-- managed by alphagen-agent wiki compile; manual edits below -->"
GENERATED_BY = "alphagen-agent wiki compile"


@dataclass
class TypedEdge:
    source: str
    target: str
    relation: str
    confidence: float
    evidence: str
    method: str = "deterministic"


@dataclass
class CompileStats:
    source_pages: int
    hubs_written: int
    hubs_skipped: int
    typed_edges: int
    topics_considered: int
    output_dir: str
    typed_edges_path: str


class WikiCompiler:
    """Compile raw wiki pages into auditable LLM-Wiki artifacts.

    The compiler intentionally stays deterministic: it turns page tags,
    wikilinks, and source metadata into hub pages and typed edges without
    asking an LLM to rewrite private research history.
    """

    NOISE_TAGS = {
        "compiled",
        "hub",
        "wiki",
        "docs",
        "documentation",
        "worldquant",
        "worldquantbrain",
        "worldquant-docs",
    }

    def __init__(self, store: WikiStore):
        self.store = store
        self.root = store.root

    def compile(
        self,
        *,
        max_hubs: int = 40,
        min_pages: int = 3,
        pages_per_hub: int = 12,
        max_doc_frac: float = 0.45,
    ) -> CompileStats:
        pages, _ = self.store.load_pages()
        source_pages = [p for p in pages if p.extra.get("generated_by") != GENERATED_BY]
        tag_counts = self._tag_counts(source_pages)
        max_docs = max(int(len(source_pages) * max_doc_frac), min_pages, 10)
        topics = [
            tag
            for tag, count in tag_counts.most_common()
            if count >= min_pages
            and count <= max_docs
            and tag.lower() not in self.NOISE_TAGS
        ][:max_hubs]

        hub_dir = self.root / "hubs"
        hubs_written = 0
        hubs_skipped = 0
        topic_to_hub_id: dict[str, str] = {}
        edges: list[TypedEdge] = []

        for topic in topics:
            ranked_pages = self._rank_topic_pages(topic, source_pages)[:pages_per_hub]
            if not ranked_pages:
                continue
            hub_path = hub_dir / f"{_slugify(topic)}.md"
            hub_id = self._page_id(hub_path)
            topic_to_hub_id[topic] = hub_id
            body = self._render_hub(topic, ranked_pages)
            if self._write_managed(hub_path, body):
                hubs_written += 1
            else:
                hubs_skipped += 1

            for page, score in ranked_pages:
                confidence = min(0.92, 0.62 + 0.06 * score)
                page_id = self._page_id(page.path)
                edges.append(TypedEdge(
                    source=page_id,
                    target=hub_id,
                    relation="has_topic",
                    confidence=round(confidence, 3),
                    evidence=f"tag/title/body match for topic '{topic}'",
                ))
                edges.append(TypedEdge(
                    source=hub_id,
                    target=page_id,
                    relation="summarizes",
                    confidence=round(confidence, 3),
                    evidence=f"compiled hub membership for topic '{topic}'",
                ))

        edges.extend(self._wikilink_edges(source_pages))
        edges.extend(self._shared_source_edges(source_pages))
        typed_edges_path = self.root / "typed_edges.json"
        self._write_typed_edges(typed_edges_path, edges, source_pages, topics)

        return CompileStats(
            source_pages=len(source_pages),
            hubs_written=hubs_written,
            hubs_skipped=hubs_skipped,
            typed_edges=len(edges),
            topics_considered=len(topics),
            output_dir=str(hub_dir),
            typed_edges_path=str(typed_edges_path),
        )

    @staticmethod
    def _tag_counts(pages: Iterable[Page]) -> Counter[str]:
        counts: Counter[str] = Counter()
        for page in pages:
            counts.update(str(tag).strip() for tag in page.tags if str(tag).strip())
        return counts

    def _rank_topic_pages(self, topic: str, pages: list[Page]) -> list[tuple[Page, int]]:
        topic_lc = topic.lower()
        scored: list[tuple[Page, int]] = []
        for page in pages:
            score = 0
            tags_lc = {t.lower() for t in page.tags}
            if topic_lc in tags_lc:
                score += 5
            identity = f"{page.title} {page.slug} {page.path}".lower()
            if topic_lc in identity:
                score += 3
            body_hits = page.body.lower().count(topic_lc)
            if body_hits:
                score += min(3, 1 + int(math.log2(body_hits + 1)))
            if score:
                scored.append((page, score))
        scored.sort(
            key=lambda item: (
                item[1],
                len(item[0].wikilinks),
                item[0].title.lower(),
            ),
            reverse=True,
        )
        return scored

    def _render_hub(self, topic: str, ranked_pages: list[tuple[Page, int]]) -> str:
        tags = ["compiled", "hub", topic]
        frontmatter = {
            "title": f"Hub: {topic}",
            "type": "concept",
            "tags": tags,
            "sources": [],
            "generated_by": GENERATED_BY,
            "compiled_topic": topic,
            "confidence": 0.78,
        }
        lines = [
            "---",
            yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False).strip(),
            "---",
            "",
            f"# Hub: {topic}",
            "",
            MANAGED_MARKER,
            "",
            "自动编译页：把同一主题下的页面集中成可检索、可审计的入口。"
            "如果这里的归类不准，优先改原页面 tags 或手写一个更权威的概念页。",
            "",
            "## Related pages",
            "",
        ]
        for page, score in ranked_pages:
            link = self._wikilink_target(page)
            summary = _summary_plain(page)
            lines.append(f"- [[{link}]] — **{page.title}**; score `{score}`; {summary}")
        lines.extend([
            "",
            "## Maintenance notes",
            "",
            "- 这个页面由 `alphagen-agent wiki compile` 生成。",
            "- `typed_edges.json` 中会同步生成 `has_topic` / `summarizes` 关系。",
        ])
        return "\n".join(lines).rstrip() + "\n"

    def _wikilink_edges(self, pages: list[Page]) -> list[TypedEdge]:
        targets = self._targets_by_link(pages)
        edges: list[TypedEdge] = []
        seen: set[tuple[str, str, str]] = set()
        for page in pages:
            source = self._page_id(page.path)
            for link in page.wikilinks:
                target_page = targets.get(_normalize_link(link))
                if not target_page:
                    continue
                target = self._page_id(target_page.path)
                key = (source, target, "references")
                if source == target or key in seen:
                    continue
                seen.add(key)
                edges.append(TypedEdge(
                    source=source,
                    target=target,
                    relation="references",
                    confidence=0.9,
                    evidence=f"explicit wikilink [[{link}]]",
                ))
        return edges

    def _shared_source_edges(self, pages: list[Page]) -> list[TypedEdge]:
        source_to_pages: dict[str, list[Page]] = defaultdict(list)
        for page in pages:
            for source in page.sources:
                if source:
                    source_to_pages[source].append(page)

        edges: list[TypedEdge] = []
        for source, bucket in source_to_pages.items():
            if not (2 <= len(bucket) <= 12):
                continue
            for i, a in enumerate(bucket):
                for b in bucket[i + 1:]:
                    edges.append(TypedEdge(
                        source=self._page_id(a.path),
                        target=self._page_id(b.path),
                        relation="same_source_as",
                        confidence=0.62,
                        evidence=f"shared source: {source}",
                    ))
                    edges.append(TypedEdge(
                        source=self._page_id(b.path),
                        target=self._page_id(a.path),
                        relation="same_source_as",
                        confidence=0.62,
                        evidence=f"shared source: {source}",
                    ))
        return edges

    def _targets_by_link(self, pages: list[Page]) -> dict[str, Page]:
        targets: dict[str, Page] = {}
        for page in pages:
            for key in {
                page.slug,
                page.title,
                str(page.path),
                self._page_id(page.path),
            }:
                targets[_normalize_link(key)] = page
        return targets

    def _write_typed_edges(self, path: Path, edges: list[TypedEdge], pages: list[Page], topics: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_by": GENERATED_BY,
            "version": 1,
            "stats": {
                "source_pages": len(pages),
                "topics": len(topics),
                "edges": len(edges),
            },
            "edges": [asdict(edge) for edge in edges],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _write_managed(self, path: Path, text: str) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and MANAGED_MARKER not in path.read_text(encoding="utf-8"):
            return False
        path.write_text(text, encoding="utf-8")
        return True

    def _wikilink_target(self, page: Page) -> str:
        return self._page_id(page.path)

    def _page_id(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root).with_suffix("")).replace("\\", "/")
        except ValueError:
            return str(path.with_suffix("")).replace("\\", "/")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^\w.-]+", "-", text.strip().lower(), flags=re.UNICODE)
    slug = re.sub(r"-{2,}", "-", slug).strip("-._")
    return slug[:96] or "topic"


def _normalize_link(link: str) -> str:
    text = str(link).strip().replace("\\", "/")
    if text.endswith(".md"):
        text = text[:-3]
    return text


def _plain_text(text: str) -> str:
    text = re.sub(r"\[\[([^\]\|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = text.replace("[[", "").replace("]]", "")
    return text


def _summary_plain(page: Page, max_chars: int = 180) -> str:
    text = _plain_text(page.body)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text

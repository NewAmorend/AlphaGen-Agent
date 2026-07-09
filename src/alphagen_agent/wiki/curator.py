from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
import json
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

from .schema import Page, PageType
from .store import WikiStore


_CODE_SYMBOL_RE = re.compile(r"`([a-z][a-z0-9_]{2,})`")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[一-鿿]+")


@dataclass(frozen=True)
class CuratorFinding:
    severity: str
    category: str
    message: str
    path: str = ""
    detail: str = ""


@dataclass(frozen=True)
class CuratorSuggestion:
    action: str
    target: str
    title: str
    reason: str
    applyable: bool = False
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class CuratorReport:
    pages: int
    errors: int
    broken_links: int
    findings: list[CuratorFinding] = field(default_factory=list)
    suggestions: list[CuratorSuggestion] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pages": self.pages,
            "errors": self.errors,
            "broken_links": self.broken_links,
            "findings": [asdict(f) for f in self.findings],
            "suggestions": [asdict(s) for s in self.suggestions],
            "applied": list(self.applied),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @property
    def applyable_suggestions(self) -> list[CuratorSuggestion]:
        return [s for s in self.suggestions if s.applyable]


class KnowledgeCurator:
    """Deterministic wiki curation agent.

    The curator audits wiki health and proposes low-risk maintenance actions. It is
    intentionally conservative: semantic consolidation is suggested, while apply()
    only writes deterministic bench additions and a machine-readable report.
    """

    def __init__(self, store: WikiStore, bench_path: str | Path | None = None):
        self.store = store
        self.bench_path = Path(bench_path) if bench_path else self.store.root / "bench" / "retrieval_golden.yml"

    def audit(
        self,
        *,
        since: date | None = None,
        max_suggestions: int = 30,
    ) -> CuratorReport:
        pages, errors = self.store.load_pages()
        broken = self.store.find_broken_links(pages)
        report = CuratorReport(
            pages=len(pages),
            errors=len(errors),
            broken_links=sum(len(misses) for _, misses in broken),
        )
        self._add_parse_findings(report, errors)
        self._add_broken_link_findings(report, broken)
        self._add_duplicate_title_findings(report, pages)
        self._add_todo_findings(report, pages, max_items=10)
        self._add_lesson_rollup_suggestions(report, pages, since=since)
        self._add_missing_field_suggestions(report, pages, max_items=10)
        self._add_bench_suggestions(report, pages)
        report.suggestions = report.suggestions[:max_suggestions]
        return report

    def apply(
        self,
        report: CuratorReport,
        *,
        max_bench_additions: int = 20,
        write_report: bool = True,
    ) -> CuratorReport:
        bench_actions = [s for s in report.applyable_suggestions if s.action == "add_bench_query"]
        additions = bench_actions[:max_bench_additions]
        if additions:
            count = self._append_bench_queries(additions)
            report.applied.append(f"appended {count} retrieval bench queries to {self._display_path(self.bench_path)}")
        if write_report:
            path = self.store.root / "curation_report.json"
            path.write_text(report.to_json() + "\n", encoding="utf-8")
            report.applied.append(f"wrote {self._display_path(path)}")
        return report

    def _add_parse_findings(self, report: CuratorReport, errors: list[tuple[Path, str]]) -> None:
        for path, err in errors:
            report.findings.append(
                CuratorFinding(
                    severity="critical",
                    category="parse_error",
                    path=self._display_path(path),
                    message="Page cannot be parsed",
                    detail=err,
                )
            )

    def _add_broken_link_findings(self, report: CuratorReport, broken: list[tuple[Page, list[str]]]) -> None:
        for page, misses in broken:
            report.findings.append(
                CuratorFinding(
                    severity="high",
                    category="broken_wikilink",
                    path=self._display_path(page.path),
                    message=f"{len(misses)} broken wikilink(s)",
                    detail=", ".join(misses[:8]),
                )
            )

    def _add_duplicate_title_findings(self, report: CuratorReport, pages: list[Page]) -> None:
        by_title: dict[str, list[Page]] = defaultdict(list)
        for page in pages:
            by_title[page.title.strip().lower()].append(page)
        for title, items in sorted(by_title.items()):
            if title and len(items) > 1:
                report.findings.append(
                    CuratorFinding(
                        severity="medium",
                        category="duplicate_title",
                        message=f"Duplicate title: {items[0].title}",
                        detail=", ".join(self._display_path(p.path) for p in items[:8]),
                    )
                )

    def _add_todo_findings(self, report: CuratorReport, pages: list[Page], max_items: int) -> None:
        count = 0
        for page in pages:
            if "TODO" not in page.body:
                continue
            if page.type in {PageType.ENTRY, PageType.LESSON}:
                continue
            report.findings.append(
                CuratorFinding(
                    severity="low",
                    category="todo_content",
                    path=self._display_path(page.path),
                    message="Page still contains TODO content",
                    detail=page.title,
                )
            )
            count += 1
            if count >= max_items:
                break

    def _add_lesson_rollup_suggestions(
        self,
        report: CuratorReport,
        pages: list[Page],
        *,
        since: date | None,
    ) -> None:
        lessons = [p for p in pages if p.type is PageType.LESSON and self._is_after(p, since)]
        buckets: dict[str, list[Page]] = defaultdict(list)
        for lesson in lessons:
            for bucket in self._classify_lesson(lesson):
                buckets[bucket].append(lesson)
        existing_targets = {self._target_for_page(p) for p in pages}
        for bucket, items in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            if len(items) < 2:
                continue
            target = f"patterns/{bucket}"
            action = "update_pattern" if target in existing_targets else "create_pattern"
            report.suggestions.append(
                CuratorSuggestion(
                    action=action,
                    target=target,
                    title=_humanize_slug(bucket),
                    reason=f"{len(items)} lesson pages mention this failure mode; consolidate them into a stable pattern page.",
                    applyable=False,
                    payload={"source_pages": [self._display_path(p.path) for p in items[:12]]},
                )
            )

    def _add_missing_field_suggestions(
        self,
        report: CuratorReport,
        pages: list[Page],
        *,
        max_items: int,
    ) -> None:
        field_targets = {p.slug for p in pages if p.type is PageType.FIELD}
        operator_targets = {p.slug for p in pages if p.type is PageType.OPERATOR}
        counter: Counter[str] = Counter()
        for page in pages:
            if page.type not in {PageType.ENTRY, PageType.LESSON, PageType.RECIPE, PageType.PATTERN}:
                continue
            for symbol in _CODE_SYMBOL_RE.findall(page.body):
                if symbol in field_targets or symbol in operator_targets:
                    continue
                if symbol.startswith(("ts_", "group_")):
                    continue
                counter[symbol] += 1
        for symbol, count in counter.most_common(max_items):
            if count < 2:
                continue
            report.suggestions.append(
                CuratorSuggestion(
                    action="create_field_page",
                    target=f"fields/{symbol}",
                    title=symbol,
                    reason=f"Referenced {count} times in curated pages but has no field page.",
                    applyable=False,
                    payload={"field_id": symbol},
                )
            )

    def _add_bench_suggestions(self, report: CuratorReport, pages: list[Page]) -> None:
        expected_targets = self._bench_expected_targets()
        important = {PageType.FIELD, PageType.PATTERN, PageType.RECIPE}
        for page in sorted(pages, key=lambda p: (p.type.value, str(p.path))):
            if page.type not in important:
                continue
            target = self._target_for_page(page)
            if target in expected_targets:
                continue
            report.suggestions.append(
                CuratorSuggestion(
                    action="add_bench_query",
                    target=target,
                    title=page.title,
                    reason="Important curated knowledge page is not covered by retrieval_golden.yml.",
                    applyable=True,
                    payload=self._bench_payload(page),
                )
            )

    def _append_bench_queries(self, suggestions: Iterable[CuratorSuggestion]) -> int:
        self.bench_path.parent.mkdir(parents=True, exist_ok=True)
        existing = self.bench_path.read_text(encoding="utf-8") if self.bench_path.exists() else ""
        chunks: list[str] = []
        count = 0
        expected_targets = self._bench_expected_targets()
        for suggestion in suggestions:
            target = suggestion.target
            if target in expected_targets:
                continue
            payload = suggestion.payload
            row = {
                "query": payload.get("query") or suggestion.title,
                "expected": payload.get("expected") or [target],
                "note": payload.get("note") or f"Curator coverage for {suggestion.title}",
            }
            chunks.append(yaml.safe_dump([row], allow_unicode=True, sort_keys=False).strip())
            expected_targets.add(target)
            count += 1
        if chunks:
            try:
                existing_data = yaml.safe_load(existing) if existing.strip() else []
            except Exception:
                existing_data = None
            prefix = "" if existing_data == [] else existing.rstrip() + "\n\n"
            self.bench_path.write_text(prefix + "\n\n".join(chunks) + "\n", encoding="utf-8")
        return count

    def _bench_expected_targets(self) -> set[str]:
        if not self.bench_path.exists():
            return set()
        try:
            data = yaml.safe_load(self.bench_path.read_text(encoding="utf-8")) or []
        except Exception:
            return set()
        targets: set[str] = set()
        if not isinstance(data, list):
            return targets
        for row in data:
            if not isinstance(row, dict):
                continue
            expected = row.get("expected") or []
            if isinstance(expected, str):
                expected = [expected]
            targets.update(_normalize_target(e) for e in expected)
        return targets

    def _bench_payload(self, page: Page) -> dict[str, Any]:
        target = self._target_for_page(page)
        terms = _query_terms(page)
        return {
            "query": " ".join(terms),
            "expected": [target],
            "note": f"Curator coverage for {page.type.value}: {page.title}",
        }

    def _classify_lesson(self, page: Page) -> set[str]:
        text = " ".join([page.title, " ".join(page.tags), page.body]).lower()
        buckets: set[str] = set()
        rules = {
            "high-turnover": ("turnover", "换手"),
            "low-coverage-nan": ("coverage", "nan", "missing", "缺失", "覆盖"),
            "direction-sign-error": ("negative fitness", "方向", "reverse", "sign"),
            "self-correlation-crowding": ("self-correlation", "correlation", "拥挤", "重复"),
            "overfit-parameter-search": ("overfit", "parameter", "参数", "过拟合"),
        }
        for bucket, needles in rules.items():
            if any(n in text for n in needles):
                buckets.add(bucket)
        return buckets

    def _is_after(self, page: Page, since: date | None) -> bool:
        if since is None or page.created is None:
            return True
        return page.created >= since

    def _target_for_page(self, page: Page) -> str:
        rel = page.path
        try:
            rel = page.path.relative_to(self.store.root)
        except ValueError:
            pass
        return _normalize_target(str(rel))

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.store.root))
        except ValueError:
            return str(path)


def parse_since(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def _normalize_target(value: Any) -> str:
    text = str(value).strip().replace("\\", "/")
    if text.endswith(".md"):
        text = text[:-3]
    if text.startswith("wiki/"):
        text = text[5:]
    return text


def _humanize_slug(slug: str) -> str:
    return slug.replace("-", " ").replace("_", " ")


def _query_terms(page: Page) -> list[str]:
    raw_terms = [page.title, page.slug, *page.tags[:6]]
    for key in ("field_id", "operator_name", "dataset_id"):
        value = page.extra.get(key)
        if isinstance(value, str):
            raw_terms.append(value)
    terms: list[str] = []
    seen: set[str] = set()
    for raw in raw_terms:
        for token in _WORD_RE.findall(str(raw)):
            token = token.strip()
            key = token.lower()
            if not token or key in seen:
                continue
            seen.add(key)
            terms.append(token)
    return terms[:10] or [page.slug]

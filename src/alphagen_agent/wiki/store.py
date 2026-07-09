from __future__ import annotations

from pathlib import Path
from typing import Iterator

from loguru import logger

from .schema import Page, parse_page


_RESERVED_DIRS = {"dictionary"}
_RESERVED_FILES = {"SCHEMA.md", "index.md", "log.md"}


class WikiStore:
    """File-system view of the wiki/ tree."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def exists(self) -> bool:
        return self.root.is_dir()

    def iter_page_paths(self) -> Iterator[Path]:
        if not self.exists():
            return
        for path in sorted(self.root.rglob("*.md")):
            rel = path.relative_to(self.root)
            if rel.parts and rel.parts[0] in _RESERVED_DIRS:
                continue
            if rel.name in _RESERVED_FILES and len(rel.parts) == 1:
                continue
            yield path

    def load_pages(self) -> tuple[list[Page], list[tuple[Path, str]]]:
        pages: list[Page] = []
        errors: list[tuple[Path, str]] = []
        for path in self.iter_page_paths():
            try:
                pages.append(parse_page(path))
            except Exception as exc:
                logger.warning(f"Skipping {path}: {exc}")
                errors.append((path, str(exc)))
        return pages, errors

    def find_broken_links(self, pages: list[Page]) -> list[tuple[Page, list[str]]]:
        targets = self._link_targets(pages)
        broken: list[tuple[Page, list[str]]] = []
        for p in pages:
            misses = [link for link in p.wikilinks if _normalize_link(link) not in targets]
            if misses:
                broken.append((p, misses))
        return broken

    def _link_targets(self, pages: list[Page]) -> set[str]:
        targets: set[str] = set()
        for p in pages:
            rel = p.path
            try:
                rel = p.path.relative_to(self.root)
            except ValueError:
                pass
            rel_s = _normalize_link(str(rel))
            targets.update({
                _normalize_link(p.slug),
                _normalize_link(p.title),
                rel_s,
                rel_s.removesuffix(".md"),
            })
        return targets

    def dictionary_path(self) -> Path:
        return self.root / "dictionary" / "base.txt"

    def auto_dictionary_path(self) -> Path:
        return self.root / "dictionary" / "auto.txt"

    def synonyms_path(self) -> Path:
        return self.root / "dictionary" / "synonyms.yaml"

    def graph_index_path(self) -> Path:
        return self.root / "graph_index.json"


class CompositeWikiStore(WikiStore):
    """Read pages from a public wiki plus optional private wiki roots.

    The primary root still owns dictionaries and generated graph artifacts. Extra
    roots are read-only inputs so auto-recorded private lessons can feed retrieval
    without being written back under the public wiki tree.
    """

    def __init__(self, root: str | Path, extra_roots: list[str | Path] | None = None):
        super().__init__(root)
        self.roots = [self.root, *(Path(p) for p in (extra_roots or []))]

    def exists(self) -> bool:
        return any(root.is_dir() for root in self.roots)

    def iter_page_paths(self) -> Iterator[Path]:
        seen: set[Path] = set()
        for root in self.roots:
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*.md")):
                rel = path.relative_to(root)
                if rel.parts and rel.parts[0] in _RESERVED_DIRS:
                    continue
                if rel.name in _RESERVED_FILES and len(rel.parts) == 1:
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield path

    def _link_targets(self, pages: list[Page]) -> set[str]:
        targets: set[str] = set()
        for p in pages:
            rel_s = None
            for root in self.roots:
                try:
                    rel_s = _normalize_link(str(p.path.relative_to(root)))
                    break
                except ValueError:
                    continue
            if rel_s is None:
                rel_s = _normalize_link(str(p.path))
            targets.update({
                _normalize_link(p.slug),
                _normalize_link(p.title),
                rel_s,
                rel_s.removesuffix(".md"),
            })
        return targets


def _normalize_link(link: str) -> str:
    text = str(link).strip().replace("\\", "/")
    if text.endswith(".md"):
        text = text[:-3]
    return text

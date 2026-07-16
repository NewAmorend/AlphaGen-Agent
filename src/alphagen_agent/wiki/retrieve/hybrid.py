from __future__ import annotations

from dataclasses import dataclass

from ..schema import Page
from .graph import GraphChannel
from .grep import GrepChannel
from .vector import VectorChannel


@dataclass
class HybridHit:
    page: Page
    score: float
    sources: list[str]   # 哪些通道贡献了：priority / grep / vector / graph
    note: str = ""


class HybridRetriever:
    """优先级 grep 置顶 + 加权 RRF (k=60, grep:vec=7:3) + graph 扩展。"""

    RRF_K = 60

    def __init__(
        self,
        pages: list[Page],
        grep: GrepChannel,
        vector: VectorChannel,
        graph: GraphChannel,
        grep_weight: int = 7,
        vector_weight: int = 3,
    ):
        self.pages = pages
        self.grep = grep
        self.vector = vector
        self.graph = graph
        self.grep_weight = grep_weight
        self.vector_weight = vector_weight

    async def search(self, query: str, top_k: int = 5) -> list[HybridHit]:
        if not self.pages:
            return []
        # 拉到比 top_k 更宽的池子做融合
        pool = max(top_k * 4, 10)
        grep_hits = self.grep.search(query, top_k=pool)
        vector_hits = await self.vector.search(query, top_k=pool)

        priority = [h for h in grep_hits if h.matched_all_originals or h.matched_identifier]
        normal_grep = [h for h in grep_hits if h not in priority]

        seen: dict[str, HybridHit] = {}

        grep_by_path = {str(h.page.path): h for h in grep_hits}
        vector_by_path = {str(h.page.path): h for h in vector_hits}
        grep_rank = {str(h.page.path): i + 1 for i, h in enumerate(grep_hits)}
        vector_rank = {str(h.page.path): i + 1 for i, h in enumerate(vector_hits)}

        for h in priority:
            note = self._grep_note(h, grep_rank.get(str(h.page.path)))
            self._add(seen, h.page, score=1.0 + h.score, source="priority", note=note)

        # 加权 RRF
        rrf_scores: dict[str, float] = {}
        for rank, h in enumerate(normal_grep):
            rrf_scores[str(h.page.path)] = rrf_scores.get(str(h.page.path), 0.0) + \
                self.grep_weight / (self.RRF_K + rank + 1)
        for rank, h in enumerate(vector_hits):
            rrf_scores[str(h.page.path)] = rrf_scores.get(str(h.page.path), 0.0) + \
                self.vector_weight / (self.RRF_K + rank + 1)

        merged_pages = {str(h.page.path): h.page for h in normal_grep + vector_hits}
        ranked = sorted(
            rrf_scores.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )
        for path, score in ranked:
            page = merged_pages.get(path)
            if page is None:
                continue
            sources: list[str] = []
            notes: list[str] = []
            if any(str(h.page.path) == path for h in normal_grep):
                sources.append("grep")
                gh = grep_by_path.get(path)
                if gh:
                    notes.append(self._grep_note(gh, grep_rank.get(path)))
            if any(str(h.page.path) == path for h in vector_hits):
                sources.append("vector")
                vh = vector_by_path.get(path)
                if vh:
                    rank = vector_rank.get(path)
                    notes.append(
                        f"vector rank #{rank}, cosine={vh.score:.3f}"
                        if rank else f"vector cosine={vh.score:.3f}"
                    )
            self._add(
                seen,
                page,
                score=score,
                source="+".join(sources) or "rrf",
                note="; ".join(n for n in notes if n),
            )

        # 用 hybrid top-3 作为种子做图扩展
        seeds = [hit.page.slug for hit in list(seen.values())[:3]]
        graph_hits = self.graph.expand(seeds, k=2)
        for gh in graph_hits:
            self._add(
                seen,
                gh.page,
                score=gh.score,
                source=f"graph:{gh.reason}",
                note=f"graph expansion from top seeds via {gh.reason}",
            )

        out = list(seen.values())
        out.sort(key=lambda h: h.score, reverse=True)
        return out[:top_k]

    def _add(self, seen: dict, page: Page, score: float, source: str, note: str = "") -> None:
        key = str(page.path)
        if key in seen:
            hit = seen[key]
            hit.score += score
            if source not in hit.sources:
                hit.sources.append(source)
            if note and note not in hit.note:
                hit.note = "; ".join(x for x in [hit.note, note] if x)
        else:
            seen[key] = HybridHit(page=page, score=score, sources=[source], note=note)

    @staticmethod
    def _grep_note(hit, rank: int | None = None) -> str:
        tokens = ", ".join(sorted(hit.matched_originals)) or "—"
        prefix = f"grep rank #{rank}" if rank else "grep"
        flags: list[str] = []
        if hit.matched_all_originals:
            flags.append("all original tokens")
        if hit.matched_identifier:
            flags.append("identifier match")
        suffix = f" ({', '.join(flags)})" if flags else ""
        return f"{prefix}: matched [{tokens}], score={hit.score:.3f}{suffix}"

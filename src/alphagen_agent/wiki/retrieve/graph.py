from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
from loguru import logger

from ..schema import Page


@dataclass
class GraphHit:
    page: Page
    score: float
    reason: str


class GraphChannel:
    """wikilink + shared_tag + shared_source edges → Louvain + PageRank.

    用倒排索引 + 频率过滤 + 硬上限构图，避免 O(n²) 在大 wiki 上 OOM。
    """

    MIN_SHARED_TAG = 2          # 至少共享 N 个标签才算"相似页"
    EXPAND_SHARED_TAG = 3       # 邻居扩展的更严格阈值
    MAX_TAG_DOC_FRAC = 0.30     # 出现在超过 30% 页里的标签 = 噪音，跳过
    MAX_TAG_BUCKET = 40         # 单个标签桶超过 40 页就降级（只在桶内 sample）
    MAX_SOURCE_BUCKET = 25
    MAX_TOTAL_NON_WIKILINK_EDGES = 50_000  # 总图边硬上限，超过截断
    MAX_NODES_FOR_LOUVAIN = 8_000
    MAX_NODES_FOR_PAGERANK = 8_000

    def __init__(self, pages: list[Page]):
        self.pages = pages
        self.pages_by_slug: dict[str, Page] = {p.slug: p for p in pages}
        self.pages_by_title: dict[str, Page] = {p.title: p for p in pages}
        self.pages_by_rel_path: dict[str, Page] = {}
        for p in pages:
            rel = str(p.path).replace("\\", "/")
            self.pages_by_rel_path[rel.removesuffix(".md")] = p
            if "/wiki/" in rel:
                self.pages_by_rel_path[rel.split("/wiki/", 1)[1].removesuffix(".md")] = p
        self.graph = self._build_graph(pages)
        self.pagerank: dict[str, float] = {}
        self.communities: dict[str, int] = {}
        n = self.graph.number_of_nodes()
        if 3 <= n <= self.MAX_NODES_FOR_PAGERANK:
            try:
                self.pagerank = nx.pagerank(self.graph, alpha=0.85, max_iter=30)
            except Exception as exc:
                logger.debug(f"PageRank failed: {exc}")
                self.pagerank = {nd: 1.0 / n for nd in self.graph.nodes}
        elif n > self.MAX_NODES_FOR_PAGERANK:
            logger.info(f"Skipping PageRank: {n} nodes > {self.MAX_NODES_FOR_PAGERANK}")
        if self.graph.number_of_edges() > 0 and n <= self.MAX_NODES_FOR_LOUVAIN:
            try:
                import community as community_louvain
                undirected = self.graph.to_undirected()
                self.communities = community_louvain.best_partition(undirected, random_state=42)
            except Exception as exc:
                logger.debug(f"Louvain failed: {exc}")
                self.communities = {}
        elif n > self.MAX_NODES_FOR_LOUVAIN:
            logger.info(f"Skipping Louvain: {n} nodes > {self.MAX_NODES_FOR_LOUVAIN}")

    def _resolve(self, name: str) -> Page | None:
        key = str(name).strip().replace("\\", "/").removesuffix(".md")
        return self.pages_by_slug.get(key) or self.pages_by_title.get(name) or self.pages_by_rel_path.get(key)

    def _build_graph(self, pages: list[Page]) -> nx.DiGraph:
        g = nx.DiGraph()
        for p in pages:
            g.add_node(p.slug, type=p.type.value, title=p.title)

        # wikilink edges — always cheap, never capped
        wikilink_count = 0
        for p in pages:
            for link in p.wikilinks:
                target = self._resolve(link)
                if target and target.slug != p.slug:
                    g.add_edge(p.slug, target.slug, kind="wikilink")
                    wikilink_count += 1

        n_pages = len(pages)
        edge_budget = self.MAX_TOTAL_NON_WIKILINK_EDGES

        # --- shared_tag via inverted index ---
        tag_to_slugs: dict[str, list[str]] = defaultdict(list)
        for p in pages:
            for t in p.tags:
                tag_to_slugs[t].append(p.slug)

        max_doc = max(int(n_pages * self.MAX_TAG_DOC_FRAC), 5)
        pair_counts: Counter[tuple[str, str]] = Counter()
        skipped_noise_tags = 0
        for tag, slugs in tag_to_slugs.items():
            if len(slugs) > max_doc:
                skipped_noise_tags += 1
                continue
            bucket = slugs[: self.MAX_TAG_BUCKET]
            for i, a in enumerate(bucket):
                for b in bucket[i + 1 :]:
                    key = (a, b) if a < b else (b, a)
                    pair_counts[key] += 1

        added_shared_tag = 0
        for (a, b), count in pair_counts.most_common():
            if count < self.MIN_SHARED_TAG:
                break
            if added_shared_tag >= edge_budget:
                break
            g.add_edge(a, b, kind="shared_tag", weight=count)
            g.add_edge(b, a, kind="shared_tag", weight=count)
            added_shared_tag += 2
        edge_budget -= added_shared_tag

        # --- shared_source via inverted index ---
        source_to_slugs: dict[str, list[str]] = defaultdict(list)
        for p in pages:
            for s in p.sources:
                source_to_slugs[s].append(p.slug)

        added_shared_source = 0
        for src, slugs in source_to_slugs.items():
            if len(slugs) > max_doc:
                continue  # too generic
            bucket = slugs[: self.MAX_SOURCE_BUCKET]
            for i, a in enumerate(bucket):
                for b in bucket[i + 1 :]:
                    if added_shared_source >= edge_budget:
                        break
                    if g.has_edge(a, b):
                        continue
                    g.add_edge(a, b, kind="shared_source")
                    g.add_edge(b, a, kind="shared_source")
                    added_shared_source += 2
                if added_shared_source >= edge_budget:
                    break
            if added_shared_source >= edge_budget:
                break

        logger.info(
            f"Graph built: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges "
            f"(wikilink={wikilink_count}, shared_tag={added_shared_tag}, "
            f"shared_source={added_shared_source}, noise_tags_skipped={skipped_noise_tags})"
        )
        return g

    def expand(self, seed_slugs: Iterable[str], k: int = 2) -> list[GraphHit]:
        if self.graph.number_of_nodes() < 10:
            return []
        seeds = list(seed_slugs)
        if not seeds:
            return []
        candidates: Counter[str] = Counter()
        reasons: dict[str, str] = {}
        for seed in seeds:
            if seed not in self.graph:
                continue
            for nbr in self.graph.successors(seed):
                edge = self.graph.get_edge_data(seed, nbr) or {}
                kind = edge.get("kind", "")
                if kind == "wikilink":
                    candidates[nbr] += 2
                    reasons.setdefault(nbr, "wikilink")
                elif kind == "shared_tag" and edge.get("weight", 0) >= self.EXPAND_SHARED_TAG:
                    candidates[nbr] += 1
                    reasons.setdefault(nbr, f"shared_tag={edge.get('weight')}")
        for s in seeds:
            candidates.pop(s, None)
        ranked = sorted(
            candidates.items(),
            key=lambda kv: (kv[1], self.pagerank.get(kv[0], 0.0)),
            reverse=True,
        )
        hits: list[GraphHit] = []
        for slug, votes in ranked[:k]:
            page = self.pages_by_slug.get(slug)
            if page:
                hits.append(GraphHit(page=page, score=votes * 0.03, reason=reasons[slug]))
        return hits

    def to_json(self) -> dict:
        return {
            "nodes": [
                {
                    "slug": n,
                    "title": self.graph.nodes[n].get("title"),
                    "type": self.graph.nodes[n].get("type"),
                    "pagerank": self.pagerank.get(n, 0.0),
                    "community": self.communities.get(n, -1),
                }
                for n in self.graph.nodes
            ],
            "edges": [
                {"src": u, "dst": v, **self.graph.get_edge_data(u, v)}
                for u, v in self.graph.edges
            ],
            "stats": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "communities": len(set(self.communities.values())) if self.communities else 0,
            },
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")

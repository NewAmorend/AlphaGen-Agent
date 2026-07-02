from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class GoldenQuery:
    query: str
    expected: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True)
class QueryEvalResult:
    query: str
    expected: tuple[str, ...]
    ranks: tuple[int, ...]
    hit_at_1: bool
    hit_at_k: bool
    reciprocal_rank: float
    top_hits: tuple[dict[str, Any], ...]
    note: str = ""

    @property
    def best_rank(self) -> int | None:
        return min(self.ranks) if self.ranks else None


@dataclass(frozen=True)
class RetrievalEvalReport:
    top_k: int
    n_queries: int
    hit_at_1: float
    hit_at_k: float
    mrr: float
    source_counts: dict[str, int] = field(default_factory=dict)
    results: tuple[QueryEvalResult, ...] = ()

    @property
    def misses(self) -> tuple[QueryEvalResult, ...]:
        return tuple(r for r in self.results if not r.hit_at_k)

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "n_queries": self.n_queries,
            "hit_at_1": self.hit_at_1,
            "hit_at_k": self.hit_at_k,
            "mrr": self.mrr,
            "source_counts": self.source_counts,
            "results": [
                {
                    "query": r.query,
                    "expected": list(r.expected),
                    "best_rank": r.best_rank,
                    "hit_at_1": r.hit_at_1,
                    "hit_at_k": r.hit_at_k,
                    "reciprocal_rank": r.reciprocal_rank,
                    "top_hits": list(r.top_hits),
                    "note": r.note,
                }
                for r in self.results
            ],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


DEFAULT_GOLDEN_QUERIES: tuple[GoldenQuery, ...] = (
    GoldenQuery(
        query="动量 趋势 延续 ts_delta ts_rank",
        expected=("concepts/momentum", "momentum", "operators/ts_delta", "ts_delta", "operators/ts_rank", "ts_rank"),
        note="momentum concept and time-series momentum operators",
    ),
    GoldenQuery(
        query="反转 均值回归 价格过度反应 pv",
        expected=("concepts/momentum-reversal-pv", "momentum-reversal-pv"),
        note="price-volume reversal concept",
    ),
    GoldenQuery(
        query="质量 因子 profitability gross profits assets qmj",
        expected=("concepts/quality-factors", "quality-factors", "papers/2019-asness-frazzini-pedersen-quality-minus-junk"),
        note="quality factor concept and QMJ paper",
    ),
    GoldenQuery(
        query="流动性 微观结构 Amihud turnover volume",
        expected=("concepts/liquidity-microstructure", "liquidity-microstructure", "papers/2002-amihud-illiquidity"),
        note="liquidity and microstructure knowledge",
    ),
    GoldenQuery(
        query="分析师 预期修正 analyst revisions earnings estimate",
        expected=("concepts/analyst-revisions", "analyst-revisions", "datasets/analyst4", "analyst4"),
        note="analyst revisions concept and dataset",
    ),
    GoldenQuery(
        query="低波动 anomaly volatility risk low vol",
        expected=("concepts/low-volatility-anomaly", "low-volatility-anomaly", "papers/2011-baker-bradley-wurgler-benchmarks-low-vol"),
        note="low-volatility anomaly",
    ),
    GoldenQuery(
        query="新闻 情绪 事件 news sentiment social media",
        expected=("concepts/news-sentiment-events", "news-sentiment-events", "datasets/news12", "news12"),
        note="news and sentiment event signals",
    ),
    GoldenQuery(
        query="换手率 太高 trade_when hump ts_decay_linear",
        expected=("operators/trade_when", "trade_when", "operators/hump", "hump", "operators/ts_decay_linear", "ts_decay_linear"),
        note="turnover control operators",
    ),
)


def load_golden_queries(path: str | Path | None = None) -> tuple[GoldenQuery, ...]:
    if path is None:
        return DEFAULT_GOLDEN_QUERIES
    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or []
    if not isinstance(data, list):
        raise ValueError("Golden query file must be a YAML list")
    out: list[GoldenQuery] = []
    for i, row in enumerate(data, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Golden query #{i} must be a mapping")
        query = str(row.get("query", "")).strip()
        expected = row.get("expected") or []
        if isinstance(expected, str):
            expected = [expected]
        if not query or not expected:
            raise ValueError(f"Golden query #{i} needs query and expected")
        out.append(
            GoldenQuery(
                query=query,
                expected=tuple(_normalize_expected(e) for e in expected if str(e).strip()),
                note=str(row.get("note", "")),
            )
        )
    return tuple(out)


async def evaluate_retriever(
    retriever,
    queries: Iterable[GoldenQuery],
    *,
    top_k: int = 5,
) -> RetrievalEvalReport:
    results: list[QueryEvalResult] = []
    source_counts: dict[str, int] = {}
    queries = tuple(queries)
    for item in queries:
        hits = await retriever.search(item.query, top_k=top_k)
        top_hits: list[dict[str, Any]] = []
        ranks: list[int] = []
        expected = tuple(_normalize_expected(e) for e in item.expected)
        for rank, hit in enumerate(hits, 1):
            page = hit.page
            identifiers = _page_identifiers(page)
            if any(_matches_expected(e, identifiers) for e in expected):
                ranks.append(rank)
            for src in hit.sources:
                source_counts[src] = source_counts.get(src, 0) + 1
            top_hits.append({
                "rank": rank,
                "score": hit.score,
                "title": page.title,
                "slug": page.slug,
                "path": _normalize_expected(str(page.path)),
                "type": page.type.value,
                "sources": list(hit.sources),
            })
        best = min(ranks) if ranks else None
        results.append(
            QueryEvalResult(
                query=item.query,
                expected=expected,
                ranks=tuple(ranks),
                hit_at_1=best == 1,
                hit_at_k=best is not None,
                reciprocal_rank=(1.0 / best) if best else 0.0,
                top_hits=tuple(top_hits),
                note=item.note,
            )
        )
    n = len(results) or 1
    return RetrievalEvalReport(
        top_k=top_k,
        n_queries=len(results),
        hit_at_1=sum(1 for r in results if r.hit_at_1) / n,
        hit_at_k=sum(1 for r in results if r.hit_at_k) / n,
        mrr=sum(r.reciprocal_rank for r in results) / n,
        source_counts=source_counts,
        results=tuple(results),
    )


def _page_identifiers(page) -> set[str]:
    path = _normalize_expected(str(page.path))
    stem = Path(path).stem
    return {
        path,
        path.removesuffix(".md"),
        stem,
        page.slug,
        page.title,
        _normalize_expected(page.slug),
    }


def _matches_expected(expected: str, identifiers: set[str]) -> bool:
    if expected in identifiers:
        return True
    suffix = "/" + expected
    return any(identifier.endswith(suffix) for identifier in identifiers)


def _normalize_expected(value: Any) -> str:
    text = str(value).strip().replace("\\", "/")
    if text.endswith(".md"):
        text = text[:-3]
    return text

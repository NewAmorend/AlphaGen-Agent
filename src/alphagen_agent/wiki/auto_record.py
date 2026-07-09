from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from loguru import logger

from ..models import AlphaRecord, BacktestResult, QualityGrade
from .store import WikiStore


_FUNC_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_FIELD_RE = re.compile(r"\b([a-z][a-z0-9_]{4,})\b")


class AutoRecorder:
    """Backtest 后把结果沉淀回 wiki。"""

    def __init__(self, store: WikiStore, index):
        self.store = store
        self.index = index

    async def record(
        self,
        records: list[AlphaRecord],
        results: list[BacktestResult],
    ) -> dict[str, int]:
        if not records or not results:
            return {"entries": 0, "lessons": 0}

        record_by_id = {r.id: r for r in records if r.id is not None}
        today = date.today().isoformat()
        entries_dir = self.store.root / "entries"
        lessons_dir = self.store.root / "lessons"
        entries_dir.mkdir(parents=True, exist_ok=True)
        lessons_dir.mkdir(parents=True, exist_ok=True)

        entries_written = 0
        rejects: list[tuple[AlphaRecord, BacktestResult]] = []

        for res in results:
            rec = record_by_id.get(res.alpha_id)
            if not rec:
                continue
            if res.grade in (QualityGrade.HIGH, QualityGrade.MEDIUM):
                if self._write_entry(entries_dir, rec, res, today):
                    entries_written += 1
            elif res.grade == QualityGrade.REJECT:
                rejects.append((rec, res))

        lessons_written = self._write_batch_lesson(lessons_dir, rejects, today)

        if entries_written or lessons_written:
            try:
                await self.index.build(incremental=True)
            except Exception as exc:
                logger.warning(f"Wiki incremental rebuild after auto-record failed: {exc}")

        return {"entries": entries_written, "lessons": lessons_written}

    def _write_entry(
        self,
        out_dir: Path,
        rec: AlphaRecord,
        res: BacktestResult,
        today: str,
    ) -> bool:
        path = out_dir / f"{today}-alpha-{rec.id}.md"
        if path.exists():
            return False
        funcs, fields = self._extract_symbols(rec.expression)
        tags = self._infer_tags(funcs, fields, res)
        sources_section = self._render_symbols(funcs, fields)
        body = f"""# Alpha #{rec.id}

**表达式**：

```
{rec.expression}
```

**指标**：fitness={res.fitness}, sharpe={res.sharpe}, turnover={res.turnover}, returns={res.returns}, drawdown={res.drawdown}

**生成策略**：{rec.strategy.value}，LLM={rec.llm_model or 'N/A'}
**评级**：{res.grade.value if res.grade else 'N/A'}
**WQ Alpha ID**：{res.wq_alpha_id or 'N/A'}

## 涉及的算子/字段

{sources_section}

## 为什么 work（猜测）

- TODO 待人工补充：该表达式的因子直觉
"""
        front = self._frontmatter(
            title=f"Alpha #{rec.id} [{(res.grade or QualityGrade.MEDIUM).value}]",
            type_="entry",
            tags=tags,
            extra={
                "alpha_id": rec.id,
                "fitness": _round(res.fitness),
                "sharpe": _round(res.sharpe),
                "turnover": _round(res.turnover),
                "returns": _round(res.returns),
                "created": today,
            },
        )
        path.write_text(front + body, encoding="utf-8")
        logger.info(f"Wrote wiki entry {path.name}")
        return True

    def _write_batch_lesson(
        self,
        out_dir: Path,
        rejects: list[tuple[AlphaRecord, BacktestResult]],
        today: str,
    ) -> int:
        if not rejects:
            return 0
        buckets: dict[str, list[tuple[AlphaRecord, BacktestResult]]] = defaultdict(list)
        for rec, res in rejects:
            buckets[self._classify_reject(res)].append((rec, res))

        seq = 1
        while (out_dir / f"{today}-batch-{seq}.md").exists():
            seq += 1
        path = out_dir / f"{today}-batch-{seq}.md"

        func_counter: Counter[str] = Counter()
        field_counter: Counter[str] = Counter()
        sections: list[str] = []
        for reason, items in sorted(buckets.items()):
            sections.append(f"## {reason}（{len(items)} 个）\n")
            for rec, res in items[:8]:
                funcs, fields = self._extract_symbols(rec.expression)
                func_counter.update(funcs)
                field_counter.update(fields)
                sections.append(
                    f"- `{rec.expression}` — fitness={res.fitness}, sharpe={res.sharpe}, turnover={res.turnover}"
                )
            if len(items) > 8:
                sections.append(f"- … 另有 {len(items) - 8} 个同类\n")
            sections.append("")

        top_funcs = ", ".join(f for f, _ in func_counter.most_common(8)) or "无"
        top_fields = ", ".join(f for f, _ in field_counter.most_common(8)) or "无"
        body = f"""# 批次失败教训 — {today} #{seq}

共 {len(rejects)} 个被拒，按失败模式聚类如下。

{chr(10).join(sections)}

## 集中出现的算子

{top_funcs}

## 集中出现的字段

{top_fields}

## 建议（待人工归纳）

- TODO：把以上模式整理进 concepts/ 或 operators/ 下的对应页
"""
        front = self._frontmatter(
            title=f"Batch lessons {today} #{seq}",
            type_="lesson",
            tags=sorted({"reject", *(self._classify_reject(res).replace(" ", "_") for _, res in rejects)})[:8],
            extra={"created": today, "count": len(rejects)},
        )
        path.write_text(front + body, encoding="utf-8")
        logger.info(f"Wrote wiki lesson {path.name} ({len(rejects)} rejects)")
        return 1

    @staticmethod
    def _classify_reject(res: BacktestResult) -> str:
        if res.fitness is None:
            return "missing metrics"
        if res.fitness < 0:
            return "negative fitness（信号方向反了？）"
        if res.fitness < 0.1:
            return "fitness 接近 0（无信号）"
        if res.turnover is not None and res.turnover > 0.7:
            return "turnover 超阈"
        if res.sharpe is not None and res.sharpe < 0.5:
            return "sharpe 偏低"
        return "其他不达标"

    @staticmethod
    def _extract_symbols(expr: str) -> tuple[list[str], list[str]]:
        funcs = sorted({m.group(1) for m in _FUNC_RE.finditer(expr)})
        non_funcs = _FUNC_RE.sub(" ", expr)
        fields = sorted({m.group(1) for m in _FIELD_RE.finditer(non_funcs) if m.group(1) not in funcs})
        return funcs, fields

    @staticmethod
    def _infer_tags(funcs: list[str], fields: list[str], res: BacktestResult) -> list[str]:
        tags = ["entry"]
        if res.grade:
            tags.append(res.grade.value)
        tags.extend(funcs[:4])
        tags.extend(fields[:4])
        return tags

    @staticmethod
    def _render_symbols(funcs: list[str], fields: list[str]) -> str:
        parts: list[str] = []
        parts.extend(f"[[{s}]]" for s in funcs[:8])
        parts.extend(f"`{s}`" for s in fields[:8])
        return ", ".join(parts) if parts else "（解析为空）"

    @staticmethod
    def _frontmatter(title: str, type_: str, tags: list[str], extra: dict) -> str:
        import yaml
        data = {"title": title, "type": type_, "tags": tags, **extra}
        return "---\n" + yaml.safe_dump(data, sort_keys=False, allow_unicode=True) + "---\n\n"


def _round(x):
    if x is None:
        return None
    try:
        return round(float(x), 4)
    except (TypeError, ValueError):
        return None

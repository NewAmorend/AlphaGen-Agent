from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from alphagen_agent.db import Database
from alphagen_agent.generator.refine import RefineAlphaGenerator
from alphagen_agent.models import (
    AlphaRecord,
    BacktestResult,
    GenerationStrategy,
    QualityGrade,
    WQDataField,
    WQOperator,
)


# ----- db: candidate selection -----

@pytest.mark.asyncio
async def test_list_refine_candidates_picks_only_medium(tmp_path):
    db = Database(str(tmp_path / "wq.db"))
    await db.connect()
    try:
        # 3 alphas: high / medium / reject — 应只取 medium
        for grade, fitness in [
            (QualityGrade.HIGH, 1.5),
            (QualityGrade.MEDIUM, 0.83),
            (QualityGrade.REJECT, -0.1),
        ]:
            aid = await db.insert_alpha(AlphaRecord(
                expression=f"expr_{grade.value}",
                strategy=GenerationStrategy.LLM,
                created_at=datetime.now(),
            ))
            await db.insert_backtest_result(BacktestResult(
                alpha_id=aid, fitness=fitness, sharpe=1.5, turnover=0.5,
                grade=grade,
                checks=[{"name": "LOW_FITNESS", "result": "FAIL" if grade != QualityGrade.HIGH else "PASS"}],
            ))

        rows = await db.list_refine_candidates(limit=10)
        assert len(rows) == 1
        assert rows[0]["grade"] == "medium"
        assert rows[0]["fitness"] == 0.83
        assert rows[0]["failed_checks"] == ["LOW_FITNESS"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_list_refine_candidates_sorts_by_fitness_desc(tmp_path):
    db = Database(str(tmp_path / "wq.db"))
    await db.connect()
    try:
        for fit in [0.71, 0.95, 0.83]:
            aid = await db.insert_alpha(AlphaRecord(
                expression=f"e{fit}", strategy=GenerationStrategy.LLM, created_at=datetime.now(),
            ))
            await db.insert_backtest_result(BacktestResult(
                alpha_id=aid, fitness=fit, sharpe=1.5, turnover=0.5,
                grade=QualityGrade.MEDIUM,
                checks=[{"name": "LOW_FITNESS", "result": "FAIL"}],
            ))

        rows = await db.list_refine_candidates(limit=10)
        fitnesses = [r["fitness"] for r in rows]
        assert fitnesses == sorted(fitnesses, reverse=True)
    finally:
        await db.close()


# ----- refine generator -----

@pytest.mark.asyncio
async def test_refine_generator_prompt_contains_base_and_hints():
    captured = {}

    async def _capture(prompt, **kw):
        captured["prompt"] = prompt
        # 返回 5 个合法但不同的微调变体
        return "\n".join([
            "group_neutralize(rank(ts_delta(close, 60)), subindustry)",
            "ts_decay_linear(rank(ts_delta(close, 60)), 20)",
            "winsorize(rank(ts_delta(close, 60)), 4)",
            "rank(ts_delta(close, 120))",
            "rank(ts_delta(close, 252))",
        ])

    llm = AsyncMock()
    llm.generate = _capture

    refiner = RefineAlphaGenerator(llm=llm)
    base = {
        "alpha_id": 99,
        "expression": "rank(ts_delta(close, 60))",
        "fitness": 0.83,
        "sharpe": 1.54,
        "turnover": 0.6,
        "returns": 0.05,
        "failed_checks": ["LOW_FITNESS"],
    }
    fields = [WQDataField(id="close", description="Close price")]
    ops = [WQOperator(name="ts_delta", description="Time delta"),
           WQOperator(name="group_neutralize", description="Group neutralize")]
    variants = await refiner.refine(base, fields, ops, count=5)

    p = captured["prompt"]
    # 基础表达式 + 失败检查 + 针对 LOW_FITNESS 的 hint 都在
    assert "rank(ts_delta(close, 60))" in p
    assert "LOW_FITNESS" in p
    assert "add(rank" in p or "复合" in p   # hint for LOW_FITNESS
    assert "0.830" in p                       # formatted fitness
    assert len(variants) == 5


@pytest.mark.asyncio
async def test_refine_generator_dedupes_base_verbatim():
    async def _gen(prompt, **kw):
        return "\n".join([
            "rank(ts_delta(close, 60))",                # 与 base 完全相同
            "rank(ts_delta(close, 120))",
            "group_neutralize(rank(ts_delta(close, 60)), sector)",
        ])
    llm = AsyncMock()
    llm.generate = _gen

    refiner = RefineAlphaGenerator(llm=llm)
    base = {
        "alpha_id": 1, "expression": "rank(ts_delta(close, 60))",
        "fitness": 0.83, "failed_checks": ["LOW_FITNESS"],
    }
    variants = await refiner.refine(base, [WQDataField(id="close")], [], count=10)
    assert "rank(ts_delta(close, 60))" not in variants    # base 被去掉
    assert any("subindustry" in v or "sector" in v or "120" in v for v in variants)


@pytest.mark.asyncio
async def test_refine_generator_handles_empty_failed_checks():
    async def _gen(prompt, **kw):
        return "rank(volume)"
    llm = AsyncMock()
    llm.generate = _gen
    refiner = RefineAlphaGenerator(llm=llm)
    base = {"alpha_id": 1, "expression": "ts_delta(close, 5)", "fitness": 0.95, "failed_checks": []}
    variants = await refiner.refine(
        base, [WQDataField(id="close"), WQDataField(id="volume")], [], count=3
    )
    assert len(variants) == 1

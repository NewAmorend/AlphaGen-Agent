from __future__ import annotations

from datetime import datetime

import pytest

from alphagen_agent.db import Database
from alphagen_agent.generator.llm import LLMAlphaGenerator
from alphagen_agent.models import AlphaRecord, BacktestResult, GenerationStrategy


# --------------------------------------------------------------------------- #1
# 批内去重：同一次 generate 产出里，同骨架（换字段/窗口）的表达式只保留第一个。
def test_dedup_by_skeleton_collapses_same_structure():
    exprs = [
        "ts_decay_linear(rank(fnd6_assets), 20)",
        "ts_decay_linear(rank(mdl177_x), 60)",            # 同骨架（FIELD/N）→ 丢
        "group_neutralize(rank(close), subindustry)",     # 不同结构 → 留
        "ts_decay_linear(rank(fnd6_eps), 5)",             # 又一个同骨架 → 丢
    ]
    out = LLMAlphaGenerator._dedup_by_skeleton(exprs)
    assert out == [
        "ts_decay_linear(rank(fnd6_assets), 20)",
        "group_neutralize(rank(close), subindustry)",
    ]


def test_dedup_by_skeleton_keeps_unique():
    exprs = [
        "rank(ts_delta(close, 5))",
        "zscore(ts_mean(volume, 20))",
    ]
    assert LLMAlphaGenerator._dedup_by_skeleton(exprs) == exprs


# --------------------------------------------------------------------------- #2
# 历史低分骨架排除：同骨架的历史最佳 fitness 始终 < 阈值 → 进排除集；
# 同骨架只要有过一次 ≥ 阈值（有潜力）→ 不排除，值得换字段/窗口重试。
@pytest.mark.asyncio
async def test_get_low_fitness_skeletons(tmp_path):
    db = Database(str(tmp_path / "wq.db"))
    await db.connect()
    try:
        async def _add(expr: str, fitness: float) -> int:
            aid = await db.insert_alpha(
                AlphaRecord(
                    expression=expr,
                    strategy=GenerationStrategy.LLM,
                    created_at=datetime.now(),
                )
            )
            await db.insert_backtest_result(
                BacktestResult(alpha_id=aid, fitness=fitness, created_at=datetime.now())
            )
            return aid

        # 骨架 A：两个低分实例 → 历史最佳 0.15 < 0.3 → 排除
        await _add("rank(ts_delta(fnd6_assets, 20))", 0.10)
        await _add("rank(ts_delta(mdl177_x, 60))", 0.15)
        # 骨架 B：一次低分一次高分 → 历史最佳 0.9 ≥ 0.3 → 不排除
        await _add("ts_decay_linear(rank(close), 5)", 0.05)
        await _add("ts_decay_linear(rank(volume), 20)", 0.90)

        low = await db.get_low_fitness_skeletons(max_fitness=0.3)

        from alphagen_agent.db import expression_skeleton
        skel_a = expression_skeleton("rank(ts_delta(fnd6_assets, 20))")
        skel_b = expression_skeleton("ts_decay_linear(rank(close), 5)")
        assert skel_a in low
        assert skel_b not in low
    finally:
        await db.close()


# --------------------------------------------------------------------------- #6
# 重复度可观测：按 outer-2 wrapper 家族聚合，看库里结构集中度。
@pytest.mark.asyncio
async def test_get_skeleton_distribution(tmp_path):
    db = Database(str(tmp_path / "wq.db"))
    await db.connect()
    try:
        async def _add(expr: str, fitness: float) -> None:
            aid = await db.insert_alpha(
                AlphaRecord(
                    expression=expr,
                    strategy=GenerationStrategy.LLM,
                    created_at=datetime.now(),
                )
            )
            await db.insert_backtest_result(
                BacktestResult(alpha_id=aid, fitness=fitness, created_at=datetime.now())
            )

        # ts_decay_linear(rank(...)) 家族 3 个；group_neutralize(rank(...)) 1 个
        await _add("ts_decay_linear(rank(fnd6_assets), 20)", 0.4)
        await _add("ts_decay_linear(rank(close), 5)", 0.8)
        await _add("ts_decay_linear(rank(volume), 60)", 0.6)
        await _add("group_neutralize(rank(close), subindustry)", 0.3)

        dist = await db.get_skeleton_distribution(limit=10)
        assert dist["total_backtested"] == 4
        assert dist["unique_outer2"] == 2
        top = dist["top_outer2"][0]
        assert top["count"] == 3
        assert top["max_fitness"] == pytest.approx(0.8)
    finally:
        await db.close()

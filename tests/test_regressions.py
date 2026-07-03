from __future__ import annotations

from datetime import datetime

import pytest

from alphagen_agent.config import Settings
from alphagen_agent.db import Database
from alphagen_agent.engine.backtest import BacktestEngine
from alphagen_agent.generator.factor import FactorMiningGenerator
from alphagen_agent.generator.template import TemplateAlphaGenerator
from alphagen_agent.models import (
    AlphaRecord,
    AlphaStatus,
    BacktestResult,
    GenerationStrategy,
    QualityGrade,
    WQDataField,
)


@pytest.mark.asyncio
async def test_non_llm_generators_accept_orchestrator_context_kwargs():
    fields = [WQDataField(id="fnd6_assets"), WQDataField(id="mdl177_x")]

    for generator in (TemplateAlphaGenerator(), FactorMiningGenerator()):
        expressions = await generator.generate(
            fields,
            [],
            count=1,
            extra_exclude_skeletons=set(),
            family_distribution={"total_backtested": 0},
        )
        assert len(expressions) == 1


@pytest.mark.asyncio
async def test_high_backtest_status_stays_high_quality(tmp_path, monkeypatch):
    db = Database(str(tmp_path / "wq.db"))
    await db.connect()
    try:
        alpha_id = await db.insert_alpha(
            AlphaRecord(expression="rank(fnd6_assets)", strategy=GenerationStrategy.LLM)
        )
        alpha = await db.get_alpha(alpha_id)
        assert alpha is not None

        engine = BacktestEngine(wq=object(), db=db, settings=Settings(_env_file=None))

        async def _fake_backtest_expression(aid: int, expression: str) -> BacktestResult:
            return BacktestResult(
                alpha_id=aid,
                fitness=2.0,
                sharpe=1.5,
                turnover=0.3,
                grade=QualityGrade.HIGH,
            )

        monkeypatch.setattr(engine, "_backtest_expression", _fake_backtest_expression)
        await engine._backtest_single(alpha)

        updated = await db.get_alpha(alpha_id)
        assert updated is not None
        assert updated.status is AlphaStatus.HIGH_QUALITY
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_user_facing_backtest_queries_use_latest_result(tmp_path):
    db = Database(str(tmp_path / "wq.db"))
    await db.connect()
    try:
        alpha_id = await db.insert_alpha(
            AlphaRecord(expression="rank(fnd6_assets)", strategy=GenerationStrategy.LLM)
        )
        await db.insert_backtest_result(
            BacktestResult(
                alpha_id=alpha_id,
                fitness=2.0,
                sharpe=1.6,
                turnover=0.3,
                grade=QualityGrade.HIGH,
                created_at=datetime(2026, 1, 1, 9, 0, 0),
            )
        )
        await db.insert_backtest_result(
            BacktestResult(
                alpha_id=alpha_id,
                fitness=0.2,
                sharpe=0.4,
                turnover=0.9,
                grade=QualityGrade.REJECT,
                created_at=datetime(2026, 1, 1, 10, 0, 0),
            )
        )

        assert await db.list_high_quality_alphas(min_fitness=1.0) == []
        assert await db.list_submittable_alphas(min_fitness=1.0) == []
        assert await db.list_refine_candidates(limit=10) == []
        stats = await db.get_stats()
        assert stats["high_quality_count"] == 0
    finally:
        await db.close()

from __future__ import annotations

from datetime import datetime

import pytest

from wq_agent.db import Database
from wq_agent.generator.llm import LLMAlphaGenerator
from wq_agent.models import (
    AlphaRecord,
    AlphaStatus,
    BacktestResult,
    GenerationStrategy,
    QualityGrade,
)


@pytest.mark.asyncio
async def test_list_recent_backtested_alphas_round_trip(tmp_path):
    db = Database(str(tmp_path / "wq.db"))
    await db.connect()
    try:
        a = AlphaRecord(
            expression="rank(ts_delta(close, 60))",
            strategy=GenerationStrategy.LLM,
            status=AlphaStatus.GENERATED,
            created_at=datetime.now(),
        )
        aid = await db.insert_alpha(a)
        await db.insert_backtest_result(BacktestResult(
            alpha_id=aid, fitness=0.83, sharpe=1.54, turnover=0.6,
            grade=QualityGrade.MEDIUM,
            checks=[
                {"name": "LOW_SHARPE", "result": "PASS"},
                {"name": "LOW_FITNESS", "result": "FAIL", "value": 0.83, "limit": 1.0},
            ],
        ))

        rows = await db.list_recent_backtested_alphas(limit=5)
        assert len(rows) == 1
        r = rows[0]
        assert r["alpha"] == "rank(ts_delta(close, 60))"
        assert r["performance"]["fitness"] == 0.83
        assert r["performance"]["grade"] == "medium"
        assert r["failed_checks"] == ["LOW_FITNESS"]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_list_recent_excludes_alphas_without_backtest(tmp_path):
    db = Database(str(tmp_path / "wq.db"))
    await db.connect()
    try:
        await db.insert_alpha(AlphaRecord(
            expression="rank(volume)",
            strategy=GenerationStrategy.LLM,
            created_at=datetime.now(),
        ))
        rows = await db.list_recent_backtested_alphas()
        assert rows == []
    finally:
        await db.close()


def test_previous_section_empty_when_none():
    assert LLMAlphaGenerator._build_previous_results_section(None) == ""
    assert LLMAlphaGenerator._build_previous_results_section([]) == ""


def test_previous_section_triggers_sign_flip_advice():
    rows = [{
        "alpha": "rank(ts_delta(close, 60))",
        "performance": {"fitness": -0.7, "sharpe": -1.0, "turnover": 0.4, "grade": "reject"},
        "failed_checks": ["LOW_FITNESS", "LOW_SHARPE"],
    }]
    section = LLMAlphaGenerator._build_previous_results_section(rows)
    assert "reverse(" in section          # 信号反转建议
    assert "rank(ts_delta(close, 60))" in section
    assert "LOW_FITNESS" in section


def test_previous_section_triggers_near_miss_refine_advice():
    rows = [{
        "alpha": "rank(volume)",
        "performance": {"fitness": 0.83, "sharpe": 1.54, "turnover": 0.6, "grade": "medium"},
        "failed_checks": ["LOW_FITNESS"],
    }]
    section = LLMAlphaGenerator._build_previous_results_section(rows)
    assert "MEDIUM" in section
    assert "微调变体" in section          # 推到 HIGH 的建议
    assert "group_neutralize" in section


def test_previous_section_aggregates_failure_modes():
    rows = [
        {"alpha": f"expr{i}", "performance": {"fitness": 0.05, "sharpe": 0.2, "turnover": 0.4,
                                                "grade": "reject"},
         "failed_checks": ["LOW_FITNESS"]}
        for i in range(4)
    ]
    section = LLMAlphaGenerator._build_previous_results_section(rows)
    assert "复合" in section              # 推荐 add(rank(A), rank(B))
    assert "add(rank" in section

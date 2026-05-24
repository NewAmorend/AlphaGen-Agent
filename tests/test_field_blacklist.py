from __future__ import annotations

import pytest

from wq_agent.db import Database
from wq_agent.engine.backtest import extract_field_candidates
from wq_agent.generator.llm import LLMAlphaGenerator


def test_extract_field_candidates_skips_operators_and_keywords():
    expr = "group_neutralize(rank(ts_delta(fnd6_assets, 60)), subindustry)"
    fields = extract_field_candidates(expr)
    # fnd6_assets 是字段；其它都是算子或 group key
    assert "fnd6_assets" in fields
    assert "rank" not in fields
    assert "ts_delta" not in fields
    assert "group_neutralize" not in fields
    assert "subindustry" not in fields


def test_extract_field_candidates_skips_operator_kwargs():
    """Regression: 之前 quantile(_, driver=gaussian, sigma=1.0)、winsorize(_, std=4)
    会把 driver/gaussian/sigma/std 误识别为字段，污染 blacklist。"""
    expr = "quantile(rank(fnd6_eps), driver=gaussian, sigma=1.0)"
    fields = extract_field_candidates(expr)
    assert "fnd6_eps" in fields
    assert "driver" not in fields
    assert "gaussian" not in fields
    assert "sigma" not in fields

    expr2 = "winsorize(rank(close), std=4)"
    fields2 = extract_field_candidates(expr2)
    assert "close" in fields2
    assert "std" not in fields2


def test_extract_field_candidates_multi_field():
    expr = "add(rank(divide(fnd6_cashflow_op, fnd6_assets)), rank(ts_delta(close, 60)))"
    fields = extract_field_candidates(expr)
    assert {"fnd6_cashflow_op", "fnd6_assets", "close"} <= set(fields)
    # 不应该把数字 / 算子混进去
    assert all(not f.isdigit() for f in fields)


def test_nesting_depth_counts_max_open_parens():
    assert LLMAlphaGenerator._nesting_depth("rank(x)") == 1
    assert LLMAlphaGenerator._nesting_depth("rank(ts_delta(x, 60))") == 2
    assert LLMAlphaGenerator._nesting_depth(
        "quantile(normalize(ts_decay_linear(ts_rank(ts_delta(x, 60)), 20)))"
    ) == 5


def test_clean_expressions_drops_overly_nested():
    from wq_agent.llm.base import BaseLLMProvider
    class _DummyLLM(BaseLLMProvider):
        async def generate(self, *a, **k): return ""
        async def close(self): pass

    gen = LLMAlphaGenerator(_DummyLLM())
    ideas = [
        "rank(ts_delta(close, 60))",                                              # depth 2 ✓
        "group_neutralize(rank(ts_delta(close, 60)), subindustry)",               # depth 3 ✓
        "quantile(normalize(ts_decay_linear(ts_rank(ts_delta(close, 60)), 20)))", # depth 5 ✗
    ]
    cleaned = gen._clean_expressions(ideas)
    # 前两个保留，第三个被剔
    assert "rank(ts_delta(close, 60))" in cleaned
    assert "group_neutralize(rank(ts_delta(close, 60)), subindustry)" in cleaned
    assert not any("decay_linear(ts_rank" in e for e in cleaned)


@pytest.mark.asyncio
async def test_blacklist_round_trip(tmp_path):
    db = Database(str(tmp_path / "wq.db"))
    await db.connect()
    try:
        await db.bump_field_blacklist(["fnd6_eventv110_x", "fnd6_eventv110_y"], reason="sim_error")
        await db.bump_field_blacklist(["fnd6_eventv110_x"], reason="sim_error")
        await db.bump_field_blacklist(["fnd6_eventv110_x"], reason="sim_error")
        # x 失败 3 次 → 进 blacklist；y 失败 1 次 → 不进
        bl = await db.get_blacklisted_fields(min_fail_count=3)
        assert bl == {"fnd6_eventv110_x"}
        rows = await db.list_field_blacklist()
        x_row = next(r for r in rows if r["field_id"] == "fnd6_eventv110_x")
        assert x_row["fail_count"] == 3
        n = await db.clear_field_blacklist()
        assert n == 2
        assert (await db.get_blacklisted_fields(min_fail_count=1)) == set()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_reset_stuck_backtesting(tmp_path):
    from datetime import datetime
    from wq_agent.models import AlphaRecord, AlphaStatus, GenerationStrategy
    db = Database(str(tmp_path / "wq.db"))
    await db.connect()
    try:
        aid = await db.insert_alpha(AlphaRecord(
            expression="rank(close)",
            strategy=GenerationStrategy.LLM,
            status=AlphaStatus.GENERATED,
            created_at=datetime.now(),
        ))
        await db.update_alpha_status(aid, AlphaStatus.BACKTESTING)
        n = await db.reset_stuck_backtesting()
        assert n == 1
        alpha = await db.get_alpha(aid)
        assert alpha.status is AlphaStatus.GENERATED
    finally:
        await db.close()

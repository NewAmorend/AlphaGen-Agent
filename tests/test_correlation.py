from __future__ import annotations
from datetime import datetime
from wq_agent.config import Settings
import pytest
from wq_agent.db import Database
from wq_agent.models import AlphaRecord, BacktestResult, GenerationStrategy, AlphaStatus, QualityGrade


def test_self_corr_settings_defaults():
    s = Settings(_env_file=None)
    assert s.SELF_CORR_THRESHOLD == 0.7
    assert s.SELF_CORR_SHARPE_MARGIN == 0.10
    assert s.SELF_CORR_MIN_OVERLAP == 60


async def _seed_alpha(db, expr, grade, sharpe, wq_id, status=AlphaStatus.GENERATED):
    aid = await db.insert_alpha(AlphaRecord(expression=expr, strategy=GenerationStrategy.LLM,
                                            status=status, created_at=datetime.now()))
    await db.insert_backtest_result(BacktestResult(
        alpha_id=aid, sharpe=sharpe, fitness=1.2, grade=grade,
        wq_alpha_id=wq_id, created_at=datetime.now()))
    return aid


@pytest.mark.asyncio
async def test_pnl_cache_round_trip(tmp_path):
    db = Database(str(tmp_path / "wq.db"))
    await db.connect()
    try:
        assert await db.get_cached_pnl(1) is None
        await db.upsert_pnl(1, "WQ123", ["2020-01-02", "2020-01-03"], [0.1, -0.2])
        got = await db.get_cached_pnl(1)
        assert got == (["2020-01-02", "2020-01-03"], [0.1, -0.2])
        # upsert overwrites
        await db.upsert_pnl(1, "WQ123", ["2020-01-02"], [0.5])
        assert await db.get_cached_pnl(1) == (["2020-01-02"], [0.5])
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_list_reference_alphas(tmp_path):
    db = Database(str(tmp_path / "wq.db"))
    await db.connect()
    try:
        sub = await _seed_alpha(db, "rank(close)", QualityGrade.HIGH, 1.6, "WQSUB",
                                status=AlphaStatus.SUBMITTED)
        hi = await _seed_alpha(db, "rank(open)", QualityGrade.HIGH, 1.4, "WQHI")
        await _seed_alpha(db, "rank(low)", QualityGrade.REJECT, 0.1, "WQREJ")  # excluded
        ref = await db.list_reference_alphas()
        assert [r["alpha_id"] for r in ref["submitted"]] == [sub]
        assert ref["submitted"][0]["sharpe"] == 1.6
        assert ref["submitted"][0]["wq_alpha_id"] == "WQSUB"
        assert [r["alpha_id"] for r in ref["high"]] == [hi]   # HIGH but not submitted
    finally:
        await db.close()

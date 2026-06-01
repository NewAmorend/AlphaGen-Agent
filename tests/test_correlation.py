from __future__ import annotations

from datetime import datetime

import pytest

from wq_agent.config import Settings
from wq_agent.db import Database
from wq_agent.engine.correlation import (
    CorrelationScreener,
    align,
    hard_gate,
    is_hard_redundant,
    max_correlation,
    parse_pnl_response,
    pearson,
)
from wq_agent.models import (
    AlphaRecord,
    AlphaStatus,
    BacktestResult,
    GenerationStrategy,
    QualityGrade,
)


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


def test_parse_pnl_response_diffs_to_daily_returns():
    data = {"records": [["2020-01-02", 0.0], ["2020-01-03", 1.5], ["2020-01-06", 1.0]]}
    dates, returns = parse_pnl_response(data)
    assert dates == ["2020-01-03", "2020-01-06"]
    assert returns == [1.5, -0.5]


@pytest.mark.asyncio
async def test_get_pnl_retries_on_empty_200_then_parses(monkeypatch):
    """WQ 突发节流会返回 200+空 body；get_pnl 应退避重试而非当成永久失败。"""
    import wq_agent.wq.client as clientmod
    from wq_agent.wq.client import WQClient

    wq = WQClient(Settings(_env_file=None))

    class _Resp:
        def __init__(self, status, text):
            self.status_code = status
            self.text = text

        def json(self):
            import json
            return json.loads(self.text)

    seq = [_Resp(200, ""), _Resp(200, "   "),
           _Resp(200, '{"records":[["d1",0.0],["d2",3.0],["d3",1.0]]}')]
    calls = {"n": 0}

    async def fake_request(method, path, **k):
        r = seq[min(calls["n"], len(seq) - 1)]
        calls["n"] += 1
        return r

    async def nosleep(*a, **k):
        pass

    monkeypatch.setattr(wq, "_request", fake_request)
    monkeypatch.setattr(clientmod.asyncio, "sleep", nosleep)
    out = await wq.get_pnl("X")
    assert out == (["d2", "d3"], [3.0, -2.0])   # diffed daily returns
    assert calls["n"] == 3                       # two empties + one good
    await wq.close()


def test_parse_pnl_response_skips_malformed():
    data = {"records": [["2020-01-02", 0.0], ["2020-01-03", None], "junk", ["2020-01-06", 2.0]]}
    dates, returns = parse_pnl_response(data)
    assert dates == ["2020-01-06"]
    assert returns == [2.0]


def test_pearson_basic():
    assert pearson([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert pearson([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert pearson([1, 1, 1], [1, 2, 3]) == 0.0          # zero variance -> 0
    assert pearson([1.0], [2.0]) == 0.0                  # too short -> 0


def test_align_by_date_overlap():
    a_d, a_r = ["d1", "d2", "d3"], [1.0, 2.0, 3.0]
    b_d, b_r = ["d2", "d3", "d4"], [9.0, 8.0, 7.0]
    va, vb = align(a_d, a_r, b_d, b_r)
    assert va == [2.0, 3.0]   # only d2, d3 overlap, sorted by date
    assert vb == [9.0, 8.0]


def test_max_correlation_picks_strongest_with_enough_overlap():
    cand_d = ["d1", "d2", "d3", "d4"]
    cand_r = [1.0, 2.0, 3.0, 4.0]
    refs = [
        {"alpha_id": 10, "sharpe": 1.5, "dates": ["d1", "d2", "d3", "d4"], "returns": [4, 3, 2, 1]},  # corr -1
        {"alpha_id": 11, "sharpe": 1.6, "dates": ["d1", "d2", "d3", "d4"], "returns": [1, 2, 3, 4]},  # corr +1
        {"alpha_id": 12, "sharpe": 9.9, "dates": ["dX"], "returns": [1.0]},                            # no overlap
    ]
    corr, ref_id, ref_sharpe = max_correlation(cand_d, cand_r, refs, min_overlap=3)
    assert ref_id == 11
    assert corr == pytest.approx(1.0)
    assert ref_sharpe == 1.6


def test_max_correlation_skips_insufficient_overlap():
    refs = [{"alpha_id": 10, "sharpe": 1.5, "dates": ["d1", "d2"], "returns": [1, 2]}]
    corr, ref_id, ref_sharpe = max_correlation(["d1", "d2"], [1, 2], refs, min_overlap=3)
    assert ref_id is None and corr == 0.0 and ref_sharpe is None


def test_is_hard_redundant_rule():
    # corr above threshold AND sharpe not >10% better -> redundant
    assert is_hard_redundant(cand_sharpe=1.30, max_corr=0.93, ref_sharpe=1.40,
                             threshold=0.7, margin=0.10) is True
    # corr above threshold BUT sharpe >10% better -> NOT redundant (WQ accepts)
    assert is_hard_redundant(cand_sharpe=1.60, max_corr=0.93, ref_sharpe=1.40,
                             threshold=0.7, margin=0.10) is False
    # corr below threshold -> NOT redundant
    assert is_hard_redundant(cand_sharpe=1.30, max_corr=0.50, ref_sharpe=1.40,
                             threshold=0.7, margin=0.10) is False
    # no ref -> NOT redundant
    assert is_hard_redundant(cand_sharpe=1.30, max_corr=0.0, ref_sharpe=None,
                             threshold=0.7, margin=0.10) is False
    # missing candidate sharpe -> fail-open, NOT redundant
    assert is_hard_redundant(cand_sharpe=None, max_corr=0.93, ref_sharpe=1.40,
                             threshold=0.7, margin=0.10) is False


def test_hard_gate_blocks_against_any_ref():
    # Candidate beats the MAX-corr ref by >10% (would be exempt vs that one alone),
    # but is also correlated with a lower-sharpe ref it does NOT beat -> must block.
    cand_d = ["d1", "d2", "d3", "d4"]
    cand_r = [1.0, 2.0, 3.0, 4.0]
    refs = [
        {"alpha_id": 21, "sharpe": 0.8, "dates": cand_d, "returns": [1.0, 2.0, 3.0, 4.0]},  # corr +1.0
        {"alpha_id": 22, "sharpe": 5.0, "dates": cand_d, "returns": [1.1, 2.0, 3.0, 4.2]},  # ~corr +1, higher sharpe
    ]
    # cand_sharpe 1.5 beats ref 21 (0.8) by >10% but not ref 22 (5.0)
    blocked, corr, ref_id = hard_gate(1.5, cand_d, cand_r, refs, min_overlap=3,
                                      threshold=0.7, margin=0.10)
    assert blocked is True
    # blocking ref is one it failed to beat (22); reported as the strongest-corr blocker
    assert ref_id == 22


class _FakeWQ:
    def __init__(self, pnl_by_id):
        self.pnl_by_id = pnl_by_id
        self.calls = 0

    async def get_pnl(self, wq_alpha_id):
        self.calls += 1
        return self.pnl_by_id.get(wq_alpha_id)


@pytest.mark.asyncio
async def test_ensure_pnl_lazy_and_cached(tmp_path):
    db = Database(str(tmp_path / "wq.db"))
    await db.connect()
    try:
        wq = _FakeWQ({"WQ1": (["d1", "d2"], [0.1, 0.2])})
        scr = CorrelationScreener(db, wq, Settings(_env_file=None))
        # first call fetches + caches
        got = await scr.ensure_pnl(1, "WQ1")
        assert got == (["d1", "d2"], [0.1, 0.2])
        assert wq.calls == 1
        # second call uses cache (no new fetch)
        got2 = await scr.ensure_pnl(1, "WQ1")
        assert got2 == (["d1", "d2"], [0.1, 0.2])
        assert wq.calls == 1
        # fetch failure -> None, not cached
        none = await scr.ensure_pnl(2, "MISSING")
        assert none is None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_screen_marks_hard_redundant(tmp_path):
    db = Database(str(tmp_path / "wq.db"))
    await db.connect()
    try:
        # submitted ref: sharpe 1.6, a known PnL
        sub = await _seed_alpha(db, "rank(close)", QualityGrade.HIGH, 1.6, "WQSUB",
                                status=AlphaStatus.SUBMITTED)
        dates = [f"d{i}" for i in range(100)]
        rets = [float(i % 7 - 3) for i in range(100)]
        await db.upsert_pnl(sub, "WQSUB", dates, rets)
        # candidate A: near-identical PnL, sharpe 1.3 (not >10% better) -> redundant
        ca = await _seed_alpha(db, "rank(vwap)", QualityGrade.HIGH, 1.3, "WQA")
        await db.upsert_pnl(ca, "WQA", dates, rets)
        # candidate B: uncorrelated PnL -> not redundant
        cb = await _seed_alpha(db, "rank(open)", QualityGrade.HIGH, 1.3, "WQB")
        await db.upsert_pnl(cb, "WQB", dates, [float((i * 13) % 5 - 2) for i in range(100)])
        # candidate C: identical PnL to ref BUT sharpe 2.0 (>10% better than 1.6) -> WQ would accept
        cc = await _seed_alpha(db, "rank(high)", QualityGrade.HIGH, 2.0, "WQC")
        await db.upsert_pnl(cc, "WQC", dates, rets)

        wq = _FakeWQ({})  # all cached, no fetch needed
        scr = CorrelationScreener(db, wq, Settings(_env_file=None))
        verdicts = {v.alpha_id: v for v in await scr.screen([ca, cb, cc])}

        assert verdicts[ca].hard_redundant is True
        assert verdicts[ca].hard_ref_id == sub
        assert verdicts[cb].hard_redundant is False
        assert verdicts[cc].hard_redundant is False   # correlated but 10%+ better sharpe
        # FAIL written for A, not for B or C
        a_checks = (await db.get_backtest_result(ca)).checks or []
        assert any(c.get("name") == "SELF_CORRELATION" and c.get("result") == "FAIL" for c in a_checks)
        b_checks = (await db.get_backtest_result(cb)).checks or []
        assert not any(c.get("name") == "SELF_CORRELATION" for c in b_checks)
        c_checks = (await db.get_backtest_result(cc)).checks or []
        assert not any(c.get("name") == "SELF_CORRELATION" for c in c_checks)
    finally:
        await db.close()

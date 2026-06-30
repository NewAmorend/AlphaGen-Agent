from __future__ import annotations

from datetime import datetime

import httpx
import pytest

from wq_agent.config import Settings
from wq_agent.db import Database
from wq_agent.models import AlphaRecord, BacktestResult, GenerationStrategy, QualityGrade
from wq_agent.submitter import (
    SubmissionLimits,
    SubmissionManager,
    date_bucket,
    parse_sort_spec,
    seconds_until_daily_time,
)
from wq_agent.wq.client import WQClient


class FakeWQClient:
    def __init__(self, outcomes: dict[str, dict]):
        self.outcomes = outcomes
        self.submitted: list[str] = []

    async def submit_alpha(self, wq_alpha_id: str) -> dict:
        self.submitted.append(wq_alpha_id)
        outcome = self.outcomes.get(wq_alpha_id, {})
        if outcome.get("submit_error"):
            return {"status": "error", "message": outcome["submit_error"], "remote_status": "400"}
        return {"status": "success", "remote_status": "SUBMITTED"}

    async def poll_alpha_submission(self, wq_alpha_id: str) -> dict:
        return self.outcomes.get(wq_alpha_id, {"status": "active", "remote_status": "ACTIVE"})


@pytest.mark.parametrize(
    ("raw", "field", "descending"),
    [
        (None, "sharpe", True),
        ("-sharpe", "sharpe", True),
        ("fitness", "fitness", False),
        ("+returns", "returns", False),
        ("turnover", "turnover", False),
    ],
)
def test_parse_sort_spec(raw, field, descending):
    spec = parse_sort_spec(raw)
    assert spec.field == field
    assert spec.descending is descending


def test_parse_sort_spec_rejects_unknown_field():
    with pytest.raises(ValueError, match="unsupported"):
        parse_sort_spec("-unknown")


def test_submission_limits_reject_per_run_above_daily():
    with pytest.raises(ValueError, match="per-run"):
        SubmissionLimits(daily_limit=2, per_run_limit=3).validate()


def test_date_bucket_uses_configured_timezone():
    now = datetime.fromisoformat("2026-01-02T02:30:00+00:00")
    assert date_bucket(now, "America/New_York") == "2026-01-01"
    assert date_bucket(now, "Asia/Shanghai") == "2026-01-02"


def test_seconds_until_daily_time_rolls_to_next_day():
    now = datetime.fromisoformat("2026-01-01T10:00:00-05:00")
    delay = seconds_until_daily_time(
        now=now,
        timezone="America/New_York",
        daily_time="09:30",
    )
    assert delay == pytest.approx(23.5 * 60 * 60)


@pytest.mark.asyncio
async def test_prepare_queue_sorts_by_sharpe_and_filters_self_correlation(tmp_path):
    db = Database(str(tmp_path / "wq.db"))
    await db.connect()
    try:
        a1 = await _insert_high_alpha(db, "rank(a)", "wq1", sharpe=1.5, fitness=1.2)
        a2 = await _insert_high_alpha(db, "rank(b)", "wq2", sharpe=2.1, fitness=1.1)
        await _insert_high_alpha(
            db,
            "rank(c)",
            "wq3",
            sharpe=3.0,
            fitness=2.0,
            checks=[{"name": "SELF_CORRELATION", "result": "FAIL"}],
        )

        manager = SubmissionManager(db, None, Settings(_env_file=None))
        plan = await manager.prepare_queue(csv_dir=tmp_path, write_csv=True)
        queued = await db.list_submission_queue(status="pending")

        assert [row["alpha_id"] for row in plan.candidates] == [a2, a1]
        assert [row["alpha_id"] for row in queued] == [a2, a1]
        assert plan.csv_path is not None
        assert plan.csv_path.exists()
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_auto_submit_respects_limits_and_keeps_pending_queue(tmp_path):
    db = Database(str(tmp_path / "wq.db"))
    await db.connect()
    try:
        a1 = await _insert_high_alpha(db, "rank(a)", "wq1", sharpe=3.0, fitness=1.5)
        a2 = await _insert_high_alpha(db, "rank(b)", "wq2", sharpe=2.0, fitness=1.4)
        a3 = await _insert_high_alpha(db, "rank(c)", "wq3", sharpe=1.8, fitness=1.3)
        settings = Settings(_env_file=None, SUBMIT_DAILY_LIMIT=2, SUBMIT_PER_RUN_LIMIT=1)
        fake = FakeWQClient({"wq1": {"status": "active", "remote_status": "ACTIVE"}})
        manager = SubmissionManager(db, fake, settings)

        result = await manager.run_once(csv_dir=tmp_path, write_csv=False)
        pending = await db.list_submission_queue(status="pending")
        submitted = await db.get_alpha(a1)

        assert result.attempted == 1
        assert result.succeeded == 1
        assert fake.submitted == ["wq1"]
        assert submitted is not None and submitted.status.value == "submitted"
        assert [row["alpha_id"] for row in pending] == [a2, a3]
        assert await db.count_submission_attempts(date_bucket(timezone=settings.SUBMIT_TIMEZONE)) == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_auto_submit_records_self_correlation_failure_without_marking_submitted(tmp_path):
    db = Database(str(tmp_path / "wq.db"))
    await db.connect()
    try:
        a1 = await _insert_high_alpha(db, "rank(a)", "wq1", sharpe=3.0, fitness=1.5)
        settings = Settings(_env_file=None, SUBMIT_DAILY_LIMIT=2, SUBMIT_PER_RUN_LIMIT=1)
        fake = FakeWQClient(
            {"wq1": {"status": "self_correlation_fail", "remote_status": "UNSUBMITTED"}}
        )
        manager = SubmissionManager(db, fake, settings)

        result = await manager.run_once(csv_dir=tmp_path, write_csv=False)
        rejected = await db.list_submission_queue(status="rejected")
        alpha = await db.get_alpha(a1)
        backtest = await db.get_backtest_result(a1)

        assert result.failed == 1
        assert rejected[0]["alpha_id"] == a1
        assert rejected[0]["last_error"] == "SELF_CORRELATION FAIL"
        assert alpha is not None and alpha.status.value != "submitted"
        assert backtest is not None
        assert any(
            c.get("name") == "SELF_CORRELATION" and c.get("result") == "FAIL"
            for c in (backtest.checks or [])
        )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_failed_submit_attempt_consumes_daily_limit(tmp_path):
    db = Database(str(tmp_path / "wq.db"))
    await db.connect()
    try:
        await _insert_high_alpha(db, "rank(a)", "wq1", sharpe=3.0, fitness=1.5)
        await _insert_high_alpha(db, "rank(b)", "wq2", sharpe=2.0, fitness=1.4)
        settings = Settings(_env_file=None, SUBMIT_DAILY_LIMIT=1, SUBMIT_PER_RUN_LIMIT=1)
        fake = FakeWQClient({"wq1": {"submit_error": "remote rejected"}})
        manager = SubmissionManager(db, fake, settings)

        first = await manager.run_once(csv_dir=tmp_path, write_csv=False)
        second = await manager.run_once(csv_dir=tmp_path, write_csv=False)

        assert first.attempted == 1
        assert first.failed == 1
        assert second.attempted == 0
        assert fake.submitted == ["wq1"]
        assert await db.count_submission_attempts(date_bucket(timezone=settings.SUBMIT_TIMEZONE)) == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_upsert_submission_queue_deduplicates_and_refreshes_pending_rows(tmp_path):
    db = Database(str(tmp_path / "wq.db"))
    await db.connect()
    try:
        alpha_id = await _insert_high_alpha(db, "rank(a)", "wq1", sharpe=1.0, fitness=1.0)

        count = await db.upsert_submission_queue(
            [
                {"alpha_id": alpha_id, "wq_alpha_id": "wq1", "sort_value": 1.0},
                {"alpha_id": alpha_id, "wq_alpha_id": "wq1-new", "sort_value": 9.0},
            ],
            sort_key="-sharpe",
        )
        rows = await db.list_submission_queue(status="pending")

        assert count == 1
        assert len(rows) == 1
        assert rows[0]["wq_alpha_id"] == "wq1"
        assert rows[0]["sort_value"] == 1.0

        await db.upsert_submission_queue(
            [{"alpha_id": alpha_id, "wq_alpha_id": "wq1-refresh", "sort_value": 2.0}],
            sort_key="-fitness",
        )
        refreshed = await db.list_submission_queue(status="pending")

        assert len(refreshed) == 1
        assert refreshed[0]["wq_alpha_id"] == "wq1-refresh"
        assert refreshed[0]["sort_key"] == "-fitness"
        assert refreshed[0]["sort_value"] == 2.0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_upsert_submission_queue_does_not_overwrite_terminal_rows(tmp_path):
    db = Database(str(tmp_path / "wq.db"))
    await db.connect()
    try:
        alpha_id = await _insert_high_alpha(db, "rank(a)", "wq1", sharpe=1.0, fitness=1.0)
        await db.upsert_submission_queue(
            [{"alpha_id": alpha_id, "wq_alpha_id": "wq1", "sort_value": 1.0}],
            sort_key="-sharpe",
        )
        await db.update_submission_queue_status(
            alpha_id,
            status="rejected",
            last_error="SELF_CORRELATION FAIL",
            increment_attempts=True,
        )

        await db.upsert_submission_queue(
            [{"alpha_id": alpha_id, "wq_alpha_id": "wq1-new", "sort_value": 2.0}],
            sort_key="-fitness",
        )
        rows = await db.list_submission_queue(status="rejected")

        assert len(rows) == 1
        assert rows[0]["wq_alpha_id"] == "wq1"
        assert rows[0]["sort_key"] == "-sharpe"
        assert rows[0]["sort_value"] == 1.0
        assert rows[0]["status"] == "rejected"
        assert rows[0]["last_error"] == "SELF_CORRELATION FAIL"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_submission_attempt_reservation_finalizes_exact_log_once(tmp_path):
    db = Database(str(tmp_path / "wq.db"))
    await db.connect()
    try:
        alpha1 = await _insert_high_alpha(db, "rank(a)", "wq1", sharpe=1.0, fitness=1.0)
        alpha2 = await _insert_high_alpha(db, "rank(b)", "wq2", sharpe=1.1, fitness=1.1)
        bucket = "2026-01-01"

        attempt1 = await db.reserve_submission_attempt(
            alpha_id=alpha1,
            wq_alpha_id="wq1",
            date_bucket=bucket,
            daily_limit=5,
        )
        attempt2 = await db.reserve_submission_attempt(
            alpha_id=alpha2,
            wq_alpha_id="wq2",
            date_bucket=bucket,
            daily_limit=5,
        )

        assert attempt1 is not None
        assert attempt2 is not None
        assert attempt2 != attempt1
        assert await db.count_submission_attempts(bucket) == 2

        await db.finalize_submission_attempt(
            attempt_id=attempt1,
            alpha_id=alpha1,
            wq_alpha_id="wq1",
            date_bucket=bucket,
            result="success",
            remote_status="ACTIVE",
        )
        await db.finalize_submission_attempt(
            attempt_id=attempt1,
            alpha_id=alpha1,
            wq_alpha_id="wq1",
            date_bucket=bucket,
            result="failure",
            remote_status="ERROR",
        )

        assert await db.count_submission_attempts(bucket) == 2
        assert db._conn is not None
        cursor = await db._conn.execute(
            """SELECT result, remote_status
               FROM submission_log
               WHERE id = ?""",
            (attempt1,),
        )
        row = await cursor.fetchone()
        assert row is not None
        assert dict(row) == {"result": "success", "remote_status": "ACTIVE"}
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_poll_alpha_submission_does_not_mark_unsubmitted_self_corr_pass(monkeypatch):
    client = WQClient(Settings(_env_file=None))

    async def fake_request(method, path, **kwargs):
        assert method == "get"
        return httpx.Response(
            200,
            json={
                "status": "UNSUBMITTED",
                "is": {"checks": [{"name": "SELF_CORRELATION", "result": "PASS"}]},
            },
        )

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(client, "_request", fake_request)
    monkeypatch.setattr("wq_agent.wq.client.asyncio.sleep", no_sleep)

    result = await client.poll_alpha_submission("wq1", timeout=0.01, poll_interval=0.01)

    assert result["status"] == "pending"
    assert result["remote_status"] == "UNSUBMITTED"


@pytest.mark.asyncio
async def test_poll_alpha_submission_reports_parse_error(monkeypatch):
    client = WQClient(Settings(_env_file=None))

    async def fake_request(method, path, **kwargs):
        assert method == "get"
        return httpx.Response(200, text="not-json")

    monkeypatch.setattr(client, "_request", fake_request)

    result = await client.poll_alpha_submission("wq1", timeout=0.01, poll_interval=0.01)

    assert result == {
        "status": "error",
        "message": "failed to parse alpha submission response",
        "remote_status": "parse_error",
    }


async def _insert_high_alpha(
    db: Database,
    expression: str,
    wq_alpha_id: str,
    *,
    sharpe: float,
    fitness: float,
    checks: list[dict] | None = None,
) -> int:
    alpha_id = await db.insert_alpha(
        AlphaRecord(expression=expression, strategy=GenerationStrategy.LLM)
    )
    await db.insert_backtest_result(
        BacktestResult(
            alpha_id=alpha_id,
            fitness=fitness,
            sharpe=sharpe,
            turnover=0.2,
            returns=0.08,
            grade=QualityGrade.HIGH,
            checks=checks or [{"name": "LOW_SHARPE", "result": "PASS"}],
            wq_alpha_id=wq_alpha_id,
            created_at=datetime(2026, 1, 1, 9, 0, alpha_id),
        )
    )
    return alpha_id

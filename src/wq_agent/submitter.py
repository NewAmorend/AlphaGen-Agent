from __future__ import annotations

import asyncio
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from loguru import logger

from .config import Settings
from .db import Database
from .wq.client import WQClient


SORT_FIELDS = {"sharpe", "fitness", "returns", "turnover", "created_at"}
QUEUE_TERMINAL_STATUSES = {"submitted", "rejected"}


@dataclass(frozen=True)
class SortSpec:
    field: str
    descending: bool = True

    @property
    def raw(self) -> str:
        return f"-{self.field}" if self.descending else self.field


@dataclass(frozen=True)
class SubmissionLimits:
    daily_limit: int
    per_run_limit: int

    def validate(self) -> None:
        if self.daily_limit < 0:
            raise ValueError("daily submission limit must be >= 0")
        if self.per_run_limit < 0:
            raise ValueError("per-run submission limit must be >= 0")
        if self.per_run_limit > self.daily_limit:
            raise ValueError("per-run submission limit cannot exceed daily submission limit")


@dataclass(frozen=True)
class SubmissionPlan:
    candidates: list[dict[str, Any]]
    queued_count: int
    csv_path: Path | None


@dataclass(frozen=True)
class SubmissionRunResult:
    attempted: int
    succeeded: int
    failed: int
    skipped: int
    daily_remaining: int
    rows: list[dict[str, Any]]


def parse_sort_spec(raw: str | None) -> SortSpec:
    value = (raw or "-sharpe").strip()
    if not value:
        value = "-sharpe"
    descending = value.startswith("-")
    field = value[1:] if descending else value
    if field.startswith("+"):
        field = field[1:]
        descending = False
    if field not in SORT_FIELDS:
        allowed = ", ".join(sorted(SORT_FIELDS))
        raise ValueError(f"unsupported submission sort field '{field}' (allowed: {allowed})")
    return SortSpec(field=field, descending=descending)


def date_bucket(now: datetime | None = None, timezone: str = "America/New_York") -> str:
    tz = _load_timezone(timezone)
    current = now or datetime.now(tz)
    if current.tzinfo is None:
        current = current.replace(tzinfo=tz)
    return current.astimezone(tz).date().isoformat()


def seconds_until_daily_time(
    *,
    now: datetime | None = None,
    timezone: str = "America/New_York",
    daily_time: str = "09:30",
) -> float:
    tz = _load_timezone(timezone)
    hour, minute = _parse_daily_time(daily_time)
    current = now or datetime.now(tz)
    if current.tzinfo is None:
        current = current.replace(tzinfo=tz)
    current = current.astimezone(tz)
    target = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= current:
        target += timedelta(days=1)
    return max(0.0, (target - current).total_seconds())


def _load_timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"unknown submission timezone '{name}'") from exc


def _parse_daily_time(value: str) -> tuple[int, int]:
    try:
        hour_text, minute_text = value.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    except (ValueError, AttributeError) as exc:
        raise ValueError("daily submission time must be HH:MM") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("daily submission time must be HH:MM")
    return hour, minute


class SubmissionManager:
    def __init__(self, db: Database, wq: WQClient | None, settings: Settings):
        self.db = db
        self.wq = wq
        self.settings = settings

    async def prepare_queue(
        self,
        *,
        sort: str | None = None,
        min_fitness: float | None = None,
        limit: int = 1000,
        csv_dir: Path | str = "runs",
        write_csv: bool = True,
    ) -> SubmissionPlan:
        sort_spec = parse_sort_spec(sort or self.settings.SUBMIT_SORT)
        threshold = self.settings.MIN_FITNESS if min_fitness is None else min_fitness
        raw_rows = await self.db.list_submittable_alphas(min_fitness=threshold, limit=limit)
        candidates = self._sort_candidates(raw_rows, sort_spec)
        for row in candidates:
            row["sort_key"] = sort_spec.raw
            row["sort_value"] = _sortable_value(row, sort_spec.field)
        queued_count = await self.db.upsert_submission_queue(candidates, sort_key=sort_spec.raw)
        csv_path = self.export_candidates(candidates, csv_dir=csv_dir) if write_csv else None
        return SubmissionPlan(candidates=candidates, queued_count=queued_count, csv_path=csv_path)

    async def run_once(
        self,
        *,
        daily_limit: int | None = None,
        per_run_limit: int | None = None,
        sort: str | None = None,
        min_fitness: float | None = None,
        csv_dir: Path | str = "runs",
        write_csv: bool = True,
    ) -> SubmissionRunResult:
        if self.wq is None:
            raise RuntimeError("WQ client is required for auto-submit")
        limits = SubmissionLimits(
            daily_limit=self.settings.SUBMIT_DAILY_LIMIT if daily_limit is None else daily_limit,
            per_run_limit=self.settings.SUBMIT_PER_RUN_LIMIT if per_run_limit is None else per_run_limit,
        )
        limits.validate()
        await self.prepare_queue(
            sort=sort,
            min_fitness=min_fitness,
            csv_dir=csv_dir,
            write_csv=write_csv,
        )
        bucket = date_bucket(timezone=self.settings.SUBMIT_TIMEZONE)
        used_today = await self.db.count_submission_successes(bucket)
        daily_remaining = max(0, limits.daily_limit - used_today)
        submit_count = min(limits.per_run_limit, daily_remaining)
        if submit_count <= 0:
            return SubmissionRunResult(
                attempted=0,
                succeeded=0,
                failed=0,
                skipped=0,
                daily_remaining=daily_remaining,
                rows=[],
            )

        queued = await self.db.list_submission_queue(status="pending", limit=submit_count)
        attempted = succeeded = failed = 0
        rows: list[dict[str, Any]] = []
        for row in queued:
            attempted += 1
            result = await self._submit_one(row, bucket)
            rows.append(result)
            if result["result"] == "success":
                succeeded += 1
            else:
                failed += 1
        skipped = await self.db.count_submission_queue(status="pending")
        return SubmissionRunResult(
            attempted=attempted,
            succeeded=succeeded,
            failed=failed,
            skipped=skipped,
            daily_remaining=max(0, daily_remaining - succeeded),
            rows=rows,
        )

    async def _submit_one(self, row: dict[str, Any], bucket: str) -> dict[str, Any]:
        assert self.wq is not None
        alpha_id = int(row["alpha_id"])
        wq_alpha_id = str(row["wq_alpha_id"])
        try:
            submit_result = await self.wq.submit_alpha(wq_alpha_id)
            if submit_result.get("status") == "error":
                return await self._record_failure(
                    alpha_id,
                    wq_alpha_id,
                    bucket,
                    submit_result.get("message") or "submit failed",
                    remote_status=submit_result.get("remote_status"),
                )

            poll = await self.wq.poll_alpha_submission(wq_alpha_id)
            status = str(poll.get("status", "pending"))
            remote_status = poll.get("remote_status")
            if status in {"active", "submitted"}:
                submitted_at = datetime.now()
                await self.db.mark_alpha_submitted(alpha_id, submitted_at)
                await self.db.update_submission_queue_status(
                    alpha_id,
                    status="submitted",
                    submitted_at=submitted_at,
                    increment_attempts=True,
                )
                await self.db.insert_submission_log(
                    alpha_id=alpha_id,
                    wq_alpha_id=wq_alpha_id,
                    date_bucket=bucket,
                    action="submit",
                    result="success",
                    remote_status=remote_status,
                )
                return {
                    "alpha_id": alpha_id,
                    "wq_alpha_id": wq_alpha_id,
                    "result": "success",
                    "remote_status": remote_status,
                }
            if status == "self_correlation_fail":
                return await self._record_failure(
                    alpha_id,
                    wq_alpha_id,
                    bucket,
                    "SELF_CORRELATION FAIL",
                    remote_status=remote_status,
                    queue_status="rejected",
                )
            return await self._record_failure(
                alpha_id,
                wq_alpha_id,
                bucket,
                poll.get("message") or f"submission pending: {status}",
                remote_status=remote_status,
                queue_status="pending",
            )
        except Exception as exc:
            logger.warning(f"auto-submit failed for alpha {alpha_id}: {exc}")
            return await self._record_failure(alpha_id, wq_alpha_id, bucket, str(exc))

    async def _record_failure(
        self,
        alpha_id: int,
        wq_alpha_id: str,
        bucket: str,
        error: str,
        *,
        remote_status: str | None = None,
        queue_status: str = "pending",
    ) -> dict[str, Any]:
        await self.db.update_submission_queue_status(
            alpha_id,
            status=queue_status,
            last_error=error[:500],
            increment_attempts=True,
        )
        await self.db.insert_submission_log(
            alpha_id=alpha_id,
            wq_alpha_id=wq_alpha_id,
            date_bucket=bucket,
            action="submit",
            result="failure",
            remote_status=remote_status,
            error=error[:500],
        )
        return {
            "alpha_id": alpha_id,
            "wq_alpha_id": wq_alpha_id,
            "result": "failure",
            "remote_status": remote_status,
            "error": error,
        }

    def _sort_candidates(self, rows: list[dict[str, Any]], sort_spec: SortSpec) -> list[dict[str, Any]]:
        return sorted(
            rows,
            key=lambda row: _sort_tuple(row, sort_spec),
        )

    def export_candidates(self, candidates: list[dict[str, Any]], *, csv_dir: Path | str) -> Path:
        out_dir = Path(csv_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"submission_candidates_{datetime.now():%Y%m%d_%H%M%S}.csv"
        fieldnames = [
            "alpha_id",
            "wq_alpha_id",
            "sort_key",
            "sort_value",
            "fitness",
            "sharpe",
            "turnover",
            "returns",
            "grade",
            "expression",
        ]
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in candidates:
                writer.writerow({name: row.get(name) for name in fieldnames})
        return out_path


async def run_daily_daemon(
    manager: SubmissionManager,
    *,
    enabled: bool,
    daily_time: str,
    sleep: Any = asyncio.sleep,
) -> None:
    if not enabled:
        raise ValueError("daily auto-submit is disabled")
    while True:
        delay = seconds_until_daily_time(
            timezone=manager.settings.SUBMIT_TIMEZONE,
            daily_time=daily_time,
        )
        logger.info(f"Next auto-submit run in {delay:.0f}s")
        await sleep(delay)
        await manager.run_once()
        await sleep(60)


def _sort_tuple(row: dict[str, Any], sort_spec: SortSpec) -> tuple[int, Any]:
    value = _sortable_value(row, sort_spec.field)
    if value is None:
        return (1, 0)
    if sort_spec.descending and isinstance(value, (int, float)):
        return (0, -value)
    return (0, value)


def _sortable_value(row: dict[str, Any], field: str) -> Any:
    value = row.get(field)
    if field == "created_at" and isinstance(value, str):
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return None
    return value

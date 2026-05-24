from __future__ import annotations

import json
import struct
from datetime import datetime
from typing import Any

import aiosqlite
from loguru import logger

from .models import AlphaRecord, AlphaStatus, BacktestResult, GenerationStrategy, QualityGrade

_SCHEMA = """
CREATE TABLE IF NOT EXISTS alphas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    expression TEXT NOT NULL,
    strategy TEXT NOT NULL,
    llm_model TEXT,
    status TEXT NOT NULL DEFAULT 'generated',
    created_at TIMESTAMP NOT NULL,
    submitted_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alpha_id INTEGER NOT NULL REFERENCES alphas(id),
    region TEXT NOT NULL,
    universe TEXT NOT NULL,
    delay INTEGER NOT NULL,
    decay INTEGER NOT NULL,
    neutralization TEXT NOT NULL,
    sharpe REAL,
    turnover REAL,
    fitness REAL,
    returns REAL,
    drawdown REAL,
    grade TEXT,
    checks TEXT,
    wq_alpha_id TEXT,
    created_at TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_alphas_status ON alphas(status);
CREATE INDEX IF NOT EXISTS idx_backtest_alpha_id ON backtest_results(alpha_id);
CREATE INDEX IF NOT EXISTS idx_backtest_fitness ON backtest_results(fitness);

CREATE TABLE IF NOT EXISTS wiki_pages (
    path TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    type TEXT NOT NULL,
    tags TEXT,
    sources TEXT,
    content_hash TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS wiki_embeddings_blob (
    page_path TEXT PRIMARY KEY REFERENCES wiki_pages(path) ON DELETE CASCADE,
    dim INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    updated_at TIMESTAMP NOT NULL
);
"""


class Database:
    def __init__(self, db_path: str = "./wq_agent.db"):
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        self.vec_extension_loaded = await self._try_load_sqlite_vec()
        await self._exec_script(_SCHEMA)

    async def _try_load_sqlite_vec(self) -> bool:
        assert self._conn is not None
        try:
            import sqlite_vec  # type: ignore
        except ImportError:
            return False
        try:
            await self._conn.enable_load_extension(True)
            await self._conn.load_extension(sqlite_vec.loadable_path())
            await self._conn.enable_load_extension(False)
            logger.debug("Loaded sqlite-vec extension")
            return True
        except Exception as exc:
            logger.debug(f"sqlite-vec not available: {exc}")
            return False

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def _exec_script(self, sql: str) -> None:
        assert self._conn is not None
        await self._conn.executescript(sql)
        await self._conn.commit()

    async def insert_alpha(self, alpha: AlphaRecord) -> int:
        assert self._conn is not None
        cursor = await self._conn.execute(
            """INSERT INTO alphas (expression, strategy, llm_model, status, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (alpha.expression, alpha.strategy.value, alpha.llm_model, alpha.status.value, alpha.created_at.isoformat()),
        )
        await self._conn.commit()
        return cursor.lastrowid

    async def batch_insert_alphas(self, alphas: list[AlphaRecord]) -> list[int]:
        assert self._conn is not None
        ids = []
        for alpha in alphas:
            cursor = await self._conn.execute(
                """INSERT INTO alphas (expression, strategy, llm_model, status, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (alpha.expression, alpha.strategy.value, alpha.llm_model, alpha.status.value, alpha.created_at.isoformat()),
            )
            ids.append(cursor.lastrowid)
        await self._conn.commit()
        return ids

    async def update_alpha_status(self, alpha_id: int, status: AlphaStatus) -> None:
        assert self._conn is not None
        await self._conn.execute(
            "UPDATE alphas SET status = ? WHERE id = ?",
            (status.value, alpha_id),
        )
        await self._conn.commit()

    async def get_alpha(self, alpha_id: int) -> AlphaRecord | None:
        assert self._conn is not None
        row = await self._conn.execute("SELECT * FROM alphas WHERE id = ?", (alpha_id,))
        record = await row.fetchone()
        if not record:
            return None
        return AlphaRecord(
            id=record["id"],
            expression=record["expression"],
            strategy=GenerationStrategy(record["strategy"]),
            llm_model=record["llm_model"],
            status=AlphaStatus(record["status"]),
            created_at=datetime.fromisoformat(record["created_at"]),
            submitted_at=datetime.fromisoformat(record["submitted_at"]) if record["submitted_at"] else None,
        )

    async def list_alphas(
        self,
        status: AlphaStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AlphaRecord]:
        assert self._conn is not None
        if status:
            cursor = await self._conn.execute(
                "SELECT * FROM alphas WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (status.value, limit, offset),
            )
        else:
            cursor = await self._conn.execute(
                "SELECT * FROM alphas ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        rows = await cursor.fetchall()
        results = []
        for r in rows:
            results.append(
                AlphaRecord(
                    id=r["id"],
                    expression=r["expression"],
                    strategy=GenerationStrategy(r["strategy"]),
                    llm_model=r["llm_model"],
                    status=AlphaStatus(r["status"]),
                    created_at=datetime.fromisoformat(r["created_at"]),
                    submitted_at=datetime.fromisoformat(r["submitted_at"]) if r["submitted_at"] else None,
                )
            )
        return results

    async def insert_backtest_result(self, result: BacktestResult) -> int:
        assert self._conn is not None
        checks_json = json.dumps(result.checks) if result.checks else None
        grade_val = result.grade.value if result.grade else None
        cursor = await self._conn.execute(
            """INSERT INTO backtest_results
               (alpha_id, region, universe, delay, decay, neutralization,
                sharpe, turnover, fitness, returns, drawdown, grade, checks, wq_alpha_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result.alpha_id, result.region, result.universe, result.delay,
                result.decay, result.neutralization, result.sharpe, result.turnover,
                result.fitness, result.returns, result.drawdown, grade_val,
                checks_json, result.wq_alpha_id, result.created_at.isoformat(),
            ),
        )
        await self._conn.commit()
        return cursor.lastrowid

    async def get_backtest_result(self, alpha_id: int) -> BacktestResult | None:
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT * FROM backtest_results WHERE alpha_id = ? ORDER BY created_at DESC LIMIT 1",
            (alpha_id,),
        )
        r = await cursor.fetchone()
        if not r:
            return None
        checks = json.loads(r["checks"]) if r["checks"] else None
        grade = QualityGrade(r["grade"]) if r["grade"] else None
        return BacktestResult(
            id=r["id"],
            alpha_id=r["alpha_id"],
            region=r["region"],
            universe=r["universe"],
            delay=r["delay"],
            decay=r["decay"],
            neutralization=r["neutralization"],
            sharpe=r["sharpe"],
            turnover=r["turnover"],
            fitness=r["fitness"],
            returns=r["returns"],
            drawdown=r["drawdown"],
            grade=grade,
            checks=checks,
            wq_alpha_id=r["wq_alpha_id"],
            created_at=datetime.fromisoformat(r["created_at"]),
        )

    async def list_recent_backtested_alphas(self, limit: int = 20) -> list[dict[str, Any]]:
        """最近回测过的 alpha + 关键指标，供 generator 当 previous_results 喂回 LLM。"""
        assert self._conn is not None
        cursor = await self._conn.execute(
            """SELECT a.id AS alpha_id, a.expression, a.strategy, a.created_at,
                      b.fitness, b.sharpe, b.turnover, b.returns, b.grade, b.checks
               FROM alphas a
               JOIN backtest_results b ON a.id = b.alpha_id
               WHERE b.fitness IS NOT NULL
               ORDER BY b.created_at DESC
               LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            checks_raw = r["checks"]
            checks: list[dict] = []
            if checks_raw:
                try:
                    checks = json.loads(checks_raw)
                except (json.JSONDecodeError, TypeError):
                    checks = []
            failed = [
                str(c.get("name", ""))
                for c in checks
                if isinstance(c, dict) and str(c.get("result", "")).upper() == "FAIL"
            ]
            out.append({
                "alpha_id": r["alpha_id"],
                "alpha": r["expression"],
                "strategy": r["strategy"],
                "performance": {
                    "fitness": r["fitness"],
                    "sharpe": r["sharpe"],
                    "turnover": r["turnover"],
                    "returns": r["returns"],
                    "grade": r["grade"],
                },
                "failed_checks": failed,
            })
        return out

    async def list_high_quality_alphas(self, min_fitness: float = 1.0) -> list[dict[str, Any]]:
        assert self._conn is not None
        cursor = await self._conn.execute(
            """SELECT a.*, b.sharpe, b.turnover, b.fitness, b.returns, b.drawdown, b.grade, b.wq_alpha_id
               FROM alphas a
               JOIN backtest_results b ON a.id = b.alpha_id
               WHERE b.fitness >= ?
               ORDER BY b.fitness DESC""",
            (min_fitness,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_stats(self) -> dict[str, int]:
        assert self._conn is not None
        stats = {}
        for status in AlphaStatus:
            cursor = await self._conn.execute(
                "SELECT COUNT(*) as cnt FROM alphas WHERE status = ?", (status.value,)
            )
            row = await cursor.fetchone()
            stats[status.value] = row["cnt"]
        cursor = await self._conn.execute("SELECT COUNT(*) as cnt FROM backtest_results WHERE fitness >= 1.0")
        row = await cursor.fetchone()
        stats["high_quality_count"] = row["cnt"]
        return stats

    async def upsert_wiki_page(
        self,
        path: str,
        title: str,
        type_: str,
        tags: list[str],
        sources: list[str],
        content_hash: str,
    ) -> None:
        assert self._conn is not None
        await self._conn.execute(
            """INSERT INTO wiki_pages (path, title, type, tags, sources, content_hash, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(path) DO UPDATE SET
                   title=excluded.title,
                   type=excluded.type,
                   tags=excluded.tags,
                   sources=excluded.sources,
                   content_hash=excluded.content_hash,
                   updated_at=excluded.updated_at""",
            (
                path,
                title,
                type_,
                json.dumps(tags, ensure_ascii=False),
                json.dumps(sources, ensure_ascii=False),
                content_hash,
                datetime.now().isoformat(),
            ),
        )
        await self._conn.commit()

    async def bulk_upsert_wiki_pages(self, rows: list[dict]) -> None:
        """Single-transaction batch upsert. Each row needs path/title/type_/tags/sources/content_hash."""
        assert self._conn is not None
        if not rows:
            return
        now = datetime.now().isoformat()
        payload = [
            (
                r["path"],
                r["title"],
                r["type_"],
                json.dumps(r["tags"], ensure_ascii=False),
                json.dumps(r["sources"], ensure_ascii=False),
                r["content_hash"],
                now,
            )
            for r in rows
        ]
        await self._conn.executemany(
            """INSERT INTO wiki_pages (path, title, type, tags, sources, content_hash, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(path) DO UPDATE SET
                   title=excluded.title,
                   type=excluded.type,
                   tags=excluded.tags,
                   sources=excluded.sources,
                   content_hash=excluded.content_hash,
                   updated_at=excluded.updated_at""",
            payload,
        )
        await self._conn.commit()

    async def delete_wiki_pages(self, keep_paths: set[str]) -> int:
        assert self._conn is not None
        cursor = await self._conn.execute("SELECT path FROM wiki_pages")
        rows = await cursor.fetchall()
        to_drop = [(r["path"],) for r in rows if r["path"] not in keep_paths]
        if not to_drop:
            return 0
        await self._conn.executemany("DELETE FROM wiki_pages WHERE path = ?", to_drop)
        await self._conn.executemany(
            "DELETE FROM wiki_embeddings_blob WHERE page_path = ?", to_drop
        )
        await self._conn.commit()
        return len(to_drop)

    async def get_wiki_hashes(self) -> dict[str, str]:
        assert self._conn is not None
        cursor = await self._conn.execute("SELECT path, content_hash FROM wiki_pages")
        rows = await cursor.fetchall()
        return {r["path"]: r["content_hash"] for r in rows}

    async def upsert_wiki_embedding(self, path: str, embedding: list[float]) -> None:
        assert self._conn is not None
        blob = struct.pack(f"{len(embedding)}f", *embedding)
        await self._conn.execute(
            """INSERT INTO wiki_embeddings_blob (page_path, dim, embedding, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(page_path) DO UPDATE SET
                   dim=excluded.dim,
                   embedding=excluded.embedding,
                   updated_at=excluded.updated_at""",
            (path, len(embedding), blob, datetime.now().isoformat()),
        )
        await self._conn.commit()

    async def load_wiki_embeddings(self) -> dict[str, list[float]]:
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT page_path, dim, embedding FROM wiki_embeddings_blob"
        )
        rows = await cursor.fetchall()
        out: dict[str, list[float]] = {}
        for r in rows:
            dim = r["dim"]
            out[r["page_path"]] = list(struct.unpack(f"{dim}f", r["embedding"]))
        return out

    async def wiki_counts(self) -> dict[str, int]:
        assert self._conn is not None
        c = await self._conn.execute("SELECT COUNT(*) AS n FROM wiki_pages")
        pages = (await c.fetchone())["n"]
        c = await self._conn.execute("SELECT COUNT(*) AS n FROM wiki_embeddings_blob")
        embeds = (await c.fetchone())["n"]
        return {"wiki_pages": pages, "wiki_embeddings": embeds}

from __future__ import annotations

import json
import re
import struct
from datetime import datetime
from typing import Any

import aiosqlite
from loguru import logger

from .models import AlphaRecord, AlphaStatus, BacktestResult, GenerationStrategy, QualityGrade


# Structural-skeleton normalizer for exemplar diversification.
# 把表达式里的字段名（>=3 字符的小写标识符）替换成 FIELD、数字替换成 N，
# 得到 "结构骨架"。两个 alpha 同骨架意味着只是换了字段/窗口，本质上结构相同。
_FIELD_RE = re.compile(r"\b[a-z_][a-z0-9_]{2,}\b")
_NUM_RE = re.compile(r"\b\d+(\.\d+)?\b")
# 不视为 field 的关键字（算子名、分组名等）—— 必须保留以区分结构。
_SKELETON_KEEP = {
    "rank", "zscore", "scale", "normalize", "winsorize", "quantile", "log", "abs",
    "sign", "sqrt", "inverse", "reverse", "power", "signed_power", "tanh", "sigmoid",
    "exp", "min", "max", "median",
    "ts_mean", "ts_std_dev", "ts_sum", "ts_max", "ts_min", "ts_delta", "ts_delay",
    "ts_shift", "ts_rank", "ts_corr", "ts_covariance", "ts_decay_linear", "ts_zscore",
    "ts_av_diff", "ts_backfill", "ts_scale", "ts_quantile", "ts_argmax", "ts_argmin",
    "ts_arg_max", "ts_arg_min", "ts_product", "ts_step", "ts_count_nans", "ts_regression",
    "decay_linear", "hump", "jump_decay", "days_from_last_change", "trade_when",
    "group_neutralize", "group_rank", "group_zscore", "group_mean", "group_min",
    "group_max", "group_scale", "group_backfill", "group_cartesian_product",
    "vector_neut", "vec_avg", "vec_max", "vec_min", "vec_sum",
    "add", "subtract", "multiply", "divide", "densify", "to_nan",
    "bucket", "combo_a", "trade_when", "if_else", "is_nan",
    "and", "or", "not", "where",
    "reduce_avg", "reduce_choose", "reduce_count", "reduce_ir", "reduce_max",
    "reduce_min", "reduce_norm", "reduce_powersum", "reduce_range", "reduce_skewness",
    "reduce_stddev", "reduce_sum", "self_corr",
    "subindustry", "industry", "sector", "country", "currency", "exchange",
}


def expression_skeleton(expr: str) -> str:
    """规约表达式为"结构骨架"，用于 exemplar 多样性去重。

    示例：
      ts_decay_linear(rank(mdl177_x), 20) -> ts_decay_linear(rank(FIELD), N)
      ts_decay_linear(rank(fnd6_y), 60)   -> ts_decay_linear(rank(FIELD), N)   ← 同骨架
      add(rank(close), ts_corr(vwap, volume, 10)) -> add(rank(FIELD), ts_corr(FIELD, FIELD, N))
    """
    if not expr:
        return ""

    def _replace(m: re.Match) -> str:
        tok = m.group(0)
        return tok if tok in _SKELETON_KEEP else "FIELD"

    skel = _FIELD_RE.sub(_replace, expr.lower())
    skel = _NUM_RE.sub("N", skel)
    skel = re.sub(r"\s+", "", skel)
    return skel


def expression_outer_signature(expr: str, levels: int = 2) -> str:
    """提取表达式最外层 N 个算子链作为"外壳签名"，用于阻止同 wrapper 家族霸屏。

    full skeleton 看整个结构，outer signature 只看最外两层算子。例如：
      ts_decay_linear(group_neutralize(rank(F), subindustry), N)  ← outer 2: ts_decay_linear(group_neutralize)
      ts_decay_linear(group_neutralize(rank(signed_power(F,N)), subindustry), N)  ← 同 outer 2

    所以两个 alpha 即使骨架不同，只要外壳是同一组 wrapper 就被视为同 family。
    """
    if not expr:
        return ""
    # 用 skeleton 形式再扫前 N 个开括号前的算子名
    skel = expression_skeleton(expr)
    sig_ops: list[str] = []
    i = 0
    while i < len(skel) and len(sig_ops) < levels:
        # 找下一个算子（以字母开头，跟到 "(" 前）
        m = re.match(r"([a-z_][a-z0-9_]*)\(", skel[i:])
        if not m:
            break
        op = m.group(1)
        sig_ops.append(op)
        i += m.end()
    return "(".join(sig_ops)

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

CREATE TABLE IF NOT EXISTS field_blacklist (
    field_id TEXT PRIMARY KEY,
    fail_count INTEGER NOT NULL DEFAULT 0,
    last_reason TEXT,
    last_seen TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS alpha_pnl (
    alpha_id INTEGER PRIMARY KEY REFERENCES alphas(id),
    wq_alpha_id TEXT,
    dates TEXT NOT NULL,
    returns TEXT NOT NULL,
    fetched_at TIMESTAMP NOT NULL
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

    async def list_refine_candidates(self, limit: int = 10, low_fitness_floor: float = 0.5) -> list[dict[str, Any]]:
        """差一点就过的 alpha：MEDIUM 全部 + 高 fitness 的 LOW。按 fitness 倒序。

        - MEDIUM：只差 1 项 WQ 关键检查 FAIL，refine 性价比最高
        - LOW with fitness >= low_fitness_floor：信号已经出来（fitness 高），但有 2+ 项 check
          fail（典型是 sharpe 不达标），refine 可以同时修多项
        - REJECT 不进——信号本身就是噪声，refine 也救不回来
        - HIGH 已可提交无需 refine

        改前：只挑 MEDIUM，但冷启动时常无 MEDIUM，refine 命令直接返回 0。
        """
        assert self._conn is not None
        cursor = await self._conn.execute(
            """SELECT a.id AS alpha_id, a.expression, b.fitness, b.sharpe, b.turnover,
                      b.returns, b.grade, b.checks, b.created_at
               FROM alphas a
               JOIN backtest_results b ON a.id = b.alpha_id
               WHERE b.fitness IS NOT NULL
                 AND (b.grade = 'medium' OR (b.grade = 'low' AND b.fitness >= ?))
               ORDER BY
                  CASE b.grade WHEN 'medium' THEN 0 ELSE 1 END,
                  b.fitness DESC, b.created_at DESC
               LIMIT ?""",
            (low_fitness_floor, limit),
        )
        rows = await cursor.fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            checks: list[dict] = []
            if r["checks"]:
                try:
                    checks = json.loads(r["checks"])
                except (json.JSONDecodeError, TypeError):
                    pass
            failed = [
                str(c.get("name", ""))
                for c in checks
                if isinstance(c, dict) and str(c.get("result", "")).upper() == "FAIL"
            ]
            out.append({
                "alpha_id": r["alpha_id"],
                "expression": r["expression"],
                "fitness": r["fitness"],
                "sharpe": r["sharpe"],
                "turnover": r["turnover"],
                "returns": r["returns"],
                "grade": r["grade"],
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

    async def list_top_fitness_alphas(
        self, limit: int = 5, min_fitness: float = 0.0,
        diversify_by_skeleton: bool = True,
    ) -> list[dict[str, Any]]:
        """按 fitness 降序取 top-N，只要 fitness > min_fitness。用于 LLM prompt 的"历史模板"段。

        和 list_high_quality_alphas 的区别：那个是用 MIN_FITNESS 阈值（通常 1.0）筛"可提交"的，
        冷启动时永远为空。这个是 "把目前最好的 N 个端给 LLM 看"，哪怕 fitness 只有 0.4 也总比没有强。

        diversify_by_skeleton（默认 True）：拉 top 4*limit，按"结构骨架"去重再取 top-N。
        防止 exemplar mono-culture——比如 ts_decay_linear(group_neutralize(rank(F),subindustry),N)
        这种结构占满 top-5 后，LLM 强行往别的字段套同样结构，反而把生成质量拖垮。
        """
        assert self._conn is not None
        # 多拉一些候选给去重留余量
        fetch_limit = limit * 4 if diversify_by_skeleton else limit
        cursor = await self._conn.execute(
            """SELECT a.expression, b.fitness, b.sharpe, b.turnover, b.returns, b.grade
               FROM alphas a
               JOIN backtest_results b ON a.id = b.alpha_id
               WHERE b.fitness IS NOT NULL AND b.fitness > ?
               ORDER BY b.fitness DESC
               LIMIT ?""",
            (min_fitness, fetch_limit),
        )
        rows = await cursor.fetchall()
        candidates = [dict(r) for r in rows]

        if not diversify_by_skeleton:
            return candidates[:limit]

        # 三层去重，从严到松：
        # (1) outer-1（最外层算子）每个最多 2 个 —— 防 LLM 强行把所有信号塞进同一个 wrapper
        # (2) outer-2（前两层算子链）每个最多 1 个 —— 防同 wrapper 子家族重复
        # (3) full skeleton 完全相同 → 跳过
        # 如果严选下不够 limit，逐步放宽（先放 outer-1 配额，再放 outer-2 配额）。
        from collections import Counter
        OUTER1_QUOTA = 2

        def _select(outer1_quota: int) -> list[dict[str, Any]]:
            outer1_counts: Counter = Counter()
            seen_outer2: set[str] = set()
            seen_skeletons: set[str] = set()
            picked: list[dict[str, Any]] = []
            for row in candidates:
                expr = row.get("expression", "")
                outer1 = expression_outer_signature(expr, levels=1)
                outer2 = expression_outer_signature(expr, levels=2)
                skel = expression_skeleton(expr)
                if outer1 and outer1_counts[outer1] >= outer1_quota:
                    continue
                if outer2 and outer2 in seen_outer2:
                    continue
                if skel in seen_skeletons:
                    continue
                outer1_counts[outer1] += 1
                seen_outer2.add(outer2)
                seen_skeletons.add(skel)
                picked.append(row)
                if len(picked) >= limit:
                    break
            return picked

        # 严格策略：同一最外层算子最多 2 个。如果库里架构太单一拿不满 limit，
        # **宁可返回更少的多样化样本，也不要 5 个同质样本拖垮 LLM 生成多样性**。
        # 经验：Run 5 / Run 7 都是 exemplars 同质化（4-5 个同 op1）后产出崩盘。
        # 至少要保证 ≥ 3 个，否则放松到 outer2 dedup。
        picked = _select(OUTER1_QUOTA)
        if len(picked) >= 3:
            return picked  # 即使不到 limit 也返回，避免引入同质化
        # 库里架构太少，放宽到只看 outer2
        picked = _select(limit)
        return picked[:limit]

    # ------------------------------------------------------------- submission
    # 用户场景：一个 alpha 提交到 WQ Brain 后就不要让 LLM 再生成同款。
    # 三件事：(1) 标记本地 alpha 为 SUBMITTED；(2) 列出"可提交"候选；
    # (3) 从 WQ Brain 同步真实提交状态（权威源在远端）。

    async def mark_alpha_submitted(
        self, alpha_id: int, submitted_at: datetime | None = None
    ) -> None:
        assert self._conn is not None
        ts = (submitted_at or datetime.now()).isoformat()
        await self._conn.execute(
            "UPDATE alphas SET status = ?, submitted_at = ? WHERE id = ?",
            (AlphaStatus.SUBMITTED.value, ts, alpha_id),
        )
        await self._conn.commit()

    async def list_submittable_alphas(
        self, min_fitness: float = 1.0, limit: int = 50,
    ) -> list[dict[str, Any]]:
        """grade=HIGH（6 项 critical check 全 PASS）+ 尚未 SUBMITTED + fitness ≥ 阈值。

        额外过滤 SELF_CORRELATION = FAIL（与已有 alpha 重复度高 → WQ 不会接受）。
        同 wq_alpha_id 只留 fitness 最高的一个——refine 经常 verbatim 重复 base，
        WQ 上是同一个 alpha entry，列两遍只会造成混淆。
        """
        assert self._conn is not None
        cursor = await self._conn.execute(
            """SELECT a.id AS alpha_id, a.expression, a.status, a.created_at,
                      b.fitness, b.sharpe, b.turnover, b.returns, b.grade,
                      b.wq_alpha_id, b.checks
               FROM alphas a
               JOIN backtest_results b ON a.id = b.alpha_id
               WHERE a.status != ?
                 AND b.grade = 'high'
                 AND b.fitness >= ?
               ORDER BY b.fitness DESC, b.created_at DESC""",
            (AlphaStatus.SUBMITTED.value, min_fitness),
        )
        rows = await cursor.fetchall()
        seen_wq_ids: set[str] = set()
        out: list[dict[str, Any]] = []
        for r in rows:
            checks: list[dict] = []
            if r["checks"]:
                try:
                    checks = json.loads(r["checks"])
                except (json.JSONDecodeError, TypeError):
                    pass
            # SELF_CORRELATION 显式 FAIL 的过滤掉；PENDING 仍允许（WQ 还没算）
            sc_fail = any(
                str(c.get("name", "")).upper() == "SELF_CORRELATION"
                and str(c.get("result", "")).upper() == "FAIL"
                for c in checks if isinstance(c, dict)
            )
            if sc_fail:
                continue
            wq_id = r["wq_alpha_id"]
            if wq_id and wq_id in seen_wq_ids:
                continue
            if wq_id:
                seen_wq_ids.add(wq_id)
            out.append({
                "alpha_id": r["alpha_id"],
                "expression": r["expression"],
                "fitness": r["fitness"],
                "sharpe": r["sharpe"],
                "turnover": r["turnover"],
                "returns": r["returns"],
                "grade": r["grade"],
                "wq_alpha_id": wq_id,
                "status": r["status"],
            })
            if len(out) >= limit:
                break
        return out

    async def list_submitted_alphas(self, limit: int = 200) -> list[dict[str, Any]]:
        assert self._conn is not None
        cursor = await self._conn.execute(
            """SELECT a.id AS alpha_id, a.expression, a.submitted_at, a.llm_model,
                      b.fitness, b.sharpe, b.turnover, b.wq_alpha_id
               FROM alphas a
               LEFT JOIN backtest_results b ON a.id = b.alpha_id
               WHERE a.status = ?
               ORDER BY a.submitted_at DESC NULLS LAST
               LIMIT ?""",
            (AlphaStatus.SUBMITTED.value, limit),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_submitted_skeletons(self) -> set[str]:
        """已 SUBMITTED 因子的骨架集合（不含 self-corr FAIL）。保留接口主要给 stats 用。

        生成 prompt 黑名单请用 get_blacklisted_skeletons() —— 涵盖更全。
        """
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT expression FROM alphas WHERE status = ?",
            (AlphaStatus.SUBMITTED.value,),
        )
        rows = await cursor.fetchall()
        skeletons = {expression_skeleton(r["expression"]) for r in rows if r["expression"]}
        skeletons.discard("")
        return skeletons

    async def get_blacklisted_skeletons(self) -> set[str]:
        """LLM/refine prompt 应避开的骨架集合，两类合并：

        (1) **已 SUBMITTED** —— 用户在 WQ 网站正式提交过的，再生成也只是重复
        (2) **SELF_CORRELATION = FAIL** —— WQ 已经判定与已有 alpha 高度相关，
            即使没人提交，再造同款也会被 WQ 拒收，浪费回测

        前者来自 status 列，后者扫 backtest_results.checks 里的 SELF_CORRELATION 项。
        """
        assert self._conn is not None
        skeletons: set[str] = set()

        # (1) SUBMITTED
        cursor = await self._conn.execute(
            "SELECT expression FROM alphas WHERE status = ?",
            (AlphaStatus.SUBMITTED.value,),
        )
        for r in await cursor.fetchall():
            if r["expression"]:
                s = expression_skeleton(r["expression"])
                if s:
                    skeletons.add(s)

        # (2) self_correlation FAIL
        cursor = await self._conn.execute(
            """SELECT a.expression, b.checks
               FROM alphas a JOIN backtest_results b ON a.id = b.alpha_id
               WHERE b.checks IS NOT NULL"""
        )
        for r in await cursor.fetchall():
            checks_raw = r["checks"]
            try:
                checks = json.loads(checks_raw)
            except (json.JSONDecodeError, TypeError):
                continue
            sc_fail = any(
                str(c.get("name", "")).upper() == "SELF_CORRELATION"
                and str(c.get("result", "")).upper() == "FAIL"
                for c in checks if isinstance(c, dict)
            )
            if sc_fail and r["expression"]:
                s = expression_skeleton(r["expression"])
                if s:
                    skeletons.add(s)

        return skeletons

    async def get_low_fitness_skeletons(self, max_fitness: float = 0.3) -> set[str]:
        """历史已回测、且(同骨架的)最佳 fitness 始终低于 max_fitness 的结构骨架。

        用于阻止"反复重测已知低分结构"——每轮生成都可能重造一个上轮已测为 LOW/REJECT
        的骨架，白白再花一次模拟去重新发现它很差，库里也堆满近似克隆。

        关键：以**骨架**为粒度取全局最佳 fitness，而非单个 alpha。同一骨架只要有过一次
        fitness ≥ max_fitness（说明该结构有潜力），就不排除——换字段/窗口仍值得重试。
        """
        assert self._conn is not None
        cursor = await self._conn.execute(
            """SELECT a.expression, b.fitness
               FROM alphas a JOIN backtest_results b ON a.id = b.alpha_id
               WHERE b.fitness IS NOT NULL"""
        )
        best: dict[str, float] = {}
        for r in await cursor.fetchall():
            expr = r["expression"]
            if not expr:
                continue
            skel = expression_skeleton(expr)
            if not skel:
                continue
            fit = r["fitness"]
            if skel not in best or fit > best[skel]:
                best[skel] = fit
        return {s for s, f in best.items() if f < max_fitness}

    async def get_skeleton_distribution(self, limit: int = 20) -> dict[str, Any]:
        """库里已回测 alpha 的结构集中度报告，用于让"重复度"可见。

        按 outer-2 wrapper 家族（最外两层算子链）聚合——这是 redundancy 的主轴：
        家族数远小于 alpha 数 = 结构单一栽培。返回 top 家族的数量 + 平均/最佳 fitness，
        既看"哪个壳子霸屏"也看"霸屏的壳子是不是真有用"。
        """
        assert self._conn is not None
        cursor = await self._conn.execute(
            """SELECT a.expression, b.fitness
               FROM alphas a JOIN backtest_results b ON a.id = b.alpha_id
               WHERE a.expression IS NOT NULL AND a.expression != ''"""
        )
        rows = await cursor.fetchall()

        # 同时按 outer-1（最外层单算子）和 outer-2（最外两层算子链）聚合。
        # outer-1 抓"整个 decay 系霸屏"这类宏观单一化，outer-2 抓更细的子家族。
        counts: dict[int, dict[str, int]] = {1: {}, 2: {}}
        fits: dict[int, dict[str, list[float]]] = {1: {}, 2: {}}
        skel_set: set[str] = set()
        total = 0
        for r in rows:
            expr = r["expression"]
            sig2 = expression_outer_signature(expr, levels=2)
            sig1 = expression_outer_signature(expr, levels=1)
            skel = expression_skeleton(expr)
            if skel:
                skel_set.add(skel)
            if not sig2:
                continue
            total += 1
            for lvl, sig in ((1, sig1), (2, sig2)):
                if not sig:
                    continue
                counts[lvl][sig] = counts[lvl].get(sig, 0) + 1
                if r["fitness"] is not None:
                    fits[lvl].setdefault(sig, []).append(r["fitness"])

        def _family(lvl: int, sig: str) -> dict[str, Any]:
            fl = fits[lvl].get(sig, [])
            return {
                "signature": sig,
                "count": counts[lvl][sig],
                "avg_fitness": (sum(fl) / len(fl)) if fl else None,
                "max_fitness": max(fl) if fl else None,
            }

        def _top(lvl: int) -> list[dict[str, Any]]:
            ranked = sorted(counts[lvl], key=lambda s: counts[lvl][s], reverse=True)[:limit]
            return [_family(lvl, s) for s in ranked]

        return {
            "total_backtested": total,
            "unique_skeletons": len(skel_set),
            "unique_outer1": len(counts[1]),
            "unique_outer2": len(counts[2]),
            "top_outer1": _top(1),
            "top_outer2": _top(2),
        }

    async def update_backtest_checks(self, alpha_id: int, checks: list[dict]) -> None:
        """覆盖本地 backtest_results.checks——WQ 的 self_correlation 是异步算的，
        sync 时拉到最新的 checks 数据写回，让 self-corr FAIL 等状态及时反映到本地。
        """
        assert self._conn is not None
        checks_json = json.dumps(checks, ensure_ascii=False)
        await self._conn.execute(
            "UPDATE backtest_results SET checks = ? WHERE alpha_id = ?",
            (checks_json, alpha_id),
        )
        await self._conn.commit()

    async def get_cached_pnl(self, alpha_id: int) -> tuple[list[str], list[float]] | None:
        """返回缓存的（日期, 每日收益）；未缓存返回 None。"""
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT dates, returns FROM alpha_pnl WHERE alpha_id = ?", (alpha_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return json.loads(row["dates"]), json.loads(row["returns"])

    async def upsert_pnl(
        self, alpha_id: int, wq_alpha_id: str | None,
        dates: list[str], returns: list[float],
    ) -> None:
        assert self._conn is not None
        await self._conn.execute(
            """INSERT INTO alpha_pnl (alpha_id, wq_alpha_id, dates, returns, fetched_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(alpha_id) DO UPDATE SET
                   wq_alpha_id=excluded.wq_alpha_id,
                   dates=excluded.dates,
                   returns=excluded.returns,
                   fetched_at=excluded.fetched_at""",
            (alpha_id, wq_alpha_id, json.dumps(dates), json.dumps(returns),
             datetime.now().isoformat()),
        )
        await self._conn.commit()

    async def list_reference_alphas(self) -> dict[str, list[dict[str, Any]]]:
        """相关性参考集：submitted（硬 gate）与 HIGH-未提交（软提示）。
        每项 {alpha_id, wq_alpha_id, sharpe}；wq_alpha_id 为空的跳过（拉不了 PnL）。
        """
        assert self._conn is not None

        async def _q(where: str, params: tuple) -> list[dict[str, Any]]:
            # 每个 alpha 可能有多条 backtest_results（重测）——只取最新一条，
            # 否则同一 alpha 会重复进参考集、且可能带不同 sharpe，污染相关性判定。
            cursor = await self._conn.execute(
                f"""SELECT a.id AS alpha_id, b.wq_alpha_id, b.sharpe
                    FROM alphas a JOIN backtest_results b ON a.id = b.alpha_id
                    WHERE {where} AND b.wq_alpha_id IS NOT NULL
                      AND b.created_at = (
                          SELECT MAX(b2.created_at) FROM backtest_results b2
                          WHERE b2.alpha_id = a.id
                      )""",
                params,
            )
            return [dict(r) for r in await cursor.fetchall()]

        submitted = await _q("a.status = ?", (AlphaStatus.SUBMITTED.value,))
        high = await _q("a.status != ? AND b.grade = 'high'", (AlphaStatus.SUBMITTED.value,))
        return {"submitted": submitted, "high": high}

    async def find_alpha_by_wq_id(self, wq_alpha_id: str) -> int | None:
        """返回 wq_id 对应的第一个 local alpha id（向后兼容；多匹配请用 find_alphas_by_wq_id）。"""
        ids = await self.find_alphas_by_wq_id(wq_alpha_id)
        return ids[0] if ids else None

    async def find_alphas_by_wq_id(self, wq_alpha_id: str) -> list[int]:
        """返回所有 wq_id 对应的 local alpha ids。

        refine 经常 verbatim 重复（不同 local id 但同 wq_alpha_id），sync 需要全部更新
        否则 self_correlation 状态只反映到其中一个，其它仍 PENDING。
        """
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT alpha_id FROM backtest_results WHERE wq_alpha_id = ?",
            (wq_alpha_id,),
        )
        rows = await cursor.fetchall()
        return [r["alpha_id"] for r in rows]

    async def find_alpha_by_expression(self, expression: str) -> int | None:
        """fallback 匹配——如果 wq_alpha_id 对不上（可能 simulation 时没存），
        按 expression 完全相等找。"""
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT id FROM alphas WHERE expression = ? LIMIT 1",
            (expression,),
        )
        r = await cursor.fetchone()
        return r["id"] if r else None

    async def upsert_external_submitted_alpha(
        self,
        wq_alpha_id: str,
        expression: str,
        date_submitted: datetime | None,
        is_metrics: dict | None = None,
        region: str = "USA",
        universe: str = "TOP3000",
        delay: int = 1,
        neutralization: str = "INDUSTRY",
    ) -> int:
        """同步：把 WQ Brain 上有但本地没有的 alpha 灌进来，标记 SUBMITTED。

        这样它的骨架就进入 get_submitted_skeletons() 的黑名单，LLM 不会再生成。
        is_metrics 是 WQ 返回的 'is' 字段 dict（含 sharpe/fitness/turnover/checks/returns）。
        """
        assert self._conn is not None
        ts = (date_submitted or datetime.now()).isoformat()
        # 先看 expression 是否已存在
        existing = await self.find_alpha_by_expression(expression)
        if existing is not None:
            await self.mark_alpha_submitted(existing, date_submitted)
            return existing
        # 否则插入新的 external alpha
        cursor = await self._conn.execute(
            """INSERT INTO alphas (expression, strategy, llm_model, status, created_at, submitted_at)
               VALUES (?, 'llm', 'external:wq_brain', ?, ?, ?)""",
            (expression, AlphaStatus.SUBMITTED.value, ts, ts),
        )
        new_id = cursor.lastrowid
        # 如果有 is metrics，也写一条 backtest_result 以便 list_*_alphas 能看见 fitness
        if is_metrics:
            checks_json = json.dumps(is_metrics.get("checks", []), ensure_ascii=False) if is_metrics.get("checks") else None
            await self._conn.execute(
                """INSERT INTO backtest_results
                   (alpha_id, region, universe, delay, decay, neutralization,
                    sharpe, turnover, fitness, returns, drawdown, grade, checks, wq_alpha_id, created_at)
                   VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)""",
                (new_id, region, universe, delay, neutralization,
                 is_metrics.get("sharpe"), is_metrics.get("turnover"), is_metrics.get("fitness"),
                 is_metrics.get("returns"), is_metrics.get("drawdown"),
                 checks_json, wq_alpha_id, ts),
            )
        await self._conn.commit()
        return new_id

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

    async def reset_stuck_backtesting(self) -> int:
        """启动时回收上次被杀掉留下的 backtesting 状态。"""
        assert self._conn is not None
        cursor = await self._conn.execute(
            "UPDATE alphas SET status = 'generated' WHERE status = 'backtesting'"
        )
        n = cursor.rowcount
        await self._conn.commit()
        return n

    async def bump_field_blacklist(self, field_ids: list[str], reason: str = "sim_error") -> None:
        assert self._conn is not None
        if not field_ids:
            return
        now = datetime.now().isoformat()
        for fid in field_ids:
            await self._conn.execute(
                """INSERT INTO field_blacklist (field_id, fail_count, last_reason, last_seen)
                   VALUES (?, 1, ?, ?)
                   ON CONFLICT(field_id) DO UPDATE SET
                       fail_count = fail_count + 1,
                       last_reason = excluded.last_reason,
                       last_seen = excluded.last_seen""",
                (fid, reason, now),
            )
        await self._conn.commit()

    async def get_blacklisted_fields(self, min_fail_count: int = 3) -> set[str]:
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT field_id FROM field_blacklist WHERE fail_count >= ?",
            (min_fail_count,),
        )
        rows = await cursor.fetchall()
        return {r["field_id"] for r in rows}

    async def list_field_blacklist(self) -> list[dict[str, Any]]:
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT field_id, fail_count, last_reason, last_seen FROM field_blacklist ORDER BY fail_count DESC"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def clear_field_blacklist(self) -> int:
        assert self._conn is not None
        cursor = await self._conn.execute("DELETE FROM field_blacklist")
        await self._conn.commit()
        return cursor.rowcount

    async def wiki_counts(self) -> dict[str, int]:
        assert self._conn is not None
        c = await self._conn.execute("SELECT COUNT(*) AS n FROM wiki_pages")
        pages = (await c.fetchone())["n"]
        c = await self._conn.execute("SELECT COUNT(*) AS n FROM wiki_embeddings_blob")
        embeds = (await c.fetchone())["n"]
        return {"wiki_pages": pages, "wiki_embeddings": embeds}

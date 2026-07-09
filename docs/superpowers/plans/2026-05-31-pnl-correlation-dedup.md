# PnL Correlation Dedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Catch return-correlated duplicate alphas locally (before submit / before refine) using daily-PnL Pearson correlation, mirroring WQ Brain's self_correlation rejection.

**Architecture:** Lazily fetch + cache each alpha's daily-return vector from WQ's PnL recordset. A `CorrelationScreener` computes a candidate's max correlation against the reference set (submitted = hard gate, HIGH = soft flag) and, on a hard hit, writes a `SELF_CORRELATION=FAIL` check via the existing `update_backtest_checks` — which makes the existing submit-list filter, skeleton blacklist, and refine-skip all activate for free.

**Tech Stack:** Python 3.11, aiosqlite, httpx, pytest/pytest-asyncio. Run tests with `PYTHONPATH=src /home/agentuser/wq-agent/.venv/bin/python -m pytest`.

---

## File Structure

- **Modify** `src/alphagen_agent/config.py` — 3 new settings.
- **Modify** `src/alphagen_agent/db.py` — `alpha_pnl` cache table + `get_cached_pnl`/`upsert_pnl`/`list_reference_alphas`.
- **Modify** `src/alphagen_agent/wq/client.py` — `get_pnl()`.
- **Create** `src/alphagen_agent/engine/correlation.py` — pure helpers (`parse_pnl_response`, `pearson`, `align`, `max_correlation`, `is_hard_redundant`) + `CorrelationScreener`.
- **Modify** `scripts/batch_produce.py` — screen refine candidates.
- **Modify** `src/alphagen_agent/cli.py` — `screen-corr` command.
- **Create** `tests/test_correlation.py` — all unit tests (fake WQ client, tmp db).

All test runs in this plan use:
`PYTHONPATH=src /home/agentuser/wq-agent/.venv/bin/python -m pytest <args> -p no:cacheprovider`

---

## Task 1: Config settings

**Files:**
- Modify: `src/alphagen_agent/config.py` (after the `DEDUP_FITNESS_FLOOR` block, ~line 38)
- Test: `tests/test_correlation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_correlation.py
from __future__ import annotations
from alphagen_agent.config import Settings


def test_self_corr_settings_defaults():
    s = Settings(_env_file=None)
    assert s.SELF_CORR_THRESHOLD == 0.7
    assert s.SELF_CORR_SHARPE_MARGIN == 0.10
    assert s.SELF_CORR_MIN_OVERLAP == 60
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src /home/agentuser/wq-agent/.venv/bin/python -m pytest tests/test_correlation.py::test_self_corr_settings_defaults -v -p no:cacheprovider`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'SELF_CORR_THRESHOLD'`

- [ ] **Step 3: Add the settings**

In `src/alphagen_agent/config.py`, immediately after the existing line `DEDUP_FITNESS_FLOOR: float = 0.3`:

```python
    # PnL 收益相关性防重复（gap #4）
    SELF_CORR_THRESHOLD: float = 0.7        # 相关性 cutoff（对齐 WQ）
    SELF_CORR_SHARPE_MARGIN: float = 0.10   # sharpe 超越豁免线（高 10% 则 WQ 仍收）
    SELF_CORR_MIN_OVERLAP: int = 60         # 两向量重叠不足此天数则跳过（判"未知"）
```

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2. Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/alphagen_agent/config.py tests/test_correlation.py
git commit -m "feat(corr): add self-correlation dedup settings"
```

---

## Task 2: PnL cache table + accessors

**Files:**
- Modify: `src/alphagen_agent/db.py` (add table to `_SCHEMA` ~line 145; add methods near `update_backtest_checks` ~line 737)
- Test: `tests/test_correlation.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_correlation.py
import pytest
from alphagen_agent.db import Database


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src /home/agentuser/wq-agent/.venv/bin/python -m pytest tests/test_correlation.py::test_pnl_cache_round_trip -v -p no:cacheprovider`
Expected: FAIL — `AttributeError: 'Database' object has no attribute 'get_cached_pnl'`

- [ ] **Step 3a: Add the table to `_SCHEMA`**

In `src/alphagen_agent/db.py`, inside the `_SCHEMA = """..."""` string, after the `field_blacklist` table (before the closing `"""` at ~line 146):

```sql
CREATE TABLE IF NOT EXISTS alpha_pnl (
    alpha_id INTEGER PRIMARY KEY REFERENCES alphas(id),
    wq_alpha_id TEXT,
    dates TEXT NOT NULL,
    returns TEXT NOT NULL,
    fetched_at TIMESTAMP NOT NULL
);
```

- [ ] **Step 3b: Add the accessor methods**

In `src/alphagen_agent/db.py`, immediately after `update_backtest_checks` (~line 748), add:

```python
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
```

(`json` and `datetime` are already imported at the top of `db.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2. Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/alphagen_agent/db.py tests/test_correlation.py
git commit -m "feat(corr): add alpha_pnl cache table + accessors"
```

---

## Task 3: Reference-set query (submitted + HIGH)

**Files:**
- Modify: `src/alphagen_agent/db.py` (add method after `upsert_pnl`)
- Test: `tests/test_correlation.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_correlation.py
from datetime import datetime
from alphagen_agent.models import AlphaRecord, BacktestResult, GenerationStrategy, AlphaStatus, QualityGrade


async def _seed_alpha(db, expr, grade, sharpe, wq_id, status=AlphaStatus.GENERATED):
    aid = await db.insert_alpha(AlphaRecord(expression=expr, strategy=GenerationStrategy.LLM,
                                            status=status, created_at=datetime.now()))
    await db.insert_backtest_result(BacktestResult(
        alpha_id=aid, sharpe=sharpe, fitness=1.2, grade=grade,
        wq_alpha_id=wq_id, created_at=datetime.now()))
    return aid


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src /home/agentuser/wq-agent/.venv/bin/python -m pytest tests/test_correlation.py::test_list_reference_alphas -v -p no:cacheprovider`
Expected: FAIL — `AttributeError: 'Database' object has no attribute 'list_reference_alphas'`

- [ ] **Step 3: Add the method**

In `src/alphagen_agent/db.py`, after `upsert_pnl`:

```python
    async def list_reference_alphas(self) -> dict[str, list[dict[str, Any]]]:
        """相关性参考集：submitted（硬 gate）与 HIGH-未提交（软提示）。
        每项 {alpha_id, wq_alpha_id, sharpe}；wq_alpha_id 为空的跳过（拉不了 PnL）。
        """
        assert self._conn is not None

        async def _q(where: str, params: tuple) -> list[dict[str, Any]]:
            cursor = await self._conn.execute(
                f"""SELECT a.id AS alpha_id, b.wq_alpha_id, b.sharpe
                    FROM alphas a JOIN backtest_results b ON a.id = b.alpha_id
                    WHERE {where} AND b.wq_alpha_id IS NOT NULL""",
                params,
            )
            return [dict(r) for r in await cursor.fetchall()]

        submitted = await _q("a.status = ?", (AlphaStatus.SUBMITTED.value,))
        high = await _q("a.status != ? AND b.grade = 'high'", (AlphaStatus.SUBMITTED.value,))
        return {"submitted": submitted, "high": high}
```

(`AlphaStatus` and `Any` are already imported in `db.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2. Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/alphagen_agent/db.py tests/test_correlation.py
git commit -m "feat(corr): add reference-set query (submitted + HIGH)"
```

---

## Task 4: WQ client `get_pnl` + PnL parser

**Files:**
- Create: `src/alphagen_agent/engine/correlation.py` (parser only for now)
- Modify: `src/alphagen_agent/wq/client.py` (add `get_pnl`)
- Test: `tests/test_correlation.py`

> **VERIFY-AT-RUNTIME:** The exact `/alphas/{id}/recordsets/pnl` response shape is not yet
> confirmed against a live alpha. This task assumes `{"records": [[date, cumulative_pnl], ...]}`.
> The parser below is tolerant (skips malformed rows). Before relying on it in production, run
> `get_pnl` once against a real `wq_alpha_id` and confirm column order/format; adjust
> `parse_pnl_response` only if different. This does not change any other task.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_correlation.py
from alphagen_agent.engine.correlation import parse_pnl_response


def test_parse_pnl_response_diffs_to_daily_returns():
    data = {"records": [["2020-01-02", 0.0], ["2020-01-03", 1.5], ["2020-01-06", 1.0]]}
    dates, returns = parse_pnl_response(data)
    # daily returns = diff of cumulative pnl; first date dropped
    assert dates == ["2020-01-03", "2020-01-06"]
    assert returns == [1.5, -0.5]


def test_parse_pnl_response_skips_malformed():
    data = {"records": [["2020-01-02", 0.0], ["2020-01-03", None], "junk", ["2020-01-06", 2.0]]}
    dates, returns = parse_pnl_response(data)
    # None and "junk" rows dropped before diffing
    assert dates == ["2020-01-06"]
    assert returns == [2.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src /home/agentuser/wq-agent/.venv/bin/python -m pytest tests/test_correlation.py::test_parse_pnl_response_diffs_to_daily_returns -v -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'alphagen_agent.engine.correlation'`

- [ ] **Step 3a: Create the parser module**

Create `src/alphagen_agent/engine/correlation.py`:

```python
from __future__ import annotations


def parse_pnl_response(data: dict) -> tuple[list[str], list[float]]:
    """WQ PnL recordset（累计 PnL）→（日期, 每日收益）。

    假设 records 为 [[date, cumulative_pnl], ...]。容错：跳过缺值/格式错误的行。
    每日收益 = 累计 PnL 的逐日差分（首日丢弃）。
    """
    records = data.get("records") or []
    dates: list[str] = []
    cum: list[float] = []
    for rec in records:
        if not isinstance(rec, (list, tuple)) or len(rec) < 2:
            continue
        d, v = rec[0], rec[1]
        if v is None:
            continue
        try:
            cum.append(float(v))
        except (TypeError, ValueError):
            continue
        dates.append(str(d))
    daily = [cum[i] - cum[i - 1] for i in range(1, len(cum))]
    return dates[1:], daily
```

- [ ] **Step 3b: Add `get_pnl` to the WQ client**

In `src/alphagen_agent/wq/client.py`, after `get_alpha_check` (~line 392), add:

```python
    async def get_pnl(self, wq_alpha_id: str) -> tuple[list[str], list[float]] | None:
        """拉 alpha 的 PnL recordset → （日期, 每日收益）。失败返回 None（fail-open）。"""
        from ..engine.correlation import parse_pnl_response
        try:
            resp = await self._request("get", f"/alphas/{wq_alpha_id}/recordsets/pnl")
        except Exception as exc:
            logger.warning(f"get_pnl({wq_alpha_id}) request failed: {exc}")
            return None
        if resp.status_code != 200:
            logger.warning(f"get_pnl({wq_alpha_id}) status {resp.status_code}")
            return None
        try:
            return parse_pnl_response(resp.json())
        except Exception as exc:
            logger.warning(f"get_pnl({wq_alpha_id}) parse failed: {exc}")
            return None
```

(`logger` is already imported in `client.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src /home/agentuser/wq-agent/.venv/bin/python -m pytest tests/test_correlation.py -k parse_pnl -v -p no:cacheprovider`
Expected: PASS (both parser tests)

- [ ] **Step 5: Commit**

```bash
git add src/alphagen_agent/engine/correlation.py src/alphagen_agent/wq/client.py tests/test_correlation.py
git commit -m "feat(corr): add PnL parser + WQ get_pnl endpoint"
```

---

## Task 5: Pure math — `pearson` + `align`

**Files:**
- Modify: `src/alphagen_agent/engine/correlation.py`
- Test: `tests/test_correlation.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_correlation.py
from alphagen_agent.engine.correlation import pearson, align


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src /home/agentuser/wq-agent/.venv/bin/python -m pytest tests/test_correlation.py -k "pearson or align" -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'pearson'`

- [ ] **Step 3: Implement `pearson` + `align`**

Append to `src/alphagen_agent/engine/correlation.py`:

```python
def pearson(a: list[float], b: list[float]) -> float:
    """Pearson 相关系数。长度 < 2 或任一方零方差 → 0.0。"""
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a, b = a[:n], b[:n]
    ma = sum(a) / n
    mb = sum(b) / n
    da = [x - ma for x in a]
    db = [y - mb for y in b]
    num = sum(x * y for x, y in zip(da, db))
    va = sum(x * x for x in da)
    vb = sum(y * y for y in db)
    if va <= 0 or vb <= 0:
        return 0.0
    return num / ((va ** 0.5) * (vb ** 0.5))


def align(
    dates_a: list[str], ra: list[float],
    dates_b: list[str], rb: list[float],
) -> tuple[list[float], list[float]]:
    """按日期取重叠段，返回两条对齐向量（按日期排序）。"""
    ma = dict(zip(dates_a, ra))
    mb = dict(zip(dates_b, rb))
    common = sorted(set(ma) & set(mb))
    return [ma[d] for d in common], [mb[d] for d in common]
```

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2. Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/alphagen_agent/engine/correlation.py tests/test_correlation.py
git commit -m "feat(corr): add pearson + date-alignment helpers"
```

---

## Task 6: Decision logic — `max_correlation` + `is_hard_redundant`

**Files:**
- Modify: `src/alphagen_agent/engine/correlation.py`
- Test: `tests/test_correlation.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_correlation.py
from alphagen_agent.engine.correlation import max_correlation, is_hard_redundant


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src /home/agentuser/wq-agent/.venv/bin/python -m pytest tests/test_correlation.py -k "max_correlation or hard_redundant" -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'max_correlation'`

- [ ] **Step 3: Implement both**

Append to `src/alphagen_agent/engine/correlation.py`:

```python
def max_correlation(
    cand_dates: list[str], cand_returns: list[float],
    refs: list[dict], min_overlap: int,
) -> tuple[float, int | None, float | None]:
    """候选对参考集求最大 |相关|（重叠 < min_overlap 的 ref 跳过）。

    refs 每项需含 alpha_id / sharpe / dates / returns。
    返回 (max_corr, ref_alpha_id, ref_sharpe)；无可比 ref → (0.0, None, None)。
    """
    best_corr = 0.0
    best_id: int | None = None
    best_sharpe: float | None = None
    for ref in refs:
        va, vb = align(cand_dates, cand_returns, ref.get("dates", []), ref.get("returns", []))
        if len(va) < min_overlap:
            continue
        corr = pearson(va, vb)
        if abs(corr) > abs(best_corr):
            best_corr, best_id, best_sharpe = corr, ref["alpha_id"], ref.get("sharpe")
    return best_corr, best_id, best_sharpe


def is_hard_redundant(
    cand_sharpe: float | None, max_corr: float, ref_sharpe: float | None,
    threshold: float, margin: float,
) -> bool:
    """硬 gate：|相关| > threshold 且 候选 sharpe 没比命中的 ref 高 margin → 判重。

    镜像 WQ：高相关但 sharpe 明显更好（≥ margin）的不算重复（WQ 视为升级版）。
    缺 ref 或缺 sharpe → 不判重（fail-open）。
    """
    if ref_sharpe is None or abs(max_corr) <= threshold:
        return False
    if cand_sharpe is None:
        return True  # 相关超阈值又拿不到候选 sharpe → 保守判重
    return cand_sharpe < (1.0 + margin) * ref_sharpe
```

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2. Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/alphagen_agent/engine/correlation.py tests/test_correlation.py
git commit -m "feat(corr): add max-correlation + hard-gate decision rule"
```

---

## Task 7: `CorrelationScreener.ensure_pnl` (lazy cache)

**Files:**
- Modify: `src/alphagen_agent/engine/correlation.py`
- Test: `tests/test_correlation.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_correlation.py
from alphagen_agent.engine.correlation import CorrelationScreener
from alphagen_agent.config import Settings


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src /home/agentuser/wq-agent/.venv/bin/python -m pytest tests/test_correlation.py::test_ensure_pnl_lazy_and_cached -v -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'CorrelationScreener'`

- [ ] **Step 3: Add the class skeleton + `ensure_pnl`**

Append to `src/alphagen_agent/engine/correlation.py`:

```python
from dataclasses import dataclass

from loguru import logger


@dataclass
class Verdict:
    alpha_id: int
    hard_redundant: bool
    hard_corr: float
    hard_ref_id: int | None
    soft_corr: float
    soft_ref_id: int | None


class CorrelationScreener:
    """对候选 alpha 算 PnL 相关性，命中硬 gate 写 SELF_CORRELATION FAIL。"""

    def __init__(self, db, wq, settings):
        self.db = db
        self.wq = wq
        self.settings = settings

    async def ensure_pnl(
        self, alpha_id: int, wq_alpha_id: str | None,
    ) -> tuple[list[str], list[float]] | None:
        """懒加载：缓存命中直接返回；否则拉取并缓存。失败返回 None（不缓存）。"""
        cached = await self.db.get_cached_pnl(alpha_id)
        if cached is not None:
            return cached
        if not wq_alpha_id:
            return None
        fetched = await self.wq.get_pnl(wq_alpha_id)
        if fetched is None:
            return None
        dates, returns = fetched
        await self.db.upsert_pnl(alpha_id, wq_alpha_id, dates, returns)
        return dates, returns
```

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2. Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/alphagen_agent/engine/correlation.py tests/test_correlation.py
git commit -m "feat(corr): add CorrelationScreener.ensure_pnl lazy cache"
```

---

## Task 8: `CorrelationScreener.screen` — verdicts + write FAIL

**Files:**
- Modify: `src/alphagen_agent/engine/correlation.py`
- Test: `tests/test_correlation.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_correlation.py
import json as _json


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

        wq = _FakeWQ({})  # all cached, no fetch needed
        scr = CorrelationScreener(db, wq, Settings(_env_file=None))
        verdicts = {v.alpha_id: v for v in await scr.screen([ca, cb])}

        assert verdicts[ca].hard_redundant is True
        assert verdicts[ca].hard_ref_id == sub
        assert verdicts[cb].hard_redundant is False
        # FAIL written for A, not for B
        a_checks = (await db.get_backtest_result(ca)).checks or []
        assert any(c.get("name") == "SELF_CORRELATION" and c.get("result") == "FAIL" for c in a_checks)
        b_checks = (await db.get_backtest_result(cb)).checks or []
        assert not any(c.get("name") == "SELF_CORRELATION" for c in b_checks)
    finally:
        await db.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src /home/agentuser/wq-agent/.venv/bin/python -m pytest tests/test_correlation.py::test_screen_marks_hard_redundant -v -p no:cacheprovider`
Expected: FAIL — `AttributeError: 'CorrelationScreener' object has no attribute 'screen'`

- [ ] **Step 3: Implement `screen` + helpers**

Append these methods inside `CorrelationScreener` in `src/alphagen_agent/engine/correlation.py`:

```python
    async def _load_refs(self, ref_rows: list[dict]) -> list[dict]:
        """给参考集行补上 PnL（懒加载），丢掉拉不到的。"""
        out: list[dict] = []
        for r in ref_rows:
            pnl = await self.ensure_pnl(r["alpha_id"], r.get("wq_alpha_id"))
            if pnl is None:
                continue
            out.append({"alpha_id": r["alpha_id"], "sharpe": r.get("sharpe"),
                        "dates": pnl[0], "returns": pnl[1]})
        return out

    async def screen(self, candidate_ids: list[int]) -> list[Verdict]:
        ref = await self.db.list_reference_alphas()
        submitted_refs = await self._load_refs(ref["submitted"])
        high_refs = await self._load_refs(ref["high"])
        min_overlap = self.settings.SELF_CORR_MIN_OVERLAP
        thr = self.settings.SELF_CORR_THRESHOLD
        margin = self.settings.SELF_CORR_SHARPE_MARGIN

        verdicts: list[Verdict] = []
        for aid in candidate_ids:
            bt = await self.db.get_backtest_result(aid)
            if bt is None:
                continue
            pnl = await self.ensure_pnl(aid, bt.wq_alpha_id)
            if pnl is None:
                verdicts.append(Verdict(aid, False, 0.0, None, 0.0, None))
                continue
            cd, cr = pnl
            # 硬 gate：vs submitted（排除自己）
            sub = [r for r in submitted_refs if r["alpha_id"] != aid]
            h_corr, h_ref, h_ref_sh = max_correlation(cd, cr, sub, min_overlap)
            hard = is_hard_redundant(bt.sharpe, h_corr, h_ref_sh, thr, margin)
            # 软提示：vs 未提交 HIGH（排除自己）
            hi = [r for r in high_refs if r["alpha_id"] != aid]
            s_corr, s_ref, _ = max_correlation(cd, cr, hi, min_overlap)

            if hard:
                checks = list(bt.checks or [])
                checks = [c for c in checks if str(c.get("name", "")).upper() != "SELF_CORRELATION"]
                checks.append({"name": "SELF_CORRELATION", "result": "FAIL",
                               "value": round(h_corr, 4), "limit": thr, "source": "local_pnl"})
                await self.db.update_backtest_checks(aid, checks)
                logger.info(f"#{aid} hard-redundant: corr={h_corr:.3f} vs #{h_ref} -> SELF_CORRELATION FAIL")
            elif abs(s_corr) > thr:
                logger.info(f"#{aid} soft-corr {s_corr:.3f} vs unsubmitted HIGH #{s_ref} (advisory)")

            verdicts.append(Verdict(aid, hard, h_corr, h_ref, s_corr, s_ref))
        return verdicts
```

- [ ] **Step 4: Run test to verify it passes**

Run: same as Step 2. Expected: PASS

- [ ] **Step 5: Run the full correlation test file**

Run: `PYTHONPATH=src /home/agentuser/wq-agent/.venv/bin/python -m pytest tests/test_correlation.py -v -p no:cacheprovider`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/alphagen_agent/engine/correlation.py tests/test_correlation.py
git commit -m "feat(corr): add screen() verdicts + write local SELF_CORRELATION FAIL"
```

---

## Task 9: Wire screener into the batch_produce refine filter

**Files:**
- Modify: `scripts/batch_produce.py` (the refine pass, after `cands = await orch.db.list_refine_candidates(...)`)

- [ ] **Step 1: Add the screening step before candidate filtering**

In `scripts/batch_produce.py`, locate the refine pass loop. Replace the block:

```python
        for p in range(1, refine_passes + 1):
            cands = await orch.db.list_refine_candidates(limit=100)
            todo = [c for c in cands if c["alpha_id"] not in refined and not _is_redundant(c)]
```

with:

```python
        from alphagen_agent.engine.correlation import CorrelationScreener
        screener = CorrelationScreener(orch.db, orch.wq, orch.settings)
        for p in range(1, refine_passes + 1):
            cands = await orch.db.list_refine_candidates(limit=100)
            # 先用 PnL 相关性筛一遍——命中硬 gate 的会被写 SELF_CORRELATION FAIL，
            # 下面的 _is_redundant 立刻就能把它们挡掉（复用现有约定）。
            await screener.screen([c["alpha_id"] for c in cands])
            cands = await orch.db.list_refine_candidates(limit=100)  # 重新读，拿到刚写的 FAIL
            todo = [c for c in cands if c["alpha_id"] not in refined and not _is_redundant(c)]
```

- [ ] **Step 2: Smoke-check the script still imports/parses**

Run: `PYTHONPATH=src /home/agentuser/wq-agent/.venv/bin/python scripts/batch_produce.py --help`
Expected: usage text prints, exit 0 (no import error)

- [ ] **Step 3: Run full suite (no regressions)**

Run: `PYTHONPATH=src /home/agentuser/wq-agent/.venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/batch_produce.py
git commit -m "feat(corr): screen refine candidates by PnL correlation before refining"
```

---

## Task 10: `screen-corr` CLI command

**Files:**
- Modify: `src/alphagen_agent/cli.py` (add a command near `submittable`, ~line 357)

- [ ] **Step 1: Add the command**

In `src/alphagen_agent/cli.py`, add a new command (place it right before the `submittable` command definition):

```python
@app.command(name="screen-corr")
def screen_corr(
    scope: str = typer.Option("candidates", "--scope",
                              help="candidates | high | all"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """用 PnL 相关性筛重复：命中硬 gate 的写本地 SELF_CORRELATION FAIL
    （自动踢出可提交列表 + 拉黑骨架 + refine 跳过）。"""
    _setup_logging(verbose)
    from .engine.correlation import CorrelationScreener

    async def _run():
        orch = Orchestrator()
        try:
            await orch.initialize()
            if scope == "high":
                rows = await orch.db.list_high_quality_alphas(min_fitness=0.0)
                ids = [r["id"] for r in rows]
            elif scope == "all":
                cand = await orch.db.list_refine_candidates(limit=300)
                hi = await orch.db.list_high_quality_alphas(min_fitness=0.0)
                ids = list({*(c["alpha_id"] for c in cand), *(r["id"] for r in hi)})
            else:  # candidates
                cand = await orch.db.list_refine_candidates(limit=300)
                ids = [c["alpha_id"] for c in cand]
            screener = CorrelationScreener(orch.db, orch.wq, orch.settings)
            verdicts = await screener.screen(ids)
            hard = [v for v in verdicts if v.hard_redundant]
            console.print(f"Screened [cyan]{len(verdicts)}[/cyan] alphas; "
                          f"[red]{len(hard)}[/red] hard-redundant (marked SELF_CORRELATION FAIL)")
            for v in hard:
                console.print(f"  #{v.alpha_id} corr={v.hard_corr:.3f} vs #{v.hard_ref_id}")
        finally:
            await orch.close()

    asyncio.run(_run())
```

> Note: `list_high_quality_alphas` returns rows keyed by `id` (it does `SELECT a.*`),
> while `list_refine_candidates` uses `alpha_id`. The code above accounts for both.

- [ ] **Step 2: Smoke-check the command registers**

Run: `PYTHONPATH=src /home/agentuser/wq-agent/.venv/bin/python -m alphagen_agent.cli screen-corr --help`
Expected: usage text for `screen-corr` prints, exit 0

- [ ] **Step 3: Run full suite + lint**

Run: `PYTHONPATH=src /home/agentuser/wq-agent/.venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: ALL PASS

Run: `/home/agentuser/wq-agent/.venv/bin/ruff check src/alphagen_agent/engine/correlation.py src/alphagen_agent/cli.py src/alphagen_agent/db.py src/alphagen_agent/wq/client.py src/alphagen_agent/config.py scripts/batch_produce.py tests/test_correlation.py`
Expected: All checks passed

- [ ] **Step 4: Commit**

```bash
git add src/alphagen_agent/cli.py
git commit -m "feat(corr): add screen-corr CLI command"
```

---

## Task 11: Live verification (manual, when WQ is stable)

> This task needs a live WQ account and a real `wq_alpha_id`; do it when WQ is responsive.

- [ ] **Step 1: Confirm the PnL endpoint shape**

Run a one-off (replace `<WQID>` with a real submitted alpha's wq id, e.g. #606's):

```bash
cd /home/agentuser/wq-agent
PYTHONPATH=src .venv/bin/python - <<'PY'
import asyncio
from alphagen_agent.config import get_settings
from alphagen_agent.wq.client import WQClient
async def main():
    wq = WQClient(get_settings()); await wq.connect()
    print(await wq.get_pnl("<WQID>"))   # expect (dates, daily-returns) or None
    await wq.close()
asyncio.run(main())
PY
```

Expected: a `(dates, returns)` tuple with hundreds of daily points. If `None` or wrong shape, inspect the raw response and adjust `parse_pnl_response` (Task 4), then re-run Task 4 tests.

- [ ] **Step 2: Re-screen the known case**

Run: `PYTHONPATH=src .venv/bin/python -m alphagen_agent.cli screen-corr --scope all -v`
Expected: `#617`-style siblings of submitted `#606` get flagged hard-redundant (corr ≈ 0.9 vs #606).

- [ ] **Step 3 (no commit — verification only).**

---

## Self-Review Notes

- **Spec coverage:** settings (T1), PnL cache (T2), reference set submitted+HIGH (T3), get_pnl+parser (T4), pearson/align (T5), max_correlation + WQ-mirroring hard rule (T6), lazy cache (T7), screen writes FAIL + soft flag (T8), refine-filter wiring (T9), submit-gate via screen-corr + reuse of submittable's existing self_corr filter (T10), live endpoint verification (T11). All spec sections covered.
- **Fail-open:** ensure_pnl returns None on fetch failure (T7); screen emits a non-redundant Verdict when PnL missing (T8); is_hard_redundant returns False when ref/sharpe missing (T6).
- **Reuse:** hard hits write SELF_CORRELATION FAIL via existing `update_backtest_checks`; existing `list_submittable_alphas` / `get_blacklisted_skeletons` / `batch_produce._is_redundant` consume it unchanged.

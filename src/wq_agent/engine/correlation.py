from __future__ import annotations

from dataclasses import dataclass

from loguru import logger


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
        if abs(corr) > abs(best_corr) or (abs(corr) == abs(best_corr) and corr > best_corr):
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

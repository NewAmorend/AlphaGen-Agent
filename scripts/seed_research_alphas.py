"""注入论文驱动的种子 alpha 进 DB 并跑回测。

来源（每条都对应已发表/SSRN 工作论文的实证发现）：

1. Goodwill-to-cap（Hou-Xue-Zhang 类）—— 高 goodwill 相对于市值预示负收益（潜在 impairment）
   - "The invisible burden"（Journal of Behavioral Finance）
   - "The Impact of Goodwill on Stock Returns"（Alpha Architect）

2. Goodwill momentum reversal —— 近期 goodwill 大幅增长 → M&A 激进 → 后续负收益
   - 同上一组文献

3. Operating lease intensity（ASC 842 hidden leverage）—— 高经营租赁强度 = 隐藏杠杆，
   equity beta +5%, 信用评级 -1%
   - "The Effects of ASC 842 on Value and Risk Relevance"（European Accounting Review 2025）
   - "Variable Leases Under ASC 842"（Review of Accounting Studies 2025）

4. DTA valuation allowance ratio —— 高 valuation allowance / 总 DTA 比例 = 公司对自身
   未来盈利能力缺乏信心，预示弱后续收益
   - ASU 2023-09 expanded tax footnote disclosures（2024 起强制）

5. Net DTA momentum —— DTA 净额突增常源于 NOL 累积（亏损递延），是盈利质量恶化信号

6. Lease growth signal —— 资本租赁义务突增 = 隐式 capex，类似 asset growth 异象

7. Goodwill estimate dispersion —— 分析师对 goodwill 估计的分歧大 = 不确定性溢价

8. Operating lease long-tail —— 5 年后租赁义务占比，反映长期承诺压力

每条都用 PROVEN_WRAPPERS 已验证的 ts_decay_linear(rank(...), N) 外壳包装提 sharpe 降 turnover。
"""

from __future__ import annotations

import asyncio
import sys

# 让脚本能从 src/ 找到 wq_agent 包
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wq_agent.config import get_settings
from wq_agent.db import Database
from wq_agent.models import AlphaRecord, AlphaStatus, GenerationStrategy
from wq_agent.wq.client import WQClient
from wq_agent.engine.backtest import BacktestEngine


RESEARCH_SEED_ALPHAS: list[tuple[str, str]] = [
    # (description, expression)
    (
        "Goodwill/cap reversal (Hou-Xue-Zhang style)",
        "ts_decay_linear(rank(reverse(divide(anl4_fs_actual_1qf_v4_nd_totgw_value, cap))), 20)",
    ),
    (
        "Goodwill 1y growth reversal (M&A overconfidence)",
        "ts_decay_linear(rank(reverse(ts_delta(anl4_fs_actual_1qf_v4_nd_totgw_value, 252))), 20)",
    ),
    (
        "Operating lease rent / cap (ASC 842 hidden leverage)",
        "ts_decay_linear(rank(reverse(ts_zscore(divide(fn_op_lease_rent_exp_a, cap), 252))), 20)",
    ),
    (
        "5y operating lease commitment / cap (long-tail lease)",
        "ts_decay_linear(rank(reverse(ts_zscore(divide(fn_op_lease_min_pay_due_after_5y_a, cap), 252))), 20)",
    ),
    (
        "DTA valuation allowance ratio (earnings quality)",
        "ts_decay_linear(rank(reverse(divide(deferred_tax_assets_valuation_allowance_value, deferred_tax_assets_net_value))), 20)",
    ),
    (
        "Net DTA momentum (NOL accumulation signal)",
        "ts_decay_linear(rank(reverse(ts_delta(net_deferred_tax_asset_value, 252))), 20)",
    ),
    (
        "Capital lease growth (asset-growth analog)",
        "ts_decay_linear(rank(reverse(ts_delta(capital_lease_obligation_total, 252))), 20)",
    ),
    (
        "Analyst goodwill estimate dispersion (uncertainty)",
        "ts_decay_linear(rank(divide(anl4_fs_detail_estimate_1qf_v4_nd_totgw_high, anl4_fs_detail_estimate_1qf_v4_nd_totgw_low)), 20)",
    ),
]


async def main():
    settings = get_settings()
    db = Database(settings.DB_PATH)
    await db.connect()
    wq = WQClient(settings)
    await wq.connect()
    engine = BacktestEngine(wq, db, settings)
    try:
        print(f"=== Inserting {len(RESEARCH_SEED_ALPHAS)} research-driven seed alphas ===\n")
        records = [
            AlphaRecord(
                expression=expr,
                strategy=GenerationStrategy.LLM,
                llm_model=f"research:{desc[:40]}",
                status=AlphaStatus.GENERATED,
            )
            for desc, expr in RESEARCH_SEED_ALPHAS
        ]
        ids = await db.batch_insert_alphas(records)
        for (desc, expr), rid in zip(RESEARCH_SEED_ALPHAS, ids):
            print(f"  #{rid}: {desc}")
            print(f"        {expr}")
        print()

        print(f"=== Backtesting {len(ids)} seed alphas (max_concurrent={settings.WQ_MAX_CONCURRENT}) ===\n")
        results = await engine.backtest_batch(ids)
        print(f"\n=== Results ===")
        for r in results:
            if r is None:
                continue
            fit = r.fitness if r.fitness is not None else float('nan')
            shp = r.sharpe if r.sharpe is not None else float('nan')
            tov = r.turnover if r.turnover is not None else float('nan')
            grade = r.grade.value if r.grade else 'unknown'
            print(f"  #{r.alpha_id} fit={fit:.3f} sharpe={shp:.3f} turnover={tov:.3f}  grade={grade}")
    finally:
        await wq.close()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())

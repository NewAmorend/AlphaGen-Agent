"""v2：基于 v1 的 8 条结果反思后调整方向。

v1 观察：
  - 4 条（471/473/474/477）负 sharpe → paper 方向在 USA TOP3000 反了，**去掉 reverse**
  - turnover 0.013-0.036 极低 → 季报数据本身不动 → 价量复合拉起活性
  - #475 DTA valuation allowance ratio 异常 → 分母 deferred_tax_assets_net_value 可能 0 导致 divide 失败

v2 8 条调整方向：

A) 翻 reverse 测方向 (1-4)：去掉 reverse 看 paper 信号在 TOP3000 上是否反向
B) 短窗口 + 价量复合 (5-7)：用 60d 而非 252d，且乘上价量信号借 turnover
C) 换分母 (8)：DTA valuation allowance / cap，避开 v1 异常的 net_value 分母

如果 v2 的 A 组真翻成正，说明 paper 信号在 TOP3000 上方向真的反了——这是有用的新发现，
也说明今后从 paper 拿信号要先在小样本测方向再决定 sign。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alphagen_agent.config import get_settings
from alphagen_agent.db import Database
from alphagen_agent.models import AlphaRecord, AlphaStatus, GenerationStrategy
from alphagen_agent.wq.client import WQClient
from alphagen_agent.engine.backtest import BacktestEngine


RESEARCH_SEED_ALPHAS_V2: list[tuple[str, str]] = [
    # A 组：翻 reverse 测方向
    (
        "[v2-A] Goodwill/cap FORWARD (was -0.11, flip)",
        "ts_decay_linear(rank(divide(anl4_fs_actual_1qf_v4_nd_totgw_value, cap)), 20)",
    ),
    (
        "[v2-A] Op lease rent / cap FORWARD (was -0.18, flip)",
        "ts_decay_linear(rank(ts_zscore(divide(fn_op_lease_rent_exp_a, cap), 252)), 20)",
    ),
    (
        "[v2-A] 5y op lease / cap FORWARD (was -0.17, flip)",
        "ts_decay_linear(rank(ts_zscore(divide(fn_op_lease_min_pay_due_after_5y_a, cap), 252)), 20)",
    ),
    (
        "[v2-A] Capital lease growth FORWARD (was -0.02, flip)",
        "ts_decay_linear(rank(ts_delta(capital_lease_obligation_total, 252)), 20)",
    ),
    # B 组：短窗口 + 价量复合
    (
        "[v2-B] Goodwill momentum 60d * price momentum",
        "ts_decay_linear(multiply(rank(ts_delta(anl4_fs_actual_1qf_v4_nd_totgw_value, 60)), rank(ts_delta(close, 20))), 10)",
    ),
    (
        "[v2-B] Net DTA momentum 60d * volume ratio",
        "ts_decay_linear(multiply(rank(ts_delta(net_deferred_tax_asset_value, 60)), rank(divide(volume, adv20))), 10)",
    ),
    (
        "[v2-B] Goodwill dispersion * price reversal",
        "ts_decay_linear(multiply(rank(divide(anl4_fs_detail_estimate_1qf_v4_nd_totgw_high, anl4_fs_detail_estimate_1qf_v4_nd_totgw_low)), rank(reverse(ts_delta(close, 5)))), 10)",
    ),
    # C 组：换分母（避开 v1 异常字段）
    (
        "[v2-C] DTA valuation allowance / cap (alt denom)",
        "ts_decay_linear(rank(reverse(divide(deferred_tax_assets_valuation_allowance_value, cap))), 20)",
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
        print(f"=== v2: Inserting {len(RESEARCH_SEED_ALPHAS_V2)} research-driven seed alphas ===\n")
        records = [
            AlphaRecord(
                expression=expr,
                strategy=GenerationStrategy.LLM,
                llm_model=f"research-v2:{desc[:40]}",
                status=AlphaStatus.GENERATED,
            )
            for desc, expr in RESEARCH_SEED_ALPHAS_V2
        ]
        ids = await db.batch_insert_alphas(records)
        for (desc, expr), rid in zip(RESEARCH_SEED_ALPHAS_V2, ids):
            print(f"  #{rid}: {desc}")
            print(f"        {expr}")
        print()

        print(f"=== v2: Backtesting {len(ids)} seed alphas ===\n")
        results = await engine.backtest_batch(ids)
        print("\n=== v2 Results ===")
        for r in results:
            if r is None:
                continue
            fit = r.fitness if r.fitness is not None else float('nan')
            shp = r.sharpe if r.sharpe is not None else float('nan')
            tov = r.turnover if r.turnover is not None else float('nan')
            grade = r.grade.value if r.grade else 'unknown'
            print(f"  #{r.alpha_id} fit={fit:+.3f} sharpe={shp:+.3f} turnover={tov:.3f}  grade={grade}")
    finally:
        await wq.close()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())

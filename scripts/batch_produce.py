"""批量生产因子 → 对非重复的 MEDIUM/近失因子做 refine。

一个进程内：
  1. 跑 N 批 generate+backtest（每批都吃前批反馈 + 去重/多样性 steering）
  2. 收集 MEDIUM + 高 fitness 的 LOW（近失候选），**剔除 self_correlation FAIL 的重复因子**
  3. 对每个候选 refine 出变体并回测（去重过滤同样生效）
  4. 打印 HIGH / 可提交 / 仍待精修 的汇总

用法:
  python scripts/batch_produce.py --batches 6 --count 10 --variants 10 --refine-passes 1
"""
from __future__ import annotations

import argparse
import asyncio

from wq_agent.agent.orchestrator import Orchestrator
from wq_agent.models import GenerationStrategy


def _is_redundant(cand: dict) -> bool:
    """self_correlation FAIL = 与已有 alpha 高度相关 = 重复，refine 也会被 WQ 拒。"""
    return any("SELF_CORRELATION" in str(c).upper() for c in cand.get("failed_checks", []))


async def main(batches: int, count: int, variants: int, refine_passes: int) -> None:
    orch = Orchestrator()
    await orch.initialize()
    try:
        # ---- Phase 1: 批量生产 ----
        # 单批失败（LLM 超时 / WQ 抖动）不应拖垮整轮——记录并继续下一批。
        for b in range(1, batches + 1):
            print(f"\n########## PRODUCE batch {b}/{batches} ##########", flush=True)
            try:
                await orch.run(strategy=GenerationStrategy.LLM, count=count, auto_backtest=True)
            except Exception as exc:
                print(f"  [batch {b} FAILED, skipping] {type(exc).__name__}: {exc}", flush=True)

        # ---- Phase 2: 对非重近失因子 refine ----
        refined: set[int] = set()
        from wq_agent.engine.correlation import CorrelationScreener
        screener = CorrelationScreener(orch.db, orch.wq, orch.settings)
        for p in range(1, refine_passes + 1):
            cands = await orch.db.list_refine_candidates(limit=100)
            # 先用 PnL 相关性筛一遍——命中硬 gate 的会被写 SELF_CORRELATION FAIL，
            # 下面的 _is_redundant 立刻就能把它们挡掉（复用现有约定）。
            await screener.screen([c["alpha_id"] for c in cands])
            cands = await orch.db.list_refine_candidates(limit=100)  # 重新读，拿到刚写的 FAIL
            todo = [c for c in cands if c["alpha_id"] not in refined and not _is_redundant(c)]
            skipped = [c for c in cands if c["alpha_id"] not in refined and _is_redundant(c)]
            print(
                f"\n########## REFINE pass {p}/{refine_passes}: "
                f"{len(todo)} non-redundant candidates "
                f"({len(skipped)} redundant skipped) ##########",
                flush=True,
            )
            if not todo:
                break
            for c in todo:
                refined.add(c["alpha_id"])
                fit = c.get("fitness")
                print(
                    f"  → refine #{c['alpha_id']} grade={c.get('grade')} "
                    f"fitness={fit:.3f} failed={c.get('failed_checks')}",
                    flush=True,
                )
                try:
                    await orch.refine(base_id=c["alpha_id"], count=variants, auto_backtest=True)
                except Exception as exc:
                    print(f"    [refine #{c['alpha_id']} FAILED, skipping] {type(exc).__name__}: {exc}", flush=True)

        # ---- Phase 3: 汇总 ----
        stats = await orch.db.get_stats()
        submittable = await orch.db.list_submittable_alphas(min_fitness=orch.settings.MIN_FITNESS, limit=50)
        remaining = await orch.db.list_refine_candidates(limit=100)
        print("\n########## SUMMARY ##########", flush=True)
        for k, v in stats.items():
            print(f"  {k}: {v}", flush=True)
        print(f"  submittable (HIGH, fitness>={orch.settings.MIN_FITNESS}, not submitted): {len(submittable)}", flush=True)
        for s in submittable[:20]:
            print(f"    #{s['alpha_id']} fit={s['fitness']:.3f} {s['expression'][:70]}", flush=True)
        print(f"  still-refinable near-miss remaining: {len(remaining)}", flush=True)
    finally:
        await orch.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--variants", type=int, default=10)
    ap.add_argument("--refine-passes", type=int, default=1)
    args = ap.parse_args()
    asyncio.run(main(args.batches, args.count, args.variants, args.refine_passes))

from __future__ import annotations

import asyncio
import re

from loguru import logger

from ..config import Settings
from ..db import Database
from ..models import AlphaRecord, AlphaStatus, BacktestResult, QualityGrade
from ..wq.client import WQClient
from .evaluator import AlphaEvaluator


# WQ Brain 算子名 / 关键字（不算字段）
_NOT_FIELD = frozenset({
    # 算术 / 逻辑
    "abs", "add", "and", "or", "not", "if_else", "is_nan", "to_nan", "sign",
    "log", "sqrt", "power", "signed_power", "max", "min", "divide", "multiply",
    "subtract", "inverse", "reverse", "densify",
    # 时间序列
    "ts_mean", "ts_std_dev", "ts_zscore", "ts_rank", "ts_delta", "ts_delay",
    "ts_sum", "ts_min", "ts_max", "ts_corr", "ts_covariance", "ts_decay_linear",
    "ts_regression", "ts_scale", "ts_quantile", "ts_av_diff", "ts_backfill",
    "ts_count_nans", "ts_arg_max", "ts_arg_min", "ts_product", "ts_step",
    "hump", "jump_decay", "days_from_last_change", "kth_element", "last_diff_value",
    # 横截面
    "rank", "zscore", "normalize", "quantile", "scale", "scale_down", "winsorize",
    "vector_neut",
    # 分组 / 向量
    "group_neutralize", "group_rank", "group_zscore", "group_mean", "group_min",
    "group_max", "group_scale", "group_backfill", "group_cartesian_product",
    "vec_avg", "vec_max", "vec_min", "vec_sum",
    # 变换 / 归约
    "bucket", "trade_when", "generate_stats", "combo_a", "self_corr",
    "reduce_avg", "reduce_choose", "reduce_count", "reduce_ir", "reduce_kurtosis",
    "reduce_max", "reduce_min", "reduce_norm", "reduce_percentage", "reduce_powersum",
    "reduce_range", "reduce_skewness", "reduce_stddev", "reduce_sum",
    # 关键字
    "true", "false", "in", "universe_size",
    # 常见分组字段（虽然是字段但属于 universe 必备，不该 blacklist）
    "subindustry", "industry", "sector", "country", "currency", "exchange",
    # 算子的关键字参数名 / 字面值（不是字段！）
    "driver", "sigma", "std", "range", "buckets", "lookback", "ignore", "dense",
    "sensitivity", "force", "constant", "lambda_min", "lambda_max", "target_tvr",
    "use_std", "usestd", "limit", "rate", "longscale", "shortscale", "mode",
    "nlength", "threshold", "nth", "ignorenan", "percentage", "precise",
    "filter", "value", "decay", "hump",
    "gaussian", "uniform", "nan", "algo1", "algo2",
    # 单字母 / 极短变量（绝大多数是数学公式里的占位）
    "x", "y", "z", "d", "k", "n",
})

_IDENT_RE = re.compile(r"\b[a-z][a-z0-9_]{2,}\b")


def extract_field_candidates(expression: str) -> list[str]:
    """从表达式里挑出疑似字段 ID（去掉算子名和数字）。"""
    return sorted({m for m in _IDENT_RE.findall(expression) if m not in _NOT_FIELD})


class BacktestEngine:
    def __init__(self, wq: WQClient, db: Database, settings: Settings, region_override: str | None = None):
        self.wq = wq
        self.db = db
        self.settings = settings
        self.region_override = region_override
        self.evaluator = AlphaEvaluator(settings)

    async def backtest_batch(
        self,
        alpha_ids: list[int],
        max_concurrent: int | None = None,
    ) -> list[BacktestResult]:
        semaphore = asyncio.Semaphore(max_concurrent or self.settings.WQ_MAX_CONCURRENT)
        results: list[BacktestResult] = []

        async def _run_one(alpha_id: int) -> BacktestResult | None:
            async with semaphore:
                alpha = await self.db.get_alpha(alpha_id)
                if not alpha:
                    logger.warning(f"Alpha {alpha_id} not found")
                    return None
                return await self._backtest_single(alpha)

        tasks = [_run_one(aid) for aid in alpha_ids]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for item in completed:
            if isinstance(item, Exception):
                logger.error(f"Backtest error: {item}")
            elif item is not None:
                results.append(item)

        return results

    async def backtest_expressions(
        self,
        expressions: list[tuple[int, str]],
        max_concurrent: int | None = None,
    ) -> list[BacktestResult]:
        semaphore = asyncio.Semaphore(max_concurrent or self.settings.WQ_MAX_CONCURRENT)
        results: list[BacktestResult] = []

        async def _run_one(alpha_id: int, expression: str) -> BacktestResult | None:
            async with semaphore:
                return await self._backtest_expression(alpha_id, expression)

        tasks = [_run_one(aid, expr) for aid, expr in expressions]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for item in completed:
            if isinstance(item, Exception):
                logger.error(f"Backtest error: {item}")
            elif item is not None:
                results.append(item)

        return results

    async def _backtest_single(self, alpha: AlphaRecord) -> BacktestResult | None:
        await self.db.update_alpha_status(alpha.id, AlphaStatus.BACKTESTING)
        try:
            result = await self._backtest_expression(alpha.id, alpha.expression)
        except Exception as e:
            # Retry-exhausted / network exception——之前会让 alpha 永远卡在 BACKTESTING，
            # 下一轮 reset_stuck_backtesting 才会回收。这里直接 mark failed，但**不**牵连
            # 表达式里的字段进黑名单（网络错误和字段无关，字段级别失败走 _backtest_expression
            # 内的 _record_failure 分支）。
            logger.error(f"Backtest exception for alpha {alpha.id}: {e}")
            await self.db.update_alpha_status(alpha.id, AlphaStatus.FAILED)
            return None
        if result and result.grade == QualityGrade.HIGH:
            await self.db.update_alpha_status(alpha.id, AlphaStatus.HIGH_QUALITY)
        elif result and result.grade != QualityGrade.REJECT:
            await self.db.update_alpha_status(alpha.id, AlphaStatus.EVALUATED)
        else:
            await self.db.update_alpha_status(alpha.id, AlphaStatus.FAILED)
        return result

    async def _backtest_expression(self, alpha_id: int, expression: str) -> BacktestResult | None:
        logger.info(f"Backtesting alpha {alpha_id}: {expression[:60]}...")
        region = self.region_override or self.settings.WQ_REGION

        submit_result = await self.wq.submit_simulation(expression, region=region)
        if submit_result.get("status") == "error":
            msg = submit_result.get("message", "")
            logger.error(f"Simulation submit failed for alpha {alpha_id}: {msg}")
            await self._record_failure(expression, reason=f"submit:{msg[:50]}")
            return None

        progress_url = submit_result["progress_url"]
        poll_result = await self.wq.poll_simulation(progress_url)

        if poll_result.get("status") != "complete":
            err = str(poll_result.get("message", ""))[:80]
            logger.error(f"Simulation failed for alpha {alpha_id}: {err}")
            await self._record_failure(expression, reason=err or "sim_error")
            return None

        alpha_data = poll_result.get("alpha_data", {})
        is_data = alpha_data.get("is", {})
        wq_alpha_id = poll_result.get("alpha_id")

        backtest = BacktestResult(
            alpha_id=alpha_id,
            region=region,
            universe=self.settings.WQ_UNIVERSE,
            delay=self.settings.WQ_DELAY,
            neutralization=self.settings.WQ_NEUTRALIZATION,
            sharpe=is_data.get("sharpe"),
            turnover=is_data.get("turnover"),
            fitness=is_data.get("fitness"),
            returns=is_data.get("returns"),
            checks=is_data.get("checks"),
            wq_alpha_id=wq_alpha_id,
        )

        backtest.grade = self.evaluator.evaluate(backtest)
        await self.db.insert_backtest_result(backtest)

        grade_str = backtest.grade.value if backtest.grade else "unknown"
        fitness_str = f"{backtest.fitness:.4f}" if backtest.fitness is not None else "N/A"
        logger.info(f"Alpha {alpha_id}: fitness={fitness_str}, grade={grade_str}")

        return backtest

    async def _record_failure(self, expression: str, reason: str) -> None:
        """对 WQ 拒收的表达式，把里面的疑似字段记到 blacklist；多次失败的字段会被剔除。"""
        fields = extract_field_candidates(expression)
        if fields:
            await self.db.bump_field_blacklist(fields, reason=reason)

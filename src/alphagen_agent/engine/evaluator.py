from __future__ import annotations

from ..config import Settings
from ..models import BacktestResult, QualityGrade


# WQ Brain 提交检查里"卡死提交"的关键项（命中 FAIL 就不能投递到比赛）
_CRITICAL_CHECKS = {
    "LOW_SHARPE",
    "LOW_FITNESS",
    "LOW_TURNOVER",
    "HIGH_TURNOVER",
    "CONCENTRATED_WEIGHT",
    "LOW_SUB_UNIVERSE_SHARPE",
    "IS_LADDER_SHARPE",
}


class AlphaEvaluator:
    """评估器优先用 WQ Brain 自己的 checks payload；fallback 到本地阈值（默认与 WQ 对齐）。

    WQ 官方阈值（USA TOP3000, delay=1, 截至 2026-05）：
        fitness          >= 1.00
        sharpe           >= 1.25
        sub-universe     >= 0.67
        turnover         在 [0.01, 0.70] 之间
        weight           不能集中
    """

    def __init__(self, settings: Settings):
        self.min_fitness = settings.MIN_FITNESS
        self.min_sharpe = settings.MIN_SHARPE
        self.min_turnover = settings.MIN_TURNOVER
        self.max_turnover = settings.MAX_TURNOVER
        self.min_sub_universe_sharpe = settings.MIN_SUB_UNIVERSE_SHARPE
        self.min_returns = settings.MIN_RETURNS  # WQ 不卡，保留作为可选偏好

    def evaluate(self, result: BacktestResult) -> QualityGrade:
        # 优先路径：信任 WQ 自己跑出来的 checks
        if result.checks:
            grade = self._grade_from_checks(result.checks)
            if grade is not None:
                return grade

        # Fallback：本地阈值（与 WQ 默认对齐）
        return self._grade_from_thresholds(result)

    # ------------------------------------------------------------------ checks

    def _grade_from_checks(self, checks: list[dict]) -> QualityGrade | None:
        """根据 WQ 官方 checks 列表判级。返回 None 表示 checks 不可用，退到 fallback。"""
        usable = [c for c in checks if isinstance(c, dict) and c.get("result")]
        if not usable:
            return None

        fails = [c for c in usable if str(c.get("result", "")).upper() == "FAIL"]
        critical_fails = [
            c for c in fails
            if str(c.get("name", "")).upper() in _CRITICAL_CHECKS
        ]

        if not fails:
            return QualityGrade.HIGH       # WQ 全 PASS → 可提交
        if not critical_fails:
            return QualityGrade.HIGH       # 仅非关键项 FAIL（如 COMPETITION 未匹配）
        if len(critical_fails) == 1:
            return QualityGrade.MEDIUM     # 只差一项硬指标
        if len(critical_fails) <= 2:
            return QualityGrade.LOW
        return QualityGrade.REJECT

    # ----------------------------------------------------------- thresholds

    def _grade_from_thresholds(self, result: BacktestResult) -> QualityGrade:
        if result.fitness is None:
            return QualityGrade.REJECT
        fitness = result.fitness
        sharpe = result.sharpe or 0.0
        turnover = result.turnover if result.turnover is not None else 1.0

        crit_fails = 0
        if fitness < self.min_fitness:
            crit_fails += 1
        if sharpe < self.min_sharpe:
            crit_fails += 1
        if turnover < self.min_turnover or turnover > self.max_turnover:
            crit_fails += 1

        if crit_fails == 0:
            return QualityGrade.HIGH
        if crit_fails == 1:
            return QualityGrade.MEDIUM
        if crit_fails == 2:
            return QualityGrade.LOW
        return QualityGrade.REJECT

    # ----------------------------------------------------------------- filter

    def filter_high_quality(
        self,
        results: list[BacktestResult],
        min_grade: QualityGrade = QualityGrade.HIGH,
    ) -> list[BacktestResult]:
        grade_order = {
            QualityGrade.HIGH: 4,
            QualityGrade.MEDIUM: 3,
            QualityGrade.LOW: 2,
            QualityGrade.REJECT: 1,
        }
        min_level = grade_order[min_grade]
        return [r for r in results if r.grade and grade_order.get(r.grade, 0) >= min_level]

from __future__ import annotations

from ..checks import critical_check_failures
from ..config import Settings
from ..models import BacktestResult, QualityGrade

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
        # 优先路径：使用 WQ checks，同时仍强制执行本地硬指标。WQ 在不同
        # 权限/活动上下文中可能省略检查，或把硬指标未达标标成 WARNING。
        if result.checks:
            grade = self._grade_from_checks(result.checks, result)
            if grade is not None:
                return grade

        # Fallback：本地阈值（与 WQ 默认对齐）
        return self._grade_from_thresholds(result)

    # ------------------------------------------------------------------ checks

    def _grade_from_checks(
        self,
        checks: list[dict],
        result: BacktestResult,
    ) -> QualityGrade | None:
        """根据 WQ 官方 checks 列表判级。返回 None 表示 checks 不可用，退到 fallback。"""
        usable = [c for c in checks if isinstance(c, dict) and c.get("result")]
        if not usable:
            return None

        critical_fail_names = set(critical_check_failures(usable))

        # 即使 WQ checks 缺项或状态语义发生变化，也不能绕过核心数值门槛。
        if result.fitness is None or result.fitness < self.min_fitness:
            critical_fail_names.add("LOW_FITNESS")
        if (result.sharpe or 0.0) < self.min_sharpe:
            critical_fail_names.add("LOW_SHARPE")
        turnover = result.turnover if result.turnover is not None else 1.0
        if turnover < self.min_turnover:
            critical_fail_names.add("LOW_TURNOVER")
        elif turnover > self.max_turnover:
            critical_fail_names.add("HIGH_TURNOVER")

        if not critical_fail_names:
            return QualityGrade.HIGH       # 所有关键项通过；非关键项不影响可提交性
        if len(critical_fail_names) == 1:
            return QualityGrade.MEDIUM     # 只差一项硬指标
        if len(critical_fail_names) <= 2:
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

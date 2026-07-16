from __future__ import annotations

import pytest

from alphagen_agent.config import Settings
from alphagen_agent.engine.evaluator import AlphaEvaluator
from alphagen_agent.models import BacktestResult, QualityGrade


@pytest.fixture
def evaluator() -> AlphaEvaluator:
    # 不读 .env，避免用户本地阈值污染单测
    return AlphaEvaluator(Settings(_env_file=None))


def _result(**kw) -> BacktestResult:
    base = dict(alpha_id=1, region="USA", universe="TOP3000", delay=1, neutralization="INDUSTRY")
    base.update(kw)
    return BacktestResult(**base)


# ---------- WQ checks payload path ----------

def test_checks_all_pass_returns_high(evaluator):
    r = _result(
        fitness=2.0, sharpe=1.5, turnover=0.4,
        checks=[
            {"name": "LOW_SHARPE", "result": "PASS", "value": 1.5, "limit": 1.25},
            {"name": "LOW_FITNESS", "result": "PASS", "value": 2.0, "limit": 1.0},
            {"name": "HIGH_TURNOVER", "result": "PASS", "value": 0.4, "limit": 0.7},
            {"name": "LOW_TURNOVER", "result": "PASS", "value": 0.4, "limit": 0.01},
            {"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "PASS", "value": 1.1, "limit": 0.67},
            {"name": "CONCENTRATED_WEIGHT", "result": "PASS"},
            {"name": "MATCHES_COMPETITION", "result": "PASS"},
        ],
    )
    assert evaluator.evaluate(r) is QualityGrade.HIGH


def test_only_fitness_fails_returns_medium(evaluator):
    """User 实际遇到的场景：sharpe 1.54、turnover 60%、sub-universe 1.21 都过，仅 fitness 0.83 < 1.0。"""
    r = _result(
        fitness=0.83, sharpe=1.54, turnover=0.6015,
        checks=[
            {"name": "LOW_SHARPE", "result": "PASS", "value": 1.54, "limit": 1.25},
            {"name": "LOW_TURNOVER", "result": "PASS", "value": 0.6015, "limit": 0.01},
            {"name": "HIGH_TURNOVER", "result": "PASS", "value": 0.6015, "limit": 0.70},
            {"name": "CONCENTRATED_WEIGHT", "result": "PASS"},
            {"name": "LOW_SUB_UNIVERSE_SHARPE", "result": "PASS", "value": 1.21, "limit": 0.67},
            {"name": "MATCHES_COMPETITION", "result": "PASS"},
            {"name": "LOW_FITNESS", "result": "FAIL", "value": 0.83, "limit": 1.0},
        ],
    )
    assert evaluator.evaluate(r) is QualityGrade.MEDIUM


def test_critical_warning_returns_medium(evaluator):
    """完整顾问权限会把未达标的 LOW_FITNESS 返回为 WARNING。"""
    r = _result(
        fitness=0.70, sharpe=1.77, turnover=0.582,
        checks=[
            {"name": "LOW_SHARPE", "result": "PASS", "value": 1.77, "limit": 1.58},
            {"name": "LOW_FITNESS", "result": "WARNING", "value": 0.70, "limit": 1.0},
            {"name": "HIGH_TURNOVER", "result": "PASS", "value": 0.582, "limit": 0.7},
            {"name": "LOW_TURNOVER", "result": "PASS", "value": 0.582, "limit": 0.01},
        ],
    )
    assert evaluator.evaluate(r) is QualityGrade.MEDIUM


def test_checks_cannot_bypass_numeric_thresholds(evaluator):
    """WQ checks 缺少硬指标项时，本地门槛仍然生效。"""
    r = _result(
        fitness=0.70, sharpe=1.77, turnover=0.4,
        checks=[{"name": "MATCHES_COMPETITION", "result": "PASS"}],
    )
    assert evaluator.evaluate(r) is QualityGrade.MEDIUM


def test_two_critical_fails_returns_low(evaluator):
    r = _result(
        fitness=0.5, sharpe=0.8, turnover=0.4,
        checks=[
            {"name": "LOW_FITNESS", "result": "FAIL", "value": 0.5, "limit": 1.0},
            {"name": "LOW_SHARPE", "result": "FAIL", "value": 0.8, "limit": 1.25},
            {"name": "HIGH_TURNOVER", "result": "PASS"},
        ],
    )
    assert evaluator.evaluate(r) is QualityGrade.LOW


def test_three_critical_fails_returns_reject(evaluator):
    r = _result(
        fitness=0.1, sharpe=0.2, turnover=0.95,
        checks=[
            {"name": "LOW_FITNESS", "result": "FAIL"},
            {"name": "LOW_SHARPE", "result": "FAIL"},
            {"name": "HIGH_TURNOVER", "result": "FAIL"},
        ],
    )
    assert evaluator.evaluate(r) is QualityGrade.REJECT


def test_only_non_critical_fails_still_high(evaluator):
    """COMPETITION_CHALLENGE 这种非提交门槛项 FAIL 时仍判 HIGH（可提交）。"""
    r = _result(
        fitness=2.0, sharpe=1.5, turnover=0.5,
        checks=[
            {"name": "LOW_FITNESS", "result": "PASS"},
            {"name": "LOW_SHARPE", "result": "PASS"},
            {"name": "MATCHES_COMPETITION", "result": "FAIL"},
        ],
    )
    assert evaluator.evaluate(r) is QualityGrade.HIGH


def test_only_non_critical_warnings_still_high(evaluator):
    r = _result(
        fitness=2.0, sharpe=1.5, turnover=0.5,
        checks=[
            {"name": "LOW_FITNESS", "result": "PASS"},
            {"name": "LOW_SHARPE", "result": "PASS"},
            {"name": "OSMOSIS_ALLOCATION", "result": "WARNING"},
        ],
    )
    assert evaluator.evaluate(r) is QualityGrade.HIGH


# ---------- Fallback threshold path (checks 缺失或不可用) ----------

def test_fallback_threshold_path_when_checks_missing(evaluator):
    r = _result(fitness=1.5, sharpe=1.3, turnover=0.4, checks=None)
    assert evaluator.evaluate(r) is QualityGrade.HIGH


def test_fallback_uses_wq_aligned_defaults(evaluator):
    # fitness 0.83 < 1.0 (WQ 默认) → 一项失败 → MEDIUM
    r = _result(fitness=0.83, sharpe=1.54, turnover=0.6, checks=None)
    assert evaluator.evaluate(r) is QualityGrade.MEDIUM
    # sharpe 1.0 < 1.25 (WQ 默认) → 两项失败 → LOW
    r2 = _result(fitness=0.5, sharpe=1.0, turnover=0.5, checks=None)
    assert evaluator.evaluate(r2) is QualityGrade.LOW


def test_fallback_turnover_lower_bound_now_enforced(evaluator):
    """旧版没下界，turnover=0 也算合法；WQ 实际要求 >=1%。"""
    r = _result(fitness=1.5, sharpe=1.5, turnover=0.005, checks=None)
    assert evaluator.evaluate(r) is QualityGrade.MEDIUM


def test_empty_checks_list_falls_back_to_thresholds(evaluator):
    r = _result(fitness=1.5, sharpe=1.3, turnover=0.4, checks=[])
    assert evaluator.evaluate(r) is QualityGrade.HIGH


def test_missing_fitness_rejects(evaluator):
    r = _result(fitness=None, sharpe=1.5, turnover=0.4)
    assert evaluator.evaluate(r) is QualityGrade.REJECT

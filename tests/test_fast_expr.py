from __future__ import annotations

from alphagen_agent.engine.fast_expr import validate_fast_expr
from alphagen_agent.generator.llm import LLMAlphaGenerator
from alphagen_agent.llm.base import BaseLLMProvider


class _DummyLLM(BaseLLMProvider):
    async def generate(self, *a, **k):
        return ""

    async def close(self):
        return None


def test_fast_expr_accepts_nested_valid_expression():
    result = validate_fast_expr(
        "group_neutralize(rank(ts_delta(fnd6_assets, 60)), subindustry)",
        field_ids={"fnd6_assets"},
        max_depth=4,
    )
    assert result.valid


def test_fast_expr_rejects_unknown_function():
    result = validate_fast_expr("ts_std(fnd6_assets, 20)", field_ids={"fnd6_assets"})
    assert not result.valid
    assert any(issue.code == "unknown_function" for issue in result.issues)


def test_fast_expr_rejects_wrong_arity():
    result = validate_fast_expr("ts_corr(fnd6_assets, 20)", field_ids={"fnd6_assets"})
    assert not result.valid
    assert any(issue.code == "too_few_args" for issue in result.issues)


def test_fast_expr_rejects_unknown_field_when_pool_provided():
    result = validate_fast_expr("rank(not_in_pool)", field_ids={"fnd6_assets"})
    assert not result.valid
    assert any(issue.code == "unknown_identifier" for issue in result.issues)


def test_fast_expr_allows_kwargs_and_literal_values():
    result = validate_fast_expr(
        "quantile(rank(fnd6_assets), driver=gaussian, sigma=1.0)",
        field_ids={"fnd6_assets"},
    )
    assert result.valid


def test_llm_clean_expressions_uses_fast_expr_validation():
    gen = LLMAlphaGenerator(_DummyLLM())
    cleaned = gen._clean_expressions(
        [
            "rank(fnd6_assets)",
            "ts_std(fnd6_assets, 20)",
            "rank(not_in_pool)",
            "ts_corr(fnd6_assets, fnd6_assets, 20)",
        ],
        valid_field_ids={"fnd6_assets"},
    )
    assert cleaned == [
        "rank(fnd6_assets)",
        "ts_corr(fnd6_assets, fnd6_assets, 20)",
    ]

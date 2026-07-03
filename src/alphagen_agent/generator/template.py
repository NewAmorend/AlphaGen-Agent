from __future__ import annotations

import random
from typing import Any

from loguru import logger

from ..models import WQDataField, WQOperator
from .base import BaseAlphaGenerator

_TEMPLATES = [
    {"pattern": "rank(ts_delta({field}, {period}))", "periods": [5, 10, 20], "desc": "momentum_rank"},
    {"pattern": "ts_zscore({field}, {period})", "periods": [20, 60, 120], "desc": "mean_reversion"},
    {"pattern": "rank({field}) - rank(ts_delay({field}, {period}))", "periods": [5, 10, 20], "desc": "change_rank"},
    {"pattern": "ts_rank({field}, {period})", "periods": [10, 20, 60], "desc": "ts_rank"},
    {"pattern": "divide(ts_mean({field}, {period}), ts_std_dev({field}, {period}))", "periods": [20, 60], "desc": "sharpe_like"},
    {"pattern": "ts_decay_linear({field}, {period})", "periods": [10, 20, 60], "desc": "decay_linear"},
    {"pattern": "rank(ts_corr({field}, ts_delay({field}, 1), {period}))", "periods": [20, 60], "desc": "autocorr"},
    {"pattern": "zscore(ts_mean({field}, {period}))", "periods": [20, 60], "desc": "normalized_mean"},
    {"pattern": "group_neutralize(rank({field}), sector)", "periods": [], "desc": "sector_neutral"},
    {"pattern": "ts_regression({field}, ts_delay({field}, 1), {period}, 0, 1)", "periods": [20, 60], "desc": "regression_residual"},
    {"pattern": "rank(ts_av_diff({field}, {period}))", "periods": [10, 20, 60], "desc": "av_diff"},
    {"pattern": "scale(rank(ts_delta({field}, {period})))", "periods": [5, 10, 20], "desc": "scaled_momentum"},
]


class TemplateAlphaGenerator(BaseAlphaGenerator):
    def __init__(self, max_fields: int = 30):
        self.max_fields = max_fields

    async def generate(
        self,
        data_fields: list[WQDataField],
        operators: list[WQOperator],
        previous_results: list[dict[str, Any]] | None = None,
        count: int = 18,
        forbidden_fields: list[dict[str, Any]] | None = None,
        high_fitness_exemplars: list[dict[str, Any]] | None = None,
        submitted_skeletons: set[str] | None = None,
        extra_exclude_skeletons: set[str] | None = None,
        family_distribution: dict[str, Any] | None = None,
        user_idea: str | None = None,
    ) -> list[str]:
        selected_fields = random.sample(data_fields, min(self.max_fields, len(data_fields)))
        expressions: list[str] = []

        candidates: list[tuple[str, str, int | None]] = []
        for field in selected_fields:
            for tmpl in _TEMPLATES:
                if tmpl["periods"]:
                    for p in tmpl["periods"]:
                        candidates.append((tmpl["pattern"], field.id, p))
                else:
                    candidates.append((tmpl["pattern"], field.id, None))

        random.shuffle(candidates)
        for pattern, field_id, period in candidates:
            if len(expressions) >= count:
                break
            expr = pattern.replace("{field}", field_id)
            if period is not None:
                expr = expr.replace("{period}", str(period))
            expressions.append(expr)

        logger.info(f"Template generator produced {len(expressions)} expressions")
        return expressions[:count]

from __future__ import annotations

import random
from typing import Any

from loguru import logger

from ..models import WQDataField, WQOperator
from .base import BaseAlphaGenerator

_TS_OPERATORS = [
    "ts_mean", "ts_std_dev", "ts_zscore", "ts_rank", "ts_delta",
    "ts_delay", "ts_sum", "ts_min", "ts_max", "ts_decay_linear",
    "ts_corr", "ts_av_diff",
]

_CS_OPERATORS = ["rank", "zscore", "normalize", "scale"]

_BINARY_TS = {"ts_corr", "ts_covariance", "ts_regression"}

_PERIODS = [5, 10, 20, 40, 60, 120]

_MAX_DEPTH = 3


class FactorMiningGenerator(BaseAlphaGenerator):
    def __init__(self, max_fields: int = 20, max_depth: int = _MAX_DEPTH):
        self.max_fields = max_fields
        self.max_depth = max_depth

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

        for _ in range(count * 3):
            if len(expressions) >= count:
                break
            depth = random.randint(1, self.max_depth)
            expr = self._build_expression(selected_fields, depth)
            if expr and expr not in expressions:
                expressions.append(expr)

        logger.info(f"Factor mining generator produced {len(expressions)} expressions")
        return expressions[:count]

    def _build_expression(self, fields: list[WQDataField], depth: int) -> str:
        field_id = random.choice(fields).id

        if depth == 1:
            op = random.choice(_TS_OPERATORS)
            period = random.choice(_PERIODS)
            if op in _BINARY_TS:
                field2 = random.choice(fields).id
                return f"{op}({field_id}, {field2}, {period})"
            return f"{op}({field_id}, {period})"

        if depth == 2:
            ts_op = random.choice(_TS_OPERATORS)
            cs_op = random.choice(_CS_OPERATORS)
            period = random.choice(_PERIODS)
            inner = f"{ts_op}({field_id}, {period})"
            return f"{cs_op}({inner})"

        ts_op1 = random.choice(_TS_OPERATORS)
        ts_op2 = random.choice(_TS_OPERATORS)
        cs_op = random.choice(_CS_OPERATORS)
        p1 = random.choice(_PERIODS)
        p2 = random.choice(_PERIODS)

        if ts_op1 in _BINARY_TS:
            field2 = random.choice(fields).id
            inner = f"{ts_op1}({field_id}, {field2}, {p1})"
        else:
            inner = f"{ts_op1}({field_id}, {p1})"

        if ts_op2 in _BINARY_TS:
            field2 = random.choice(fields).id
            mid = f"{ts_op2}({inner}, {field2}, {p2})"
        else:
            mid = f"{ts_op2}({inner}, {p2})"

        return f"{cs_op}({mid})"

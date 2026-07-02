from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..models import WQDataField, WQOperator


class BaseAlphaGenerator(ABC):
    @abstractmethod
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
        ...

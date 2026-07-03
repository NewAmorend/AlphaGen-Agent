from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from alphagen_agent.models import AlphaRecord, BacktestResult, GenerationStrategy, QualityGrade
from alphagen_agent.wiki.auto_record import AutoRecorder
from alphagen_agent.wiki.store import WikiStore


class _NoopIndex:
    async def build(self, incremental: bool = False):
        return None


@pytest.mark.asyncio
async def test_record_writes_entries_and_lessons(tmp_path: Path):
    root = tmp_path / "wiki"
    root.mkdir()
    store = WikiStore(root)
    recorder = AutoRecorder(store=store, index=_NoopIndex())

    records = [
        AlphaRecord(id=1, expression="rank(ts_delta(close, 5))", strategy=GenerationStrategy.LLM,
                    llm_model="kimi", created_at=datetime.now()),
        AlphaRecord(id=2, expression="ts_zscore(volume, 60)", strategy=GenerationStrategy.LLM,
                    llm_model="kimi", created_at=datetime.now()),
        AlphaRecord(id=3, expression="ts_corr(open, high, 20)", strategy=GenerationStrategy.LLM,
                    llm_model="kimi", created_at=datetime.now()),
        AlphaRecord(id=4, expression="rank(ts_mean(low, 30))", strategy=GenerationStrategy.LLM,
                    llm_model="kimi", created_at=datetime.now()),
    ]
    results = [
        BacktestResult(alpha_id=1, fitness=0.62, sharpe=1.4, turnover=0.4, returns=0.07,
                       grade=QualityGrade.HIGH),
        BacktestResult(alpha_id=2, fitness=0.55, sharpe=1.1, turnover=0.5, returns=0.06,
                       grade=QualityGrade.MEDIUM),
        BacktestResult(alpha_id=3, fitness=-0.4, sharpe=0.2, turnover=0.3, returns=-0.02,
                       grade=QualityGrade.REJECT),
        BacktestResult(alpha_id=4, fitness=0.02, sharpe=0.1, turnover=0.6, returns=0.005,
                       grade=QualityGrade.REJECT),
    ]
    stats = await recorder.record(records, results)
    assert stats == {"entries": 2, "lessons": 1}

    entry_files = list((root / "entries").glob("*.md"))
    lesson_files = list((root / "lessons").glob("*.md"))
    assert len(entry_files) == 2
    assert len(lesson_files) == 1

    lesson_text = lesson_files[0].read_text(encoding="utf-8")
    assert "negative fitness" in lesson_text
    assert "fitness 接近 0" in lesson_text

    entry_text = entry_files[0].read_text(encoding="utf-8")
    assert "fitness" in entry_text
    assert "[[ts_delta]]" in entry_text or "[[ts_zscore]]" in entry_text
    assert "[[close]]" not in entry_text
    assert "[[volume]]" not in entry_text
    assert "`close`" in entry_text or "`volume`" in entry_text
    assert "[[concepts]]" not in lesson_text
    assert "[[operators]]" not in lesson_text

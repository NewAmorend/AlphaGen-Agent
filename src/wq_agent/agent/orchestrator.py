from __future__ import annotations

from typing import Any

from loguru import logger
from rich.console import Console
from rich.table import Table

from ..config import Settings, get_settings
from ..db import Database
from ..models import (
    AlphaRecord,
    BacktestResult,
    GenerationStrategy,
    QualityGrade,
)
from ..wq.client import WQClient
from ..llm import LLMFactory
from ..llm.base import BaseLLMProvider
from ..generator.llm import LLMAlphaGenerator
from ..generator.template import TemplateAlphaGenerator
from ..generator.factor import FactorMiningGenerator
from ..generator.base import BaseAlphaGenerator
from ..engine.backtest import BacktestEngine
from ..wiki.store import WikiStore
from ..wiki.index import WikiIndex
from ..wiki.embeddings import BaseEmbeddingProvider, make_embedding_provider
from ..wiki.auto_record import AutoRecorder


console = Console()


class Orchestrator:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.db = Database(self.settings.DB_PATH)
        self.wq = WQClient(self.settings)
        self._llm: BaseLLMProvider | None = None
        self._generators: dict[GenerationStrategy, BaseAlphaGenerator] = {}
        self._embedder: BaseEmbeddingProvider | None = None
        self._wiki_index: WikiIndex | None = None
        self._auto_recorder: AutoRecorder | None = None

    async def initialize(self) -> None:
        await self.db.connect()
        await self.wq.connect()
        self._llm = LLMFactory.from_settings(self.settings)

        retriever = await self._init_wiki()

        self._generators = {
            GenerationStrategy.LLM: LLMAlphaGenerator(
                self._llm,
                wiki_retriever=retriever,
                wiki_top_k=self.settings.WIKI_RETRIEVE_TOP_K,
                wiki_summary_chars=self.settings.WIKI_SUMMARY_CHARS,
            ),
            GenerationStrategy.TEMPLATE: TemplateAlphaGenerator(),
            GenerationStrategy.FACTOR_MINING: FactorMiningGenerator(),
        }
        logger.info("Orchestrator initialized")

    async def _init_wiki(self):
        store = WikiStore(self.settings.WIKI_DIR)
        if not store.exists():
            logger.info(f"Wiki dir {self.settings.WIKI_DIR} not found; skipping wiki integration")
            return None
        try:
            self._embedder = make_embedding_provider(self.settings)
        except Exception as exc:
            logger.warning(f"Embedding provider init failed, vector channel disabled: {exc}")
            from ..wiki.embeddings import NoOpEmbeddingProvider
            self._embedder = NoOpEmbeddingProvider()
        self._wiki_index = WikiIndex(
            store=store,
            db=self.db,
            embedder=self._embedder,
            grep_weight=self.settings.WIKI_GREP_WEIGHT,
            vector_weight=self.settings.WIKI_VECTOR_WEIGHT,
        )
        try:
            stats = await self._wiki_index.build(incremental=True)
            logger.info(
                f"Wiki ready: {stats.pages} pages, {stats.embeddings} embeddings, "
                f"{stats.edges} edges, {stats.communities} communities"
            )
        except Exception as exc:
            logger.warning(f"Wiki index build failed; retrieval disabled: {exc}")
            return None
        if self.settings.WIKI_AUTO_RECORD:
            self._auto_recorder = AutoRecorder(store=store, index=self._wiki_index)
        return self._wiki_index.retriever

    async def close(self) -> None:
        await self.wq.close()
        await self.db.close()
        if self._llm:
            await self._llm.close()
        if self._embedder:
            await self._embedder.close()
        logger.info("Orchestrator closed")

    async def run(
        self,
        strategy: GenerationStrategy = GenerationStrategy.LLM,
        count: int = 18,
        auto_backtest: bool = True,
    ) -> list[AlphaRecord]:
        generator = self._generators.get(strategy)
        if not generator:
            raise ValueError(f"Unknown strategy: {strategy}")

        console.print("\n[bold cyan]Fetching data fields and operators from WQ Brain...[/bold cyan]")
        data_fields = await self.wq.get_data_fields()
        operators = await self.wq.get_operators()
        console.print(f"  Loaded [green]{len(data_fields)}[/green] fields, [green]{len(operators)}[/green] operators")

        previous = await self.db.list_recent_backtested_alphas(limit=20)
        if previous:
            console.print(f"  Feeding [yellow]{len(previous)}[/yellow] prior backtested alphas back as feedback")

        console.print(f"\n[bold cyan]Generating {count} alphas using {strategy.value} strategy...[/bold cyan]")
        expressions = await generator.generate(
            data_fields, operators, previous_results=previous, count=count,
        )
        console.print(f"  Generated [green]{len(expressions)}[/green] valid expressions")

        records = []
        for expr in expressions:
            record = AlphaRecord(
                expression=expr,
                strategy=strategy,
                llm_model=self.settings.LLM_PROVIDER if strategy == GenerationStrategy.LLM else None,
            )
            records.append(record)

        ids = await self.db.batch_insert_alphas(records)
        for record, rid in zip(records, ids):
            record.id = rid

        console.print(f"  Saved {len(ids)} alphas to database")

        if auto_backtest and ids:
            console.print(f"\n[bold cyan]Starting backtest for {len(ids)} alphas...[/bold cyan]")
            engine = BacktestEngine(self.wq, self.db, self.settings)
            results = await engine.backtest_batch(ids)
            self._display_results(records, results)
            if self._auto_recorder:
                stats = await self._auto_recorder.record(records, results)
                if stats["entries"] or stats["lessons"]:
                    console.print(
                        f"[dim]Wiki auto-record: +{stats['entries']} entries, "
                        f"+{stats['lessons']} lessons[/dim]"
                    )

        return records

    async def backtest(self, alpha_ids: list[int]) -> list[BacktestResult]:
        engine = BacktestEngine(self.wq, self.db, self.settings)
        return await engine.backtest_batch(alpha_ids)

    async def list_high_quality(self, min_fitness: float | None = None) -> list[dict[str, Any]]:
        min_fitness = min_fitness or self.settings.MIN_FITNESS
        return await self.db.list_high_quality_alphas(min_fitness)

    async def status(self) -> dict[str, int]:
        return await self.db.get_stats()

    def _display_results(
        self,
        records: list[AlphaRecord],
        results: list[BacktestResult],
    ) -> None:
        high = [r for r in results if r.grade == QualityGrade.HIGH]
        medium = [r for r in results if r.grade == QualityGrade.MEDIUM]
        low = [r for r in results if r.grade == QualityGrade.LOW]
        rejected = [r for r in results if r.grade == QualityGrade.REJECT]

        console.print(
            f"\n[bold]Results:[/bold] "
            f"[green]{len(high)} HIGH[/green] | "
            f"[yellow]{len(medium)} MEDIUM[/yellow] | "
            f"[dim]{len(low)} LOW[/dim] | "
            f"[red]{len(rejected)} REJECT[/red]"
        )

        if high:
            table = Table(title="High Quality Alphas")
            table.add_column("ID", style="cyan", justify="right")
            table.add_column("Expression", max_width=60)
            table.add_column("Fitness", style="green", justify="right")
            table.add_column("Sharpe", style="green", justify="right")
            table.add_column("Turnover", justify="right")

            result_map = {r.alpha_id: r for r in results}
            for record in records:
                if record.id and record.id in result_map:
                    r = result_map[record.id]
                    if r.grade == QualityGrade.HIGH:
                        table.add_row(
                            str(record.id),
                            record.expression[:60],
                            f"{r.fitness:.4f}" if r.fitness else "N/A",
                            f"{r.sharpe:.4f}" if r.sharpe else "N/A",
                            f"{r.turnover:.4f}" if r.turnover else "N/A",
                        )
            console.print(table)

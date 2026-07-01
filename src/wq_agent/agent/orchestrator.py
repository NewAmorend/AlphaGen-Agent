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
from ..generator.refine import RefineAlphaGenerator
from ..generator.template import TemplateAlphaGenerator
from ..generator.factor import FactorMiningGenerator
from ..generator.base import BaseAlphaGenerator
from ..engine.backtest import BacktestEngine
from ..submitter import SubmissionManager
from ..wiki.store import CompositeWikiStore, WikiStore
from ..wiki.index import WikiIndex
from ..wiki.embeddings import BaseEmbeddingProvider, make_embedding_provider
from ..wiki.auto_record import AutoRecorder


console = Console()

SUBMIT_MODE_BACKTEST_ONLY = "backtest-only"
SUBMIT_MODE_AUTO_SUBMIT = "auto-submit"
SUBMIT_MODES = {SUBMIT_MODE_BACKTEST_ONLY, SUBMIT_MODE_AUTO_SUBMIT}


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
        recovered = await self.db.reset_stuck_backtesting()
        if recovered:
            logger.info(f"Recovered {recovered} alphas stuck in 'backtesting' from prior crash")
        await self.wq.connect()
        self._llm = LLMFactory.from_settings(self.settings)

        retriever = await self._init_wiki()

        self._generators = {
            GenerationStrategy.LLM: LLMAlphaGenerator(
                self._llm,
                wiki_retriever=retriever,
                wiki_top_k=self.settings.WIKI_RETRIEVE_TOP_K,
                wiki_summary_chars=self.settings.WIKI_SUMMARY_CHARS,
                temperature=self.settings.LLM_GEN_TEMPERATURE,
            ),
            GenerationStrategy.TEMPLATE: TemplateAlphaGenerator(),
            GenerationStrategy.FACTOR_MINING: FactorMiningGenerator(),
        }
        logger.info("Orchestrator initialized")

    async def _init_wiki(self):
        public_store = WikiStore(self.settings.WIKI_DIR)
        private_store = WikiStore(self.settings.WIKI_AUTO_RECORD_DIR)
        store = CompositeWikiStore(public_store.root, [private_store.root])
        if not store.exists():
            logger.info(
                f"Wiki dirs {self.settings.WIKI_DIR} / {self.settings.WIKI_AUTO_RECORD_DIR} "
                "not found; skipping wiki integration"
            )
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
            self._auto_recorder = AutoRecorder(store=private_store, index=self._wiki_index)
            logger.info(f"Wiki auto-record target: {private_store.root}")
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
        submit_mode: str = SUBMIT_MODE_BACKTEST_ONLY,
        user_idea: str | None = None,
        dataset_categories: list[str] | None = None,
        market_region: str | None = None,
    ) -> list[AlphaRecord]:
        _validate_submit_mode(submit_mode)
        generator = self._generators.get(strategy)
        if not generator:
            raise ValueError(f"Unknown strategy: {strategy}")

        console.print("\n[bold cyan]Fetching data fields and operators from WQ Brain...[/bold cyan]")
        if dataset_categories:
            console.print(f"  Dataset categories: [yellow]{', '.join(dataset_categories)}[/yellow]")
        if market_region:
            console.print(f"  Market: [yellow]{market_region}[/yellow]")
        data_fields = await self.wq.get_data_fields(region=market_region, categories=dataset_categories)
        operators = await self.wq.get_operators()
        blacklist = await self.db.get_blacklisted_fields(min_fail_count=3)
        if blacklist:
            before = len(data_fields)
            data_fields = [f for f in data_fields if f.id not in blacklist]
            console.print(
                f"  Filtered out [yellow]{before - len(data_fields)}[/yellow] blacklisted fields "
                f"(known to fail WQ simulation)"
            )
        console.print(f"  Loaded [green]{len(data_fields)}[/green] fields, [green]{len(operators)}[/green] operators")

        previous = await self.db.list_recent_backtested_alphas(limit=20)
        if previous:
            console.print(f"  Feeding [yellow]{len(previous)}[/yellow] prior backtested alphas back as feedback")

        forbidden = await self.db.list_field_blacklist()
        if forbidden:
            console.print(f"  Surfacing top [yellow]{min(len(forbidden), 15)}[/yellow] blacklisted fields as explicit warnings to LLM")

        # 用 top-N 而非 MIN_FITNESS 阈值——冷启动时库里没 ≥1.0 的，但有 0.4 的 LOW
        # 也比让 LLM 凭空想象强。最差就是把"目前最好的 5 个"给它看。
        exemplars = await self.db.list_top_fitness_alphas(limit=5, min_fitness=0.0)
        if exemplars:
            best_fit = exemplars[0].get("fitness")
            console.print(
                f"  Surfacing [yellow]{len(exemplars)}[/yellow] top-fitness historical alphas as templates "
                f"(best fitness {best_fit:.2f})"
            )

        submitted_skeletons = await self.db.get_blacklisted_skeletons()  # submitted + self-corr FAIL
        if submitted_skeletons:
            console.print(
                f"  Excluding [yellow]{len(submitted_skeletons)}[/yellow] submitted-alpha skeletons from generation"
            )

        low_fit_skeletons: set[str] = set()
        if self.settings.DEDUP_FITNESS_FLOOR > 0:
            low_fit_skeletons = await self.db.get_low_fitness_skeletons(self.settings.DEDUP_FITNESS_FLOOR)
            low_fit_skeletons -= submitted_skeletons  # 避免重复计数
            if low_fit_skeletons:
                console.print(
                    f"  Excluding [yellow]{len(low_fit_skeletons)}[/yellow] known-low-fitness skeletons "
                    f"(best fitness < {self.settings.DEDUP_FITNESS_FLOOR})"
                )

        # 库里 wrapper 家族分布——喂给生成器提示"避开已饱和结构"，对抗单一栽培
        family_distribution = await self.db.get_skeleton_distribution(limit=10)
        from ..generator.llm import overrepresented_families
        _total = family_distribution.get("total_backtested", 0)
        over = overrepresented_families(
            family_distribution.get("top_outer1", []),
            _total,
            family_distribution.get("unique_outer1", 0),
        )
        if over and _total >= 10:
            tops = ", ".join(f"{f['signature']}×{f['count']}" for f in over[:3])
            console.print(
                f"  Steering away from [yellow]{len(over)}[/yellow] over-represented outer ops "
                f"(top: {tops})"
            )

        if user_idea and user_idea.strip():
            console.print("  Using [yellow]user research idea[/yellow] as generation guidance")

        console.print(f"\n[bold cyan]Generating {count} alphas using {strategy.value} strategy...[/bold cyan]")
        expressions = await generator.generate(
            data_fields, operators, previous_results=previous, count=count,
            forbidden_fields=forbidden, high_fitness_exemplars=exemplars,
            submitted_skeletons=submitted_skeletons,
            extra_exclude_skeletons=low_fit_skeletons,
            family_distribution=family_distribution,
            user_idea=user_idea,
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
            engine = BacktestEngine(self.wq, self.db, self.settings, region_override=market_region)
            results = await engine.backtest_batch(ids)
            self._display_results(records, results)
            await self._handle_post_backtest_submission(submit_mode)
            if self._auto_recorder:
                stats = await self._auto_recorder.record(records, results)
                if stats["entries"] or stats["lessons"]:
                    console.print(
                        f"[dim]Wiki auto-record: +{stats['entries']} entries, "
                        f"+{stats['lessons']} lessons[/dim]"
                    )

        return records

    async def backtest(
        self,
        alpha_ids: list[int],
        market_region: str | None = None,
        submit_mode: str = SUBMIT_MODE_BACKTEST_ONLY,
    ) -> list[BacktestResult]:
        _validate_submit_mode(submit_mode)
        engine = BacktestEngine(self.wq, self.db, self.settings, region_override=market_region)
        results = await engine.backtest_batch(alpha_ids)
        await self._handle_post_backtest_submission(submit_mode)
        return results

    async def refine(
        self,
        base_id: int | None = None,
        count: int = 10,
        auto_backtest: bool = True,
        dataset_categories: list[str] | None = None,
        market_region: str | None = None,
    ) -> list[AlphaRecord]:
        """对 MEDIUM 评级（差一项即过）的 alpha 生成微调变体。

        base_id=None 时自动从 db 拉 fitness 最高的 MEDIUM 候选。
        """
        if base_id is not None:
            backtest = await self.db.get_backtest_result(base_id)
            alpha = await self.db.get_alpha(base_id)
            if alpha is None or backtest is None:
                raise ValueError(f"Alpha #{base_id} not found or never backtested")
            base = {
                "alpha_id": base_id,
                "expression": alpha.expression,
                "fitness": backtest.fitness,
                "sharpe": backtest.sharpe,
                "turnover": backtest.turnover,
                "returns": backtest.returns,
                "grade": backtest.grade.value if backtest.grade else None,
                "failed_checks": [
                    str(c.get("name", ""))
                    for c in (backtest.checks or [])
                    if isinstance(c, dict) and str(c.get("result", "")).upper() == "FAIL"
                ],
            }
        else:
            candidates = await self.db.list_refine_candidates(limit=1)
            if not candidates:
                console.print(
                    "[yellow]No MEDIUM-grade alphas to refine. "
                    "Run `wq-agent run` first to produce some near-miss candidates.[/yellow]"
                )
                return []
            base = candidates[0]
            console.print(
                f"[cyan]Auto-picked alpha #{base['alpha_id']} "
                f"(fitness={base['fitness']:.3f}, failed={base['failed_checks']})[/cyan]"
            )

        if not self._llm:
            raise RuntimeError("Orchestrator not initialized")
        refiner = RefineAlphaGenerator(
            llm=self._llm,
            wiki_retriever=self._wiki_index.retriever if self._wiki_index else None,
            wiki_top_k=self.settings.WIKI_RETRIEVE_TOP_K,
            wiki_summary_chars=self.settings.WIKI_SUMMARY_CHARS,
        )

        console.print("[cyan]Fetching fields & operators for refine context...[/cyan]")
        if dataset_categories:
            console.print(f"  Dataset categories: [yellow]{', '.join(dataset_categories)}[/yellow]")
        if market_region:
            console.print(f"  Market: [yellow]{market_region}[/yellow]")
        data_fields = await self.wq.get_data_fields(region=market_region, categories=dataset_categories)
        operators = await self.wq.get_operators()

        # 把历史 top-fitness 也喂给 refine——base 可能就是 top-1，但其它 top 给 LLM 更多模式参考
        exemplars = await self.db.list_top_fitness_alphas(limit=5, min_fitness=0.0)
        submitted_skeletons = await self.db.get_blacklisted_skeletons()  # submitted + self-corr FAIL
        low_fit_skeletons: set[str] = set()
        if self.settings.DEDUP_FITNESS_FLOOR > 0:
            low_fit_skeletons = await self.db.get_low_fitness_skeletons(self.settings.DEDUP_FITNESS_FLOOR)
            low_fit_skeletons -= submitted_skeletons

        console.print(f"[cyan]Generating {count} refine variants...[/cyan]")
        variants = await refiner.refine(
            base, data_fields, operators, count=count,
            high_fitness_exemplars=exemplars,
            submitted_skeletons=submitted_skeletons,
            extra_exclude_skeletons=low_fit_skeletons,
        )
        if not variants:
            console.print("[yellow]Refine returned 0 variants[/yellow]")
            return []
        console.print(f"  Generated [green]{len(variants)}[/green] variants")

        records = [
            AlphaRecord(
                expression=expr,
                strategy=GenerationStrategy.LLM,
                llm_model=f"refine:{base['alpha_id']}",
            )
            for expr in variants
        ]
        ids = await self.db.batch_insert_alphas(records)
        for rec, rid in zip(records, ids):
            rec.id = rid

        if auto_backtest and ids:
            console.print(f"[cyan]Backtesting {len(ids)} variants...[/cyan]")
            engine = BacktestEngine(self.wq, self.db, self.settings, region_override=market_region)
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

    async def list_high_quality(self, min_fitness: float | None = None) -> list[dict[str, Any]]:
        min_fitness = min_fitness or self.settings.MIN_FITNESS
        return await self.db.list_high_quality_alphas(min_fitness)

    async def status(self) -> dict[str, int]:
        return await self.db.get_stats()

    async def _handle_post_backtest_submission(self, submit_mode: str) -> None:
        manager = SubmissionManager(self.db, self.wq, self.settings)
        if submit_mode == SUBMIT_MODE_BACKTEST_ONLY:
            plan = await manager.prepare_queue()
            if plan.csv_path:
                console.print(
                    f"[dim]Submission candidates saved to {plan.csv_path} "
                    f"({plan.queued_count} queued)[/dim]"
                )
            return
        result = await manager.run_once()
        console.print(
            "[bold cyan]Auto-submit:[/bold cyan] "
            f"attempted {result.attempted}, succeeded {result.succeeded}, "
            f"failed {result.failed}, daily remaining {result.daily_remaining}"
        )

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


def _validate_submit_mode(submit_mode: str) -> None:
    if submit_mode not in SUBMIT_MODES:
        allowed = ", ".join(sorted(SUBMIT_MODES))
        raise ValueError(f"submit_mode must be one of: {allowed}")

from __future__ import annotations

import io
from contextlib import contextmanager
from typing import Any, Awaitable, Callable

from rich.console import Console
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Footer, Header, Input, RichLog, Select, Static

from .agent.orchestrator import Orchestrator
from .config import get_settings
from .db import Database
from .models import AlphaStatus, GenerationStrategy


_STRATEGIES = [
    ("llm", GenerationStrategy.LLM.value),
    ("template", GenerationStrategy.TEMPLATE.value),
    ("factor_mining", GenerationStrategy.FACTOR_MINING.value),
]

_DATASET_CATEGORIES = [
    ("all datasets", "all"),
    ("price / volume", "pv"),
    ("fundamental", "fundamental"),
    ("analyst", "analyst"),
    ("model", "model"),
    ("news", "news"),
    ("option", "option"),
    ("sentiment", "sentiment"),
    ("social media", "socialmedia"),
]

_MARKETS = [
    ("default market", "default"),
    ("USA", "USA"),
    ("China", "CHN"),
    ("Europe", "EUR"),
    ("Asia", "ASI"),
    ("Global", "GLB"),
]


class _TuiConsole:
    def __init__(self, write: Callable[[str], None], width: int = 110):
        self.write = write
        self.width = width

    def print(self, *objects: Any, **kwargs: Any) -> None:
        buf = io.StringIO()
        console = Console(file=buf, width=self.width, color_system=None, force_terminal=False)
        console.print(*objects, **kwargs)
        text = buf.getvalue().rstrip()
        if text:
            self.write(text)


class AlphaGenAgentTui(App[None]):
    """Keyboard-first terminal workbench for common AlphaGen Agent flows."""

    CSS = """
    Screen {
        background: #0f1419;
        color: #d8dee9;
    }

    #workspace {
        height: 1fr;
    }

    #sidebar {
        width: 34;
        min-width: 30;
        padding: 1;
        border-right: solid #2f3742;
        background: #111821;
    }

    #main {
        width: 1fr;
        padding: 1 1 0 1;
    }

    #status {
        height: auto;
        min-height: 8;
        padding: 1;
        border: solid #2f874f;
        margin-bottom: 1;
    }

    #log {
        height: 1fr;
        border: solid #334155;
        padding: 0 1;
    }

    #alphas {
        height: 12;
        margin-top: 1;
        border: solid #334155;
    }

    .label {
        color: #8fb3ff;
        text-style: bold;
        margin-top: 1;
    }

    .hint {
        color: #7d8590;
        margin: 1 0;
    }

    Button {
        width: 100%;
        margin-top: 1;
    }

    Input, Select {
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+r", "refresh", "Refresh"),
        ("g", "generate", "Generate"),
        ("r", "run_batch", "Run"),
        ("f", "refine", "Refine"),
        ("b", "backtest_pending", "Backtest"),
        ("l", "refresh", "List"),
    ]

    TITLE = "AlphaGen Agent"
    SUB_TITLE = "alpha research workbench"

    def __init__(self) -> None:
        super().__init__()
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="workspace"):
            with Vertical(id="sidebar"):
                yield Static("AlphaGen Agent", classes="label")
                yield Static("Generate, test, refine. Keep the loop tight.", classes="hint")
                yield Static("Strategy", classes="label")
                yield Select(_STRATEGIES, value=GenerationStrategy.LLM.value, id="strategy")
                yield Static("Dataset", classes="label")
                yield Select(_DATASET_CATEGORIES, value="all", id="dataset")
                yield Static("Market", classes="label")
                yield Select(_MARKETS, value="default", id="market")
                yield Static("Count", classes="label")
                yield Input(value="18", placeholder="alphas per batch", id="count")
                yield Static("Batches", classes="label")
                yield Input(value="1", placeholder="full pipeline batches", id="batches")
                yield Static("Idea", classes="label")
                yield Input(placeholder="natural-language alpha thesis", id="idea")
                yield Button("Generate only", id="generate", variant="primary")
                yield Button("Run full pipeline", id="run", variant="success")
                yield Button("Refine near-miss", id="refine")
                yield Button("Backtest pending", id="backtest")
                yield Button("Refresh dashboard", id="refresh")
            with Vertical(id="main"):
                yield Static("Loading dashboard...", id="status")
                yield RichLog(id="log", wrap=True, highlight=True, markup=True)
                yield DataTable(id="alphas", zebra_stripes=True)
        yield Footer()

    async def on_mount(self) -> None:
        self.write_log("[bold cyan]AlphaGen Agent TUI ready[/bold cyan]")
        self.write_log("Use buttons or keys: g generate, r run, f refine, b backtest, ctrl+r refresh, q quit.")
        await self.refresh_dashboard()

    def write_log(self, message: str) -> None:
        self.query_one("#log", RichLog).write(message)

    async def action_refresh(self) -> None:
        await self.refresh_dashboard()

    async def action_generate(self) -> None:
        await self._run_task("generate", self._generate_only)

    async def action_run_batch(self) -> None:
        await self._run_task("run", self._run_pipeline)

    async def action_refine(self) -> None:
        await self._run_task("refine", self._refine)

    async def action_backtest_pending(self) -> None:
        await self._run_task("backtest pending", self._backtest_pending)

    @on(Button.Pressed, "#refresh")
    async def _button_refresh(self) -> None:
        await self.action_refresh()

    @on(Button.Pressed, "#generate")
    async def _button_generate(self) -> None:
        await self.action_generate()

    @on(Button.Pressed, "#run")
    async def _button_run(self) -> None:
        await self.action_run_batch()

    @on(Button.Pressed, "#refine")
    async def _button_refine(self) -> None:
        await self.action_refine()

    @on(Button.Pressed, "#backtest")
    async def _button_backtest(self) -> None:
        await self.action_backtest_pending()

    async def refresh_dashboard(self) -> None:
        settings = get_settings()
        db = Database(settings.DB_PATH)
        await db.connect()
        try:
            stats = await db.get_stats()
            recent = await db.list_alphas(limit=12)
            high = await db.list_high_quality_alphas(min_fitness=0.0)
        finally:
            await db.close()

        high_count = len([r for r in high if (r.get("fitness") or 0) >= settings.MIN_FITNESS])
        status = (
            "[bold]Dashboard[/bold]\n"
            f"generated: [cyan]{stats.get('generated', 0)}[/cyan]  "
            f"backtesting: [yellow]{stats.get('backtesting', 0)}[/yellow]  "
            f"completed/high: [green]{high_count}[/green]  "
            f"submitted: [magenta]{stats.get('submitted', 0)}[/magenta]\n"
            f"db: [dim]{settings.DB_PATH}[/dim]\n"
            f"wiki: [dim]{settings.WIKI_DIR} + {settings.WIKI_AUTO_RECORD_DIR}[/dim]"
        )
        self.query_one("#status", Static).update(status)

        table = self.query_one("#alphas", DataTable)
        table.clear(columns=True)
        table.add_columns("ID", "Status", "Strategy", "Created", "Expression")
        for alpha in recent:
            table.add_row(
                str(alpha.id or ""),
                alpha.status.value,
                alpha.strategy.value,
                alpha.created_at.strftime("%m-%d %H:%M"),
                _truncate(alpha.expression, 84),
            )

    async def _run_task(self, label: str, job: Callable[[], Awaitable[None]]) -> None:
        if self._busy:
            self.write_log("[yellow]A task is already running.[/yellow]")
            return
        self._busy = True
        self.write_log(f"\n[bold cyan]> {label}[/bold cyan]")
        try:
            await job()
            self.write_log(f"[bold green]OK {label} finished[/bold green]")
        except Exception as exc:
            self.write_log(f"[bold red]ERR {label} failed:[/bold red] {exc}")
        finally:
            self._busy = False
            await self.refresh_dashboard()

    async def _generate_only(self) -> None:
        count = self._positive_int("#count", default=18)
        strategy = self._strategy()
        idea = self._idea()
        dataset_categories = self._dataset_categories()
        market_region = self._market_region()
        await self._with_orchestrator(
            lambda orch: orch.run(
                strategy=strategy,
                count=count,
                auto_backtest=False,
                user_idea=idea,
                dataset_categories=dataset_categories,
                market_region=market_region,
            )
        )

    async def _run_pipeline(self) -> None:
        count = self._positive_int("#count", default=18)
        batches = self._positive_int("#batches", default=1)
        strategy = self._strategy()
        idea = self._idea()
        dataset_categories = self._dataset_categories()
        market_region = self._market_region()

        async def _job(orch: Orchestrator) -> None:
            for i in range(1, batches + 1):
                self.write_log(f"[bold magenta]Batch {i}/{batches}[/bold magenta]")
                await orch.run(
                    strategy=strategy,
                    count=count,
                    auto_backtest=True,
                    user_idea=idea,
                    dataset_categories=dataset_categories,
                    market_region=market_region,
                )

        await self._with_orchestrator(_job)

    async def _refine(self) -> None:
        count = self._positive_int("#count", default=10)
        dataset_categories = self._dataset_categories()
        market_region = self._market_region()
        await self._with_orchestrator(
            lambda orch: orch.refine(
                count=count,
                auto_backtest=True,
                dataset_categories=dataset_categories,
                market_region=market_region,
            )
        )

    async def _backtest_pending(self) -> None:
        market_region = self._market_region()

        async def _job(orch: Orchestrator) -> None:
            pending = await orch.db.list_alphas(status=AlphaStatus.GENERATED, limit=1000)
            ids = [a.id for a in pending if a.id]
            if not ids:
                self.write_log("[yellow]No pending generated alphas.[/yellow]")
                return
            self.write_log(f"Backtesting {len(ids)} pending alphas")
            await orch.backtest(ids, market_region=market_region)
        await self._with_orchestrator(_job)

    async def _with_orchestrator(self, job: Callable[[Orchestrator], Awaitable[Any]]) -> None:
        orch = Orchestrator()
        with self._capture_orchestrator_console():
            try:
                await orch.initialize()
                await job(orch)
            finally:
                await orch.close()

    @contextmanager
    def _capture_orchestrator_console(self):
        from .agent import orchestrator as orchestrator_module

        old_console = orchestrator_module.console
        orchestrator_module.console = _TuiConsole(self.write_log)
        try:
            yield
        finally:
            orchestrator_module.console = old_console

    def _strategy(self) -> GenerationStrategy:
        value = self.query_one("#strategy", Select).value
        return GenerationStrategy(str(value or GenerationStrategy.LLM.value))

    def _idea(self) -> str | None:
        value = self.query_one("#idea", Input).value.strip()
        return value or None

    def _dataset_categories(self) -> list[str] | None:
        value = str(self.query_one("#dataset", Select).value or "all")
        return None if value == "all" else [value]

    def _market_region(self) -> str | None:
        value = str(self.query_one("#market", Select).value or "default")
        return None if value == "default" else value

    def _positive_int(self, selector: str, default: int) -> int:
        raw = self.query_one(selector, Input).value.strip()
        try:
            value = int(raw)
        except ValueError:
            value = default
        return max(1, value)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def run_tui() -> None:
    AlphaGenAgentTui().run()

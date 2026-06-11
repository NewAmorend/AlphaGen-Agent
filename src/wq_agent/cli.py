from __future__ import annotations

import asyncio
import sys
from importlib import resources
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from loguru import logger

from .config import get_settings
from .models import GenerationStrategy, AlphaStatus
from .agent.orchestrator import Orchestrator

app = typer.Typer(
    name="wq-agent",
    help="WorldQuant Alpha Generation & Backtesting Agent Harness",
    add_completion=False,
)
wiki_app = typer.Typer(name="wiki", help="Quant Wiki maintenance commands", add_completion=False)
app.add_typer(wiki_app, name="wiki")
console = Console()


def _resource_path(name: str):
    packaged = resources.files("wq_agent").joinpath("resources", name)
    if packaged.is_dir() or packaged.is_file():
        return packaged

    # Editable/source checkout fallback. In a wheel, hatch maps these files under
    # wq_agent/resources; in a checkout, they still live at the repository root.
    source_root = Path(__file__).resolve().parents[2]
    fallback = source_root / name
    if fallback.exists():
        return fallback
    return packaged


def _copy_resource_tree(source, destination: Path, overwrite: bool = False) -> tuple[int, int]:
    copied = 0
    skipped = 0

    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            return 0, 1
        destination.write_bytes(source.read_bytes())
        return 1, 0

    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            child_copied, child_skipped = _copy_resource_tree(item, target, overwrite)
            copied += child_copied
            skipped += child_skipped
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and not overwrite:
                skipped += 1
                continue
            target.write_bytes(item.read_bytes())
            copied += 1

    return copied, skipped


@app.command(name="init")
def init_project(
    target: Path = typer.Argument(Path("."), help="Directory to initialize"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing resource files"),
    with_env: bool = typer.Option(True, "--with-env/--no-env", help="Copy .env.example"),
):
    """Initialize public resources for a new working directory."""
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)

    wiki_dir = target / "wiki"
    wiki_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]wiki[/green]: ready at {wiki_dir} (empty by default)")

    total_copied = 0
    total_skipped = 0
    source = _resource_path("templates")
    if not source.is_dir():
        console.print("[red]Missing packaged resource: templates[/red]")
        raise typer.Exit(1)
    copied, skipped = _copy_resource_tree(source, target / "templates", overwrite=overwrite)
    total_copied += copied
    total_skipped += skipped
    console.print(f"[green]templates[/green]: copied {copied}, skipped {skipped}")

    if with_env:
        source = _resource_path(".env.example")
        if source.is_file():
            copied, skipped = _copy_resource_tree(source, target / ".env.example", overwrite=overwrite)
            total_copied += copied
            total_skipped += skipped
            console.print(f"[green].env.example[/green]: copied {copied}, skipped {skipped}")

    console.print(
        f"[bold green]Initialized {target}[/bold green] "
        f"([cyan]{total_copied}[/cyan] copied, [yellow]{total_skipped}[/yellow] skipped)"
    )


def _setup_logging(verbose: bool = False) -> None:
    level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=level, format="<level>{time:HH:mm:ss}</level> | <level>{message}</level>")
    logger.add("wq_agent.log", level="DEBUG", rotation="10 MB")


def _load_user_idea(idea: Optional[str], idea_file: Optional[Path]) -> str | None:
    parts: list[str] = []
    if idea and idea.strip():
        parts.append(idea.strip())
    if idea_file:
        try:
            text = idea_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise typer.BadParameter(f"Cannot read idea file {idea_file}: {exc}") from exc
        if text:
            parts.append(text)
    return "\n\n".join(parts) if parts else None


@app.command()
def tui(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Launch the interactive terminal workbench."""
    _setup_logging(verbose)
    try:
        from .tui import run_tui
    except ModuleNotFoundError as exc:
        if exc.name == "textual":
            console.print("[red]Textual is required for the TUI. Install project dependencies first.[/red]")
            raise typer.Exit(1) from exc
        raise
    run_tui()


@app.command()
def generate(
    strategy: str = typer.Option("llm", "--strategy", "-s", help="Generation strategy: llm, template, factor_mining"),
    count: int = typer.Option(18, "--count", "-n", help="Number of alphas to generate"),
    idea: Optional[str] = typer.Option(None, "--idea", help="Natural-language research idea to guide LLM generation"),
    idea_file: Optional[Path] = typer.Option(None, "--idea-file", help="Read natural-language research idea from a UTF-8 text file"),
    no_backtest: bool = typer.Option(False, "--no-backtest", help="Skip auto-backtest"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Generate new alpha expressions."""
    _setup_logging(verbose)

    async def _run():
        orch = Orchestrator()
        try:
            await orch.initialize()
            records = await orch.run(
                strategy=GenerationStrategy(strategy),
                count=count,
                auto_backtest=not no_backtest,
                user_idea=_load_user_idea(idea, idea_file),
            )
            console.print(f"\n[bold green]Generated {len(records)} alphas[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
            raise typer.Exit(1)
        finally:
            await orch.close()

    asyncio.run(_run())


@app.command()
def backtest(
    ids: Optional[str] = typer.Option(None, "--ids", "-i", help="Comma-separated alpha IDs"),
    pending: bool = typer.Option(False, "--pending", help="Backtest all pending alphas"),
    all_generated: bool = typer.Option(False, "--all", help="Backtest all generated alphas"),
    max_concurrent: int = typer.Option(5, "--concurrent", "-c", help="Max concurrent simulations"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run backtests on generated alphas."""
    _setup_logging(verbose)

    async def _run():
        settings = get_settings()
        settings.WQ_MAX_CONCURRENT = max_concurrent
        orch = Orchestrator(settings)
        try:
            await orch.initialize()

            alpha_ids: list[int] = []
            if ids:
                alpha_ids = [int(x.strip()) for x in ids.split(",")]
            elif pending:
                all_alphas = await orch.db.list_alphas(status=AlphaStatus.GENERATED, limit=1000)
                alpha_ids = [a.id for a in all_alphas if a.id]
            elif all_generated:
                generated = await orch.db.list_alphas(status=AlphaStatus.GENERATED, limit=1000)
                failed = await orch.db.list_alphas(status=AlphaStatus.FAILED, limit=1000)
                alpha_ids = [a.id for a in generated + failed if a.id]
            else:
                console.print("[yellow]Specify --ids, --pending, or --all[/yellow]")
                raise typer.Exit(1)

            if not alpha_ids:
                console.print("[yellow]No alphas to backtest[/yellow]")
                return

            console.print(f"[bold cyan]Backtesting {len(alpha_ids)} alphas...[/bold cyan]")
            results = await orch.backtest(alpha_ids)
            console.print(f"[bold green]Backtest complete: {len(results)} results[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
            raise typer.Exit(1)
        finally:
            await orch.close()

    asyncio.run(_run())


@app.command(name="list")
def list_alphas(
    quality: Optional[str] = typer.Option(None, "--quality", "-q", help="Filter by quality: high, medium"),
    min_fitness: Optional[float] = typer.Option(None, "--min-fitness", help="Minimum fitness threshold"),
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """List alphas and their results."""
    _setup_logging(verbose)

    async def _run():
        orch = Orchestrator()
        try:
            await orch.initialize()

            if quality or min_fitness:
                threshold = min_fitness or 0.5
                items = await orch.list_high_quality(threshold)
                if items:
                    table = Table(title=f"Alphas (fitness >= {threshold})")
                    table.add_column("ID", style="cyan", justify="right")
                    table.add_column("Expression", max_width=50)
                    table.add_column("Fitness", style="green", justify="right")
                    table.add_column("Sharpe", justify="right")
                    table.add_column("Turnover", justify="right")
                    table.add_column("Returns", justify="right")
                    table.add_column("Grade", justify="center")
                    for item in items:
                        table.add_row(
                            str(item["id"]),
                            item["expression"][:50],
                            f"{item['fitness']:.4f}" if item.get("fitness") else "N/A",
                            f"{item['sharpe']:.4f}" if item.get("sharpe") else "N/A",
                            f"{item['turnover']:.4f}" if item.get("turnover") else "N/A",
                            f"{item['returns']:.4f}" if item.get("returns") else "N/A",
                            str(item.get("grade", "")),
                        )
                    console.print(table)
                else:
                    console.print("[yellow]No alphas matching criteria[/yellow]")
            else:
                alpha_status = AlphaStatus(status) if status else None
                alphas = await orch.db.list_alphas(status=alpha_status, limit=limit)
                if alphas:
                    table = Table(title="Alphas")
                    table.add_column("ID", style="cyan", justify="right")
                    table.add_column("Expression", max_width=50)
                    table.add_column("Strategy", justify="center")
                    table.add_column("Status", justify="center")
                    table.add_column("Created", justify="right")
                    for a in alphas:
                        table.add_row(
                            str(a.id),
                            a.expression[:50],
                            a.strategy.value,
                            a.status.value,
                            a.created_at.strftime("%m-%d %H:%M"),
                        )
                    console.print(table)
                else:
                    console.print("[yellow]No alphas found[/yellow]")
        finally:
            await orch.close()

    asyncio.run(_run())


@app.command()
def run(
    strategy: str = typer.Option("llm", "--strategy", "-s", help="Generation strategy"),
    count: int = typer.Option(18, "--count", "-n", help="Alphas per batch"),
    batches: int = typer.Option(1, "--batches", "-b", help="Number of batches"),
    interval: int = typer.Option(60, "--interval", help="Seconds between batches"),
    idea: Optional[str] = typer.Option(None, "--idea", help="Natural-language research idea to guide LLM generation"),
    idea_file: Optional[Path] = typer.Option(None, "--idea-file", help="Read natural-language research idea from a UTF-8 text file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Full pipeline: generate → backtest → evaluate → display."""
    _setup_logging(verbose)

    async def _run():
        orch = Orchestrator()
        try:
            await orch.initialize()
            strat = GenerationStrategy(strategy)
            user_idea = _load_user_idea(idea, idea_file)
            for batch_num in range(1, batches + 1):
                console.print(f"\n[bold magenta]═══ Batch {batch_num}/{batches} ═══[/bold magenta]")
                await orch.run(strategy=strat, count=count, auto_backtest=True, user_idea=user_idea)
                if batch_num < batches:
                    console.print(f"\n[dim]Waiting {interval}s before next batch...[/dim]")
                    await asyncio.sleep(interval)
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
            raise typer.Exit(1)
        finally:
            await orch.close()

    asyncio.run(_run())


@app.command()
def refine(
    base_id: Optional[int] = typer.Option(None, "--base-id", help="Alpha id to refine; auto-picks best MEDIUM if omitted"),
    count: int = typer.Option(10, "--count", "-n", help="Number of variants to generate"),
    no_backtest: bool = typer.Option(False, "--no-backtest"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Generate refined variants of a near-miss (MEDIUM-grade) alpha to push it to HIGH."""
    _setup_logging(verbose)

    async def _run():
        orch = Orchestrator()
        try:
            await orch.initialize()
            await orch.refine(base_id=base_id, count=count, auto_backtest=not no_backtest)
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
            raise typer.Exit(1)
        finally:
            await orch.close()

    asyncio.run(_run())


fields_app = typer.Typer(name="fields", help="Field blacklist maintenance", add_completion=False)
app.add_typer(fields_app, name="fields")


@fields_app.command("blacklist")
def fields_blacklist(
    clear: bool = typer.Option(False, "--clear", help="Wipe blacklist"),
    min_fails: int = typer.Option(3, "--min-fails", help="Show fields with fail_count >= N"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Show / clear the field blacklist (fields that repeatedly cause WQ sim errors)."""
    _setup_logging(verbose)
    from .db import Database

    async def _run():
        db = Database(get_settings().DB_PATH)
        await db.connect()
        try:
            if clear:
                n = await db.clear_field_blacklist()
                console.print(f"[green]Cleared {n} blacklist entries[/green]")
                return
            rows = await db.list_field_blacklist()
            rows = [r for r in rows if r["fail_count"] >= min_fails]
            if not rows:
                console.print(f"[yellow]No fields with fail_count >= {min_fails}[/yellow]")
                return
            table = Table(title=f"Field blacklist (fail_count >= {min_fails})")
            table.add_column("Field", style="cyan")
            table.add_column("Fails", justify="right", style="red")
            table.add_column("Last reason", max_width=50)
            table.add_column("Last seen", justify="right", style="dim")
            for r in rows:
                table.add_row(r["field_id"], str(r["fail_count"]),
                              r["last_reason"] or "", r["last_seen"][:19])
            console.print(table)
        finally:
            await db.close()

    asyncio.run(_run())


@app.command()
def status(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Show overall statistics."""
    _setup_logging(verbose)

    async def _run():
        orch = Orchestrator()
        try:
            await orch.initialize()
            stats = await orch.status()
            table = Table(title="WQ Agent Status")
            table.add_column("Metric", style="cyan")
            table.add_column("Count", justify="right")
            for key, val in stats.items():
                table.add_row(key, str(val))
            console.print(table)
        finally:
            await orch.close()

    asyncio.run(_run())


@app.command()
def diversity(
    limit: int = typer.Option(20, "--limit", "-n", help="Top wrapper families to show"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Show structural redundancy of the alpha library (outer-2 wrapper families).

    家族数 ≪ alpha 数 = 结构单一栽培。看哪个 wrapper 壳子霸屏，以及它是否真有用
    （avg/max fitness）——霸屏但低分的家族就是该让生成器换花样的信号。
    """
    _setup_logging(verbose)
    from .db import Database

    async def _run():
        db = Database(get_settings().DB_PATH)
        await db.connect()
        try:
            dist = await db.get_skeleton_distribution(limit=limit)
            total = dist["total_backtested"]
            if not total:
                console.print("[yellow]No backtested alphas yet.[/yellow]")
                return
            uniq_skel = dist["unique_skeletons"]
            uniq_o2 = dist["unique_outer2"]
            console.print(
                f"\n[bold]Backtested alphas:[/bold] {total} | "
                f"[bold]unique skeletons:[/bold] {uniq_skel} | "
                f"[bold]unique wrapper families (outer-2):[/bold] {uniq_o2}"
            )
            ratio = uniq_o2 / total if total else 0
            color = "red" if ratio < 0.2 else "yellow" if ratio < 0.4 else "green"
            console.print(
                f"  family/alpha ratio: [{color}]{ratio:.0%}[/{color}] "
                f"[dim](越低越单一；< 20% 说明结构高度集中)[/dim]"
            )
            table = Table(title=f"Top {len(dist['top_outer2'])} wrapper families by count")
            table.add_column("#", justify="right", style="dim")
            table.add_column("Count", justify="right", style="cyan")
            table.add_column("Avg fit", justify="right")
            table.add_column("Max fit", justify="right", style="green")
            table.add_column("Outer-2 signature", style="magenta")
            for i, fam in enumerate(dist["top_outer2"], 1):
                avg = fam["avg_fitness"]
                mx = fam["max_fitness"]
                table.add_row(
                    str(i),
                    str(fam["count"]),
                    f"{avg:.2f}" if avg is not None else "—",
                    f"{mx:.2f}" if mx is not None else "—",
                    fam["signature"],
                )
            console.print(table)
        finally:
            await db.close()

    asyncio.run(_run())


@app.command(name="screen-corr")
def screen_corr(
    scope: str = typer.Option("candidates", "--scope",
                              help="candidates | high | all"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """用 PnL 相关性筛重复：命中硬 gate 的写本地 SELF_CORRELATION FAIL
    （自动踢出可提交列表 + 拉黑骨架 + refine 跳过）。"""
    _setup_logging(verbose)
    from .engine.correlation import CorrelationScreener

    async def _run():
        orch = Orchestrator()
        try:
            await orch.initialize()
            if scope == "high":
                rows = await orch.db.list_high_quality_alphas(min_fitness=0.0)
                ids = [r["id"] for r in rows]
            elif scope == "all":
                cand = await orch.db.list_refine_candidates(limit=300)
                hi = await orch.db.list_high_quality_alphas(min_fitness=0.0)
                ids = list({*(c["alpha_id"] for c in cand), *(r["id"] for r in hi)})
            else:  # candidates
                cand = await orch.db.list_refine_candidates(limit=300)
                ids = [c["alpha_id"] for c in cand]
            screener = CorrelationScreener(orch.db, orch.wq, orch.settings)
            verdicts = await screener.screen(ids)
            hard = [v for v in verdicts if v.hard_redundant]
            console.print(f"Screened [cyan]{len(verdicts)}[/cyan] alphas; "
                          f"[red]{len(hard)}[/red] hard-redundant (marked SELF_CORRELATION FAIL)")
            for v in hard:
                console.print(f"  #{v.alpha_id} corr={v.hard_corr:.3f} vs #{v.hard_ref_id}")
        finally:
            await orch.close()

    asyncio.run(_run())


@app.command()
def submittable(
    min_fitness: float = typer.Option(1.0, "--min-fitness", help="Minimum fitness to show"),
    limit: int = typer.Option(50, "--limit", "-n"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """List alphas ready for submission (HIGH grade, not yet submitted).

    Filter: grade=HIGH (all 6 critical WQ checks PASS) AND status != submitted
    AND fitness >= --min-fitness AND SELF_CORRELATION not FAIL.
    """
    _setup_logging(verbose)
    from .db import Database, expression_outer_signature

    async def _run():
        db = Database(get_settings().DB_PATH)
        await db.connect()
        try:
            rows = await db.list_submittable_alphas(min_fitness=min_fitness, limit=limit)
            if not rows:
                console.print(f"[yellow]No submittable alphas (HIGH grade, fitness >= {min_fitness}, not yet submitted)[/yellow]")
                return
            table = Table(title=f"Submittable alphas ({len(rows)} candidates, fitness ≥ {min_fitness})")
            table.add_column("ID", justify="right", style="cyan")
            table.add_column("Fit", justify="right", style="green")
            table.add_column("Sharpe", justify="right")
            table.add_column("Turnover", justify="right")
            table.add_column("Outer", style="magenta", max_width=30)
            table.add_column("Expression", max_width=70)
            table.add_column("WQ id", style="dim")
            for r in rows:
                outer = expression_outer_signature(r["expression"], levels=2)
                table.add_row(
                    str(r["alpha_id"]),
                    f"{r['fitness']:.2f}" if r["fitness"] is not None else "—",
                    f"{r['sharpe']:.2f}" if r["sharpe"] is not None else "—",
                    f"{r['turnover']:.3f}" if r["turnover"] is not None else "—",
                    outer or "",
                    r["expression"][:68],
                    r["wq_alpha_id"] or "—",
                )
            console.print(table)
            console.print("\n[dim]Tip: run [bold]wq-agent submit <ID>[/bold] to mark as submitted, "
                          "or [bold]wq-agent sync-submitted[/bold] to pull real status from WQ Brain.[/dim]")
        finally:
            await db.close()

    asyncio.run(_run())


@app.command()
def submit(
    alpha_id: int = typer.Argument(..., help="Local alpha id to mark as submitted"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Mark a local alpha as SUBMITTED (local bookkeeping only).

    Does not call WQ Brain API — submit via the WQ web UI yourself, then run this
    to update local state. Use [bold]wq-agent sync-submitted[/bold] to pull real
    state from WQ Brain instead of marking one-by-one.
    """
    _setup_logging(verbose)
    from .db import Database

    async def _run():
        db = Database(get_settings().DB_PATH)
        await db.connect()
        try:
            alpha = await db.get_alpha(alpha_id)
            if alpha is None:
                console.print(f"[red]Alpha #{alpha_id} not found[/red]")
                raise typer.Exit(1)
            if alpha.status == AlphaStatus.SUBMITTED:
                console.print(f"[yellow]Alpha #{alpha_id} already marked submitted at {alpha.submitted_at}[/yellow]")
                return
            await db.mark_alpha_submitted(alpha_id)
            console.print(f"[green]✓ Marked alpha #{alpha_id} as SUBMITTED[/green]")
            console.print(f"  expr: [dim]{alpha.expression[:120]}[/dim]")
        finally:
            await db.close()

    asyncio.run(_run())


@app.command(name="sync-submitted")
def sync_submitted(
    limit: int = typer.Option(200, "--limit", "-n", help="Max alphas to fetch from WQ Brain"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Pull real submitted-alpha state from WQ Brain and reconcile with local DB.

    For each alpha on WQ Brain:
    - If a local alpha matches by wq_alpha_id or expression: mark it SUBMITTED.
    - Otherwise insert it as an external SUBMITTED record (so its skeleton goes
      on the LLM "don't regenerate" blacklist).
    """
    _setup_logging(verbose)
    from datetime import datetime
    from .db import Database
    from .wq.client import WQClient

    async def _run():
        settings = get_settings()
        db = Database(settings.DB_PATH)
        await db.connect()
        client = WQClient(settings)
        await client.connect()
        try:
            console.print(f"[cyan]Fetching up to {limit} alphas from WQ Brain (filter on dateSubmitted)...[/cyan]")
            remote = await client.get_submitted_alphas(limit=limit)
            # WQ API 返回的是 dateCreated 倒序的全部 alpha（含 simulation 草稿），
            # 真正"已提交"的判据是 dateSubmitted != null。
            truly_submitted = [r for r in remote if r.get("dateSubmitted")]
            console.print(
                f"  Got [green]{len(remote)}[/green] remote rows total, "
                f"[bold green]{len(truly_submitted)}[/bold green] truly submitted "
                f"(dateSubmitted != null), [dim]{len(remote) - len(truly_submitted)}[/dim] are just simulation drafts."
            )
            matched = 0
            inserted = 0
            already = 0
            checks_refreshed = 0  # 包括未 submit 的——self-corr 是异步计算的，本地数据常常过期
            for row in remote:
                wq_id = row.get("id")
                regular = row.get("regular") or {}
                expression = regular.get("code") if isinstance(regular, dict) else None
                if not (wq_id and expression):
                    continue
                is_data = row.get("is") or {}
                fresh_checks = is_data.get("checks")
                date_submitted = None
                if row.get("dateSubmitted"):
                    try:
                        date_submitted = datetime.fromisoformat(row["dateSubmitted"])
                    except (ValueError, TypeError):
                        pass
                # 优先 wq_alpha_id 匹配，没有就 expression 匹配
                local_ids = await db.find_alphas_by_wq_id(wq_id)
                if not local_ids:
                    by_expr = await db.find_alpha_by_expression(expression)
                    if by_expr is not None:
                        local_ids = [by_expr]

                if local_ids:
                    # 选择数据源：list endpoint 的 SELF_CORRELATION 经常是 PENDING（异步算），
                    # 只在 list 已是 PENDING 时才打 /alphas/{id}/check 拉终态——省 API 配额
                    list_sc_pending = any(
                        str(c.get("name", "")).upper() == "SELF_CORRELATION"
                        and str(c.get("result", "")).upper() == "PENDING"
                        for c in (fresh_checks or []) if isinstance(c, dict)
                    )
                    checks_to_write = fresh_checks
                    if list_sc_pending:
                        detail_checks = await client.get_alpha_check(wq_id)
                        if detail_checks:
                            checks_to_write = detail_checks
                    # 同 wq_id 的所有本地 alpha（refine verbatim 重复时常有）都要更新
                    if checks_to_write:
                        for lid in local_ids:
                            await db.update_backtest_checks(lid, checks_to_write)
                        checks_refreshed += len(local_ids)
                    # 再处理 submission 状态——所有匹配的 local id 都标
                    if date_submitted:
                        for lid in local_ids:
                            alpha = await db.get_alpha(lid)
                            if alpha and alpha.status == AlphaStatus.SUBMITTED:
                                already += 1
                            else:
                                await db.mark_alpha_submitted(lid, date_submitted)
                                matched += 1
                elif date_submitted:
                    # 远端真提交了但本地无 → 灌进来当 SUBMITTED 外部 alpha
                    settings_dict = row.get("settings") or {}
                    await db.upsert_external_submitted_alpha(
                        wq_alpha_id=wq_id,
                        expression=expression,
                        date_submitted=date_submitted,
                        is_metrics=is_data,
                        region=settings_dict.get("region", "USA"),
                        universe=settings_dict.get("universe", "TOP3000"),
                        delay=settings_dict.get("delay", 1),
                        neutralization=settings_dict.get("neutralization", "INDUSTRY"),
                    )
                    inserted += 1
            console.print(f"[green]✓ sync done[/green]: "
                          f"[cyan]{matched}[/cyan] newly marked submitted, "
                          f"[cyan]{inserted}[/cyan] external inserted, "
                          f"[dim]{already}[/dim] already-submitted, "
                          f"[yellow]{checks_refreshed}[/yellow] checks refreshed")
            blacklist = await db.get_blacklisted_skeletons()
            submitted = await db.get_submitted_skeletons()
            console.print(
                f"  Skeleton blacklist now: [yellow]{len(blacklist)}[/yellow] total "
                f"([cyan]{len(submitted)}[/cyan] submitted + "
                f"[red]{len(blacklist) - len(submitted)}[/red] self-corr FAIL) "
                "→ LLM/refine prompt will exclude these"
            )
        finally:
            await client.close()
            await db.close()

    asyncio.run(_run())


@wiki_app.command("stats")
def wiki_stats(verbose: bool = typer.Option(False, "--verbose", "-v")):
    """Show wiki page / edge / embedding counts."""
    _setup_logging(verbose)
    from .wiki.store import WikiStore
    from .db import Database

    async def _run():
        settings = get_settings()
        store = WikiStore(settings.WIKI_DIR)
        db = Database(settings.DB_PATH)
        await db.connect()
        try:
            counts = await db.wiki_counts()
            pages, errors = store.load_pages() if store.exists() else ([], [])
            table = Table(title="Quant Wiki Stats")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right")
            table.add_row("wiki dir exists", str(store.exists()))
            table.add_row("pages on disk", str(len(pages)))
            table.add_row("parse errors", str(len(errors)))
            table.add_row("pages in db", str(counts["wiki_pages"]))
            table.add_row("embeddings in db", str(counts["wiki_embeddings"]))
            table.add_row("sqlite-vec loaded", str(getattr(db, "vec_extension_loaded", False)))
            console.print(table)
        finally:
            await db.close()

    asyncio.run(_run())


@wiki_app.command("index")
def wiki_index(
    incremental: bool = typer.Option(False, "--incremental", help="Skip pages whose hash didn't change"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """(Re)build wiki indexes: page metadata, embeddings, graph."""
    _setup_logging(verbose)
    from .wiki.store import WikiStore
    from .wiki.index import WikiIndex
    from .wiki.embeddings import make_embedding_provider
    from .db import Database

    async def _run():
        settings = get_settings()
        store = WikiStore(settings.WIKI_DIR)
        if not store.exists():
            console.print(f"[yellow]Wiki dir not found at {settings.WIKI_DIR}[/yellow]")
            raise typer.Exit(0)
        db = Database(settings.DB_PATH)
        await db.connect()
        embedder = make_embedding_provider(settings)
        index = WikiIndex(
            store=store,
            db=db,
            embedder=embedder,
            grep_weight=settings.WIKI_GREP_WEIGHT,
            vector_weight=settings.WIKI_VECTOR_WEIGHT,
        )
        try:
            stats = await index.build(incremental=incremental)
            table = Table(title="Index built")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right")
            for k, v in stats.__dict__.items():
                table.add_row(k, str(v))
            console.print(table)
        finally:
            await embedder.close()
            await db.close()

    asyncio.run(_run())


@wiki_app.command("search")
def wiki_search(
    query: str = typer.Argument(..., help="Query string"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Debug retrieval: run hybrid search and print hits."""
    _setup_logging(verbose)
    from .wiki.store import WikiStore
    from .wiki.index import WikiIndex
    from .wiki.embeddings import make_embedding_provider
    from .db import Database

    async def _run():
        settings = get_settings()
        store = WikiStore(settings.WIKI_DIR)
        if not store.exists():
            console.print(f"[yellow]Wiki dir not found at {settings.WIKI_DIR}[/yellow]")
            raise typer.Exit(1)
        db = Database(settings.DB_PATH)
        await db.connect()
        embedder = make_embedding_provider(settings)
        index = WikiIndex(
            store=store,
            db=db,
            embedder=embedder,
            grep_weight=settings.WIKI_GREP_WEIGHT,
            vector_weight=settings.WIKI_VECTOR_WEIGHT,
        )
        try:
            await index.build(incremental=True)
            retriever = index.retriever
            if retriever is None:
                console.print("[yellow]Retriever unavailable[/yellow]")
                raise typer.Exit(1)
            hits = await retriever.search(query, top_k=top_k)
            if not hits:
                console.print("[yellow]No hits[/yellow]")
                return
            table = Table(title=f'Results for "{query}"')
            table.add_column("#", justify="right")
            table.add_column("Score", justify="right", style="green")
            table.add_column("Page", style="cyan")
            table.add_column("Type")
            table.add_column("Sources")
            for i, h in enumerate(hits, 1):
                table.add_row(
                    str(i),
                    f"{h.score:.3f}",
                    h.page.title,
                    h.page.type.value,
                    ", ".join(h.sources),
                )
            console.print(table)
        finally:
            await embedder.close()
            await db.close()

    asyncio.run(_run())


@wiki_app.command("eval")
def wiki_eval(
    top_k: int = typer.Option(5, "--top-k", "-k", help="Evaluate hit@k / MRR at this cutoff"),
    golden: Optional[str] = typer.Option(None, "--golden", help="Optional YAML golden-query file"),
    min_hit_at_k: Optional[float] = typer.Option(None, "--min-hit-at-k", help="Fail if hit@k is below this value (0-1)"),
    min_mrr: Optional[float] = typer.Option(None, "--min-mrr", help="Fail if MRR is below this value (0-1)"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Evaluate wiki retrieval quality against golden queries."""
    _setup_logging(verbose)
    from .wiki.store import WikiStore
    from .wiki.index import WikiIndex
    from .wiki.embeddings import make_embedding_provider
    from .wiki.eval import evaluate_retriever, load_golden_queries
    from .db import Database
    from pathlib import Path

    async def _run():
        settings = get_settings()
        store = WikiStore(settings.WIKI_DIR)
        if not store.exists():
            console.print(f"[yellow]Wiki dir not found at {settings.WIKI_DIR}[/yellow]")
            raise typer.Exit(1)
        default_bench = Path(settings.WIKI_DIR) / "bench" / "retrieval_golden.yml"
        golden_path = golden or (str(default_bench) if default_bench.exists() else None)
        queries = load_golden_queries(golden_path)
        db = Database(settings.DB_PATH)
        await db.connect()
        embedder = make_embedding_provider(settings)
        index = WikiIndex(
            store=store,
            db=db,
            embedder=embedder,
            grep_weight=settings.WIKI_GREP_WEIGHT,
            vector_weight=settings.WIKI_VECTOR_WEIGHT,
        )
        try:
            stats = await index.build(incremental=True)
            retriever = index.retriever
            if retriever is None:
                console.print("[yellow]Retriever unavailable[/yellow]")
                raise typer.Exit(1)
            report = await evaluate_retriever(retriever, queries, top_k=top_k)
            failed_thresholds: list[str] = []
            if min_hit_at_k is not None and report.hit_at_k < min_hit_at_k:
                failed_thresholds.append(f"hit@{top_k} {report.hit_at_k:.3f} < {min_hit_at_k:.3f}")
            if min_mrr is not None and report.mrr < min_mrr:
                failed_thresholds.append(f"MRR {report.mrr:.3f} < {min_mrr:.3f}")

            if json_output:
                console.print(report.to_json())
                if failed_thresholds:
                    raise typer.Exit(2)
                return

            summary = Table(title=f"Wiki Retrieval Eval ({report.n_queries} queries @ top-{top_k})")
            summary.add_column("Metric", style="cyan")
            summary.add_column("Value", justify="right")
            summary.add_row("hit@1", f"{report.hit_at_1:.1%}")
            summary.add_row(f"hit@{top_k}", f"{report.hit_at_k:.1%}")
            summary.add_row("MRR", f"{report.mrr:.3f}")
            summary.add_row("pages", str(stats.pages))
            summary.add_row("embeddings", str(stats.embeddings))
            summary.add_row("sources", ", ".join(f"{k}:{v}" for k, v in sorted(report.source_counts.items())) or "—")
            console.print(summary)

            details = Table(title="Golden Query Details")
            details.add_column("#", justify="right", style="dim")
            details.add_column("Query", max_width=38)
            details.add_column("Best", justify="right")
            details.add_column("Top hit", max_width=30)
            details.add_column("Expected", max_width=40)
            for i, row in enumerate(report.results, 1):
                top = row.top_hits[0] if row.top_hits else {}
                best = str(row.best_rank) if row.best_rank is not None else "MISS"
                details.add_row(
                    str(i),
                    row.query,
                    best,
                    str(top.get("title", "—")),
                    ", ".join(row.expected[:3]),
                )
            console.print(details)

            if report.misses:
                console.print("[yellow]Misses:[/yellow]")
                for miss in report.misses:
                    got = ", ".join(str(h.get("title")) for h in miss.top_hits[:3]) or "no hits"
                    console.print(f"  - {miss.query}: got {got}")
            if failed_thresholds:
                console.print("[red]Benchmark thresholds failed:[/red] " + "; ".join(failed_thresholds))
                raise typer.Exit(2)
        finally:
            await embedder.close()
            await db.close()

    asyncio.run(_run())


@wiki_app.command("curate")
def wiki_curate(
    apply: bool = typer.Option(False, "--apply", help="Apply low-risk curator actions"),
    since: Optional[str] = typer.Option(None, "--since", help="Only consider lessons since YYYY-MM-DD for rollup suggestions"),
    max_suggestions: int = typer.Option(30, "--max-suggestions", help="Maximum suggestions to show"),
    max_bench_additions: int = typer.Option(20, "--max-bench-additions", help="Maximum bench queries to append with --apply"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run the knowledge-curator sub-agent over the wiki."""
    _setup_logging(verbose)
    from .wiki.curator import KnowledgeCurator, parse_since
    from .wiki.store import WikiStore

    settings = get_settings()
    store = WikiStore(settings.WIKI_DIR)
    if not store.exists():
        console.print(f"[yellow]Wiki dir not found at {settings.WIKI_DIR}[/yellow]")
        raise typer.Exit(1)

    try:
        since_date = parse_since(since)
    except ValueError:
        console.print("[red]--since must be YYYY-MM-DD[/red]")
        raise typer.Exit(1)

    curator = KnowledgeCurator(store)
    report = curator.audit(since=since_date, max_suggestions=max_suggestions)
    if apply:
        report = curator.apply(report, max_bench_additions=max_bench_additions)

    if json_output:
        console.print(report.to_json())
        return

    summary = Table(title="Wiki Curator Report")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right")
    summary.add_row("pages", str(report.pages))
    summary.add_row("parse errors", str(report.errors))
    summary.add_row("broken links", str(report.broken_links))
    summary.add_row("findings", str(len(report.findings)))
    summary.add_row("suggestions", str(len(report.suggestions)))
    summary.add_row("applyable", str(len(report.applyable_suggestions)))
    console.print(summary)

    if report.findings:
        findings = Table(title="Findings")
        findings.add_column("Severity", style="yellow")
        findings.add_column("Category")
        findings.add_column("Path", max_width=36)
        findings.add_column("Message", max_width=48)
        for row in report.findings[:20]:
            findings.add_row(row.severity, row.category, row.path or "—", row.message)
        console.print(findings)

    if report.suggestions:
        suggestions = Table(title="Suggestions")
        suggestions.add_column("Action", style="cyan")
        suggestions.add_column("Target", max_width=34)
        suggestions.add_column("Apply", justify="center")
        suggestions.add_column("Reason", max_width=58)
        for row in report.suggestions[:max_suggestions]:
            suggestions.add_row(row.action, row.target, "yes" if row.applyable else "no", row.reason)
        console.print(suggestions)

    if report.applied:
        console.print("[green]Applied:[/green]")
        for item in report.applied:
            console.print(f"  - {item}")
    elif report.applyable_suggestions:
        console.print("[dim]Run with --apply to append deterministic bench coverage suggestions.[/dim]")


@wiki_app.command("import-wq")
def wiki_import_wq(
    region: Optional[str] = typer.Option(None, "--region", help="Override WQ_REGION"),
    universe: Optional[str] = typer.Option(None, "--universe", help="Override WQ_UNIVERSE"),
    delay: Optional[int] = typer.Option(None, "--delay", help="Override WQ_DELAY"),
    limit_per_dataset: Optional[int] = typer.Option(None, "--limit-per-dataset", help="Cap fields per dataset (only matters with --with-fields)"),
    with_fields: bool = typer.Option(False, "--with-fields", help="Also write 7000+ per-field pages (usually unneeded; metadata is already inlined into dataset pages + dictionary)"),
    reindex: bool = typer.Option(True, "--reindex/--no-reindex", help="Rebuild wiki index after import"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Import official WQ Brain docs (operators / datasets, optionally per-field pages) into wiki/."""
    _setup_logging(verbose)
    from .wq.client import WQClient
    from .wiki.importers import WQDocImporter
    from .wiki.store import WikiStore
    from .wiki.index import WikiIndex
    from .wiki.embeddings import make_embedding_provider
    from .db import Database
    from pathlib import Path

    async def _run():
        settings = get_settings()
        client = WQClient(settings)
        await client.connect()
        try:
            importer = WQDocImporter(Path(settings.WIKI_DIR), client)
            stats = await importer.import_all(
                region=region,
                universe=universe,
                delay=delay,
                limit_per_dataset=limit_per_dataset,
                include_fields=with_fields,
            )
            table = Table(title="WQ docs imported")
            table.add_column("Kind", style="cyan")
            table.add_column("Pages written", justify="right")
            for k, v in stats.__dict__.items():
                table.add_row(k, str(v))
            console.print(table)
        finally:
            await client.close()

        if reindex:
            store = WikiStore(settings.WIKI_DIR)
            db = Database(settings.DB_PATH)
            await db.connect()
            embedder = make_embedding_provider(settings)
            idx = WikiIndex(store=store, db=db, embedder=embedder,
                             grep_weight=settings.WIKI_GREP_WEIGHT,
                             vector_weight=settings.WIKI_VECTOR_WEIGHT)
            try:
                istats = await idx.build(incremental=True)
                console.print(f"[dim]Reindexed: {istats.pages} pages, {istats.embeddings} embeddings[/dim]")
            finally:
                await embedder.close()
                await db.close()

    asyncio.run(_run())


@wiki_app.command("import-paper")
def wiki_import_paper(
    url: Optional[str] = typer.Option(None, "--url", help="arxiv.org/abs/<id> or papers.ssrn.com/...?abstract_id=<id>"),
    manual: bool = typer.Option(False, "--manual", help="Provide all fields by flag instead of fetching"),
    title: Optional[str] = typer.Option(None, "--title"),
    authors: Optional[str] = typer.Option(None, "--authors", help="Comma-separated"),
    year: Optional[int] = typer.Option(None, "--year"),
    abstract: Optional[str] = typer.Option(None, "--abstract"),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated extra tags"),
    reindex: bool = typer.Option(False, "--reindex", help="Rebuild wiki index after writing"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Import a research paper into wiki/papers/ (arxiv / SSRN auto-fetch, or --manual)."""
    _setup_logging(verbose)
    from .wiki.importers import PaperImporter
    from .wiki.store import WikiStore
    from .wiki.index import WikiIndex
    from .wiki.embeddings import make_embedding_provider
    from .db import Database
    from pathlib import Path

    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]

    async def _run():
        settings = get_settings()
        importer = PaperImporter(Path(settings.WIKI_DIR))
        if manual:
            if not title:
                console.print("[red]--manual requires --title (and optionally --authors / --year / --url / --abstract)[/red]")
                raise typer.Exit(1)
            path = importer.import_manual(
                title=title,
                authors=[a.strip() for a in (authors or "").split(",") if a.strip()],
                year=year,
                url=url or "",
                abstract=abstract or "",
                tags=tag_list,
            )
        else:
            if not url:
                console.print("[red]Provide --url or use --manual[/red]")
                raise typer.Exit(1)
            path = await importer.import_url(url, tags=tag_list)
        console.print(f"[green]Wrote {path}[/green]")

        if reindex:
            store = WikiStore(settings.WIKI_DIR)
            db = Database(settings.DB_PATH)
            await db.connect()
            embedder = make_embedding_provider(settings)
            idx = WikiIndex(store=store, db=db, embedder=embedder,
                             grep_weight=settings.WIKI_GREP_WEIGHT,
                             vector_weight=settings.WIKI_VECTOR_WEIGHT)
            try:
                await idx.build(incremental=True)
            finally:
                await embedder.close()
                await db.close()

    asyncio.run(_run())


if __name__ == "__main__":
    app()

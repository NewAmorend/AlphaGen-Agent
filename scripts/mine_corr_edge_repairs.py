from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alphagen_agent.engine.fast_expr import validate_fast_expr


NOW = datetime.now().isoformat()


def candidates() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []

    ret = "rank(reverse(ts_zscore(ts_sum(returns, 60), 63)))"
    market_corr = (
        "rank(reverse(ts_zscore(ts_mean(correlation_last_60_days_spy, 20), 63)))"
    )
    market_base = f"add({ret}, {market_corr})"
    rows.extend(
        [
            ("market_corr_residual", f"ts_decay_linear(vector_neut({market_base}, {ret}), 5)"),
            (
                "market_corr_residual",
                f"group_neutralize(ts_decay_linear(vector_neut({market_base}, {ret}), 7), subindustry)",
            ),
            (
                "market_corr_weight",
                f"ts_decay_linear(add(multiply({ret}, 0.35), {market_corr}), 5)",
            ),
            (
                "market_corr_weight",
                f"ts_decay_linear(add({ret}, multiply({market_corr}, 0.35)), 5)",
            ),
            (
                "market_corr_interaction",
                f"group_neutralize(ts_decay_linear(multiply({ret}, {market_corr}), 7), subindustry)",
            ),
            (
                "market_corr_90",
                "ts_decay_linear(add(rank(reverse(ts_zscore(ts_sum(returns, 40), 63))), rank(reverse(ts_zscore(ts_mean(correlation_last_90_days_spy, 20), 63)))), 7)",
            ),
        ]
    )

    price_shape = "rank(reverse(group_rank(ts_av_diff(close, 20), subindustry)))"
    buzz_flow = "rank(ts_corr(snt_buzz_ret, returns, 60))"
    flow_base = f"multiply({price_shape}, {buzz_flow})"
    short_reversal = "rank(reverse(group_rank(ts_delta(close, 7), subindustry)))"
    rows.extend(
        [
            (
                "buzz_flow_residual",
                f"group_neutralize(ts_decay_linear(vector_neut({flow_base}, {short_reversal}), 10), subindustry)",
            ),
            (
                "buzz_flow_residual",
                f"ts_decay_linear(vector_neut({flow_base}, {price_shape}), 10)",
            ),
            (
                "buzz_flow_sector",
                f"ts_decay_linear(group_rank({flow_base}, sector), 10)",
            ),
            (
                "buzz_flow_vwap",
                "group_neutralize(ts_decay_linear(multiply(rank(reverse(group_rank(ts_av_diff(vwap, 20), subindustry))), rank(ts_corr(snt_buzz_ret, returns, 60))), 10), subindustry)",
            ),
            (
                "buzz_flow_alt",
                "group_neutralize(ts_decay_linear(multiply(rank(reverse(group_rank(ts_av_diff(close, 20), subindustry))), rank(ts_corr(scl12_buzz, returns, 60))), 10), subindustry)",
            ),
        ]
    )

    short = "reverse(group_zscore(ts_delta(close, 7), subindustry))"
    long = "reverse(group_rank(ts_zscore(ts_mean(returns, 60), 40), subindustry))"
    reversal_base = f"add({short}, {long})"
    rows.extend(
        [
            (
                "reversal_residual",
                f"group_neutralize(ts_decay_linear(vector_neut({reversal_base}, rank({short})), 5), subindustry)",
            ),
            (
                "reversal_weight",
                f"ts_decay_linear(add(multiply({short}, 0.35), {long}), 7)",
            ),
            (
                "reversal_interaction",
                f"group_neutralize(ts_decay_linear(multiply(rank({short}), rank({long})), 5), subindustry)",
            ),
            (
                "reversal_sector",
                "ts_decay_linear(add(reverse(group_zscore(ts_delta(close, 7), sector)), reverse(group_rank(ts_zscore(ts_mean(returns, 60), 40), sector))), 5)",
            ),
        ]
    )
    return rows


def main(db_path: str, limit: int, dry_run: bool) -> None:
    con = sqlite3.connect(db_path)
    existing = {r[0] for r in con.execute("SELECT expression FROM alphas")}
    selected: list[tuple[str, str]] = []

    for theme, expr in candidates():
        result = validate_fast_expr(expr)
        if not result.valid:
            print(f"skip_invalid [{theme}] {expr}")
            for issue in result.issues:
                print(f"  - {issue.code}: {issue.message}")
            continue
        if expr in existing:
            continue
        selected.append((theme, expr))
        if len(selected) >= limit:
            break

    print(f"selected={len(selected)} dry_run={dry_run}")
    for index, (theme, expr) in enumerate(selected, start=1):
        print(f"{index:02d} [{theme}] {expr}")

    if dry_run or not selected:
        return

    con.executemany(
        """INSERT INTO alphas (expression, strategy, llm_model, status, created_at)
           VALUES (?, 'factor_mining', ?, 'generated', ?)""",
        [(expr, theme, NOW) for theme, expr in selected],
    )
    con.commit()
    first_id = con.execute(
        "SELECT max(id) - ? + 1 FROM alphas", (len(selected),)
    ).fetchone()[0]
    last_id = con.execute("SELECT max(id) FROM alphas").fetchone()[0]
    print(f"inserted_ids={first_id}..{last_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="alphagen_agent.db")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.db, args.limit, args.dry_run)

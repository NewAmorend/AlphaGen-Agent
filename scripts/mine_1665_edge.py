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

    for price_window, buzz_window, decay in (
        (15, 60, 10),
        (20, 50, 10),
        (20, 70, 10),
        (25, 60, 10),
        (30, 60, 10),
        (20, 60, 12),
        (20, 60, 15),
        (20, 60, 20),
    ):
        raw = (
            f"multiply(rank(reverse(group_rank(ts_av_diff(close, {price_window}), subindustry))), "
            f"rank(ts_corr(snt_buzz_ret, returns, {buzz_window})))"
        )
        rows.append(
            (
                "buzz_sector_edge",
                f"ts_decay_linear(group_rank({raw}, sector), {decay})",
            )
        )

    raw = (
        "multiply(rank(reverse(group_rank(ts_av_diff(close, 20), subindustry))), "
        "rank(ts_corr(snt_buzz_ret, returns, 60)))"
    )
    rows.extend(
        [
            (
                "buzz_industry_edge",
                f"ts_decay_linear(group_rank({raw}, industry), 10)",
            ),
            (
                "buzz_sector_residual",
                f"ts_decay_linear(vector_neut(group_rank({raw}, sector), rank(ts_zscore(implied_volatility_mean_60, 63))), 10)",
            ),
            (
                "buzz_sector_residual",
                f"ts_decay_linear(vector_neut(group_rank({raw}, sector), rank(reverse(group_rank(ts_delta(close, 7), subindustry)))), 10)",
            ),
            (
                "buzz_sector_neutral",
                f"group_neutralize(ts_decay_linear(group_rank({raw}, sector), 10), subindustry)",
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
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.db, args.limit, args.dry_run)

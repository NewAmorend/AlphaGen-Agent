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
    bases = {
        "jc5_tvr": "group_rank(ts_zscore(mdl53_jc5_3year, 8), subindustry)",
        "mood_tvr": (
            "multiply(rank(ts_delta(daily_equity_mood_indicator, 5)), "
            "reverse(rank(ts_delta(close, 5))))"
        ),
        "vwap_tvr": "group_rank(subtract(vwap, close), subindustry)",
        "sector_reversal_tvr": "reverse(group_rank(ts_zscore(returns, 20), sector))",
    }
    rows: list[tuple[str, str]] = []
    for theme, base in bases.items():
        for target in (0.08, 0.12, 0.18):
            rows.append(
                (theme, f"ts_target_tvr_decay({base}, target_tvr={target})")
            )
        for limit in (0.01, 0.03):
            rows.append((theme, f"hump({base}, hump={limit})"))

    # Conventional smoothing is retained as a fallback in case target-TVR
    # simulation is unavailable for the current Brain configuration.
    for window in (10, 20, 40):
        rows.append(
            (
                "vwap_decay",
                f"group_neutralize(ts_decay_linear(group_rank(subtract(vwap, close), subindustry), {window}), subindustry)",
            )
        )
        rows.append(
            (
                "sector_reversal_decay",
                f"ts_decay_linear(reverse(group_rank(ts_zscore(returns, 20), sector)), {window})",
            )
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
    parser.add_argument("--limit", type=int, default=26)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.db, args.limit, args.dry_run)

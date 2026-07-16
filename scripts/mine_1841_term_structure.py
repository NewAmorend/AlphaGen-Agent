from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alphagen_agent.engine.fast_expr import validate_fast_expr


NOW = datetime.now().isoformat()


def leg(field: str, z_window: int = 10, decay: int = 5) -> str:
    return (
        f"ts_decay_linear(group_rank(ts_zscore({field}, {z_window}), "
        f"subindustry), {decay})"
    )


def wrap(signal: str, rank_window: int = 40) -> str:
    return f"rank(ts_rank({signal}, {rank_window}))"


def candidates() -> list[tuple[str, str]]:
    """Tight neighborhood around clean HIGH alpha #1841.

    #1841 combines the slow JC5 seven-year leg with the faster JC6 one-year
    leg, then time-ranks the composite.  Change one design axis at a time so
    each simulation teaches us something about the local response surface.
    """
    jc5_7y = leg("mdl53_jc5_7year")
    jc6_1y = leg("mdl53_jc6_1year")
    base = f"add({jc5_7y}, {jc6_1y})"

    rows: list[tuple[str, str]] = []

    # Outer persistence horizon around the winning 40-day rank.
    for window in (20, 30, 50, 60):
        rows.append((f"term1841_outer_{window}", wrap(base, window)))

    # Inner normalization / smoothing: coordinated changes preserve symmetry.
    for z_window, decay in ((8, 3), (15, 5), (15, 7), (20, 10)):
        signal = f"add({leg('mdl53_jc5_7year', z_window, decay)}, {leg('mdl53_jc6_1year', z_window, decay)})"
        rows.append((f"term1841_inner_z{z_window}_d{decay}", wrap(signal)))

    # Relative leg weights test whether the slow or fast horizon drives alpha.
    rows.extend(
        [
            ("term1841_slow_weight", wrap(f"add(multiply({jc5_7y}, 1.5), {jc6_1y})")),
            ("term1841_fast_weight", wrap(f"add({jc5_7y}, multiply({jc6_1y}, 1.5))")),
        ]
    )

    # Adjacent, already observed fields test the economic term-structure idea.
    rows.extend(
        [
            (
                "term1841_short_anchor",
                wrap(f"add({jc5_7y}, {leg('mdl53_jc6_1month')})"),
            ),
            (
                "term1841_matched_long",
                wrap(f"add({jc5_7y}, {leg('mdl53_jc6_5year')})"),
            ),
        ]
    )
    return rows


def main(db_path: str, limit: int, dry_run: bool) -> None:
    con = sqlite3.connect(db_path)
    existing = {row[0] for row in con.execute("SELECT expression FROM alphas")}
    selected: list[tuple[str, str]] = []

    for theme, expression in candidates():
        result = validate_fast_expr(expression)
        if not result.valid:
            print(f"skip_invalid [{theme}] {expression}")
            for issue in result.issues:
                print(f"  - {issue.code}: {issue.message}")
            continue
        if expression in existing:
            continue
        selected.append((theme, expression))
        if len(selected) >= limit:
            break

    print(f"selected={len(selected)} dry_run={dry_run}")
    for index, (theme, expression) in enumerate(selected, start=1):
        print(f"{index:02d} [{theme}] {expression}")

    if dry_run or not selected:
        return

    con.executemany(
        """INSERT INTO alphas (expression, strategy, llm_model, status, created_at)
           VALUES (?, 'factor_mining', ?, 'generated', ?)""",
        [(expression, theme, NOW) for theme, expression in selected],
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

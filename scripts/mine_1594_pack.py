from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alphagen_agent.db import expression_skeleton
from alphagen_agent.engine.fast_expr import validate_fast_expr


NOW = datetime.now().isoformat()


def candidates() -> list[tuple[str, str]]:
    """Follow-ups to the only currently clean HIGH candidate (#1594).

    #1594 raw signal:
      sentiment_supply * corr(volume, returns, 40)
    then vector_neut versus short price delta.

    This pack changes the neutralization basis or adds regime gates to seek
    more clean HIGH alphas without copying the same PnL path.
    """
    rows: list[tuple[str, str]] = []

    def add(theme: str, expr: str) -> None:
        rows.append((theme, expr))

    raw40 = "multiply(rank(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry)), rank(ts_corr(volume, returns, 40)))"
    raw60 = "multiply(rank(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry)), rank(ts_corr(volume, returns, 60)))"
    raw_w60 = "multiply(rank(winsorize(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry), std=4)), rank(ts_corr(volume, returns, 60)))"

    # Alternative orthogonalization bases.
    for basis, suffix in (
        ("rank(ts_delta(close, 10))", "price10"),
        ("rank(ts_av_diff(close, 20))", "avdiff20"),
        ("rank(ts_zscore(ts_sum(returns, 20), 60))", "ret20"),
        ("rank(ts_zscore(implied_volatility_mean_60, 63))", "iv60"),
        ("rank(ts_zscore(mdl53_jc5_3year, 10))", "jc5"),
    ):
        add(
            f"pack1594_neut_{suffix}",
            f"group_neutralize(ts_decay_linear(vector_neut({raw40}, {basis}), 7), subindustry)",
        )

    # Window + rank family changes.
    add(
        "pack1594_window_rank",
        f"group_neutralize(ts_decay_linear(vector_neut({raw60}, rank(ts_delta(close, 5))), 10), subindustry)",
    )
    add(
        "pack1594_window_rank",
        "group_neutralize(ts_decay_linear(vector_neut(multiply(group_rank(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry), subindustry), group_rank(ts_corr(volume, returns, 60), subindustry)), rank(ts_delta(close, 5))), 10), subindustry)",
    )
    add(
        "pack1594_window_rank",
        f"ts_decay_linear(vector_neut({raw_w60}, rank(ts_delta(close, 5))), 10)",
    )

    # Regime-gated variants. These trade less often and can reduce correlation
    # to always-on submitted references.
    base1594 = f"group_neutralize(ts_decay_linear(vector_neut({raw40}, rank(ts_delta(close, 5))), 7), subindustry)"
    add(
        "pack1594_gate",
        f"trade_when(rank(ts_corr(volume, returns, 20)), {base1594}, 0)",
    )
    add(
        "pack1594_gate",
        f"trade_when(reverse(rank(ts_zscore(implied_volatility_mean_60, 63))), {base1594}, 0)",
    )
    add(
        "pack1594_gate",
        f"trade_when(reverse(rank(beta_last_60_days_spy)), {base1594}, 0)",
    )
    add(
        "pack1594_gate",
        f"trade_when(rank(ts_zscore(snt_buzz, 20)), {base1594}, 0)",
    )

    # Add a second neutralization after nonlinear compression.
    add(
        "pack1594_compress",
        f"group_neutralize(ts_decay_linear(signed_power(vector_neut({raw40}, rank(ts_delta(close, 5))), 0.5), 7), subindustry)",
    )
    add(
        "pack1594_compress",
        f"group_neutralize(ts_decay_linear(winsorize(vector_neut({raw40}, rank(ts_delta(close, 5))), std=4), 7), subindustry)",
    )

    return rows


def main(db_path: str, limit: int, dry_run: bool) -> None:
    con = sqlite3.connect(db_path)
    existing_exprs = {r[0] for r in con.execute("SELECT expression FROM alphas")}
    existing_skeletons = {expression_skeleton(expr) for expr in existing_exprs}

    selected: list[tuple[str, str]] = []
    seen_skeletons = set(existing_skeletons)
    for theme, expr in candidates():
        result = validate_fast_expr(expr)
        if not result.valid:
            print(f"skip_invalid [{theme}] {expr}")
            for issue in result.issues:
                print(f"  - {issue.code}: {issue.message}")
            continue
        skel = expression_skeleton(expr)
        if expr in existing_exprs or skel in seen_skeletons:
            continue
        selected.append((theme, expr))
        seen_skeletons.add(skel)
        if len(selected) >= limit:
            break

    print(f"selected={len(selected)} dry_run={dry_run}")
    for idx, (theme, expr) in enumerate(selected, start=1):
        print(f"{idx:02d} [{theme}] {expr}")

    if dry_run or not selected:
        return

    con.executemany(
        """INSERT INTO alphas (expression, strategy, llm_model, status, created_at)
           VALUES (?, 'factor_mining', ?, 'generated', ?)""",
        [(expr, theme, NOW) for theme, expr in selected],
    )
    con.commit()
    first_id = con.execute("SELECT max(id) - ? + 1 FROM alphas", (len(selected),)).fetchone()[0]
    last_id = con.execute("SELECT max(id) FROM alphas").fetchone()[0]
    print(f"inserted_ids={first_id}..{last_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="alphagen_agent.db")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.db, args.limit, args.dry_run)

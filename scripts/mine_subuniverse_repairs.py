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
    """Repair candidates for #1553 and adjacent medium alphas.

    #1553: fitness=1.00, sharpe=1.37, turnover=0.115, only failed
    LOW_SUB_UNIVERSE_SHARPE. The base signal is sentiment supply interacted with
    volume/return correlation, which is less crowded than the submitted
    add(jc5, sentiment) and pure return-reversal families.
    """
    rows: list[tuple[str, str]] = []

    def add(theme: str, expr: str) -> None:
        rows.append((theme, expr))

    # Direct #1553 repairs: stabilize cross-section and sub-universe behavior.
    for corr_window, decay in ((30, 7), (40, 7), (60, 10), (80, 10)):
        add(
            "repair_1553_window",
            f"group_neutralize(ts_decay_linear(multiply(rank(winsorize(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry), std=4)), rank(ts_corr(volume, returns, {corr_window}))), {decay}), subindustry)",
        )
        add(
            "repair_1553_window",
            f"group_neutralize(ts_decay_linear(multiply(group_rank(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry), subindustry), group_rank(ts_corr(volume, returns, {corr_window}), subindustry)), {decay}), subindustry)",
        )

    # Nonlinear compression reduces concentrated weight and may alter PnL shape.
    add(
        "repair_1553_nonlinear",
        "group_neutralize(ts_decay_linear(multiply(signed_power(rank(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry)), 0.5), rank(ts_corr(volume, returns, 40))), 7), subindustry)",
    )
    add(
        "repair_1553_nonlinear",
        "ts_decay_linear(multiply(signed_power(rank(winsorize(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry), std=4)), 0.5), group_rank(ts_corr(volume, returns, 60), subindustry)), 10)",
    )

    # Gated versions: trade only when attention/liquidity regime supports the
    # signal; target lower correlation to always-on submitted alphas.
    add(
        "repair_1553_gate",
        "trade_when(rank(ts_zscore(snt_buzz, 20)), group_neutralize(ts_decay_linear(multiply(rank(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry)), rank(ts_corr(volume, returns, 40))), 7), subindustry), 0)",
    )
    add(
        "repair_1553_gate",
        "trade_when(rank(divide(volume, adv20)), group_neutralize(ts_decay_linear(multiply(rank(winsorize(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry), std=4)), rank(ts_corr(volume, returns, 60))), 10), subindustry), 0)",
    )
    add(
        "repair_1553_gate",
        "trade_when(reverse(rank(ts_zscore(implied_volatility_mean_60, 63))), group_neutralize(ts_decay_linear(multiply(rank(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry)), rank(ts_corr(volume, returns, 40))), 7), subindustry), 0)",
    )

    # Orthogonalized variants: keep the sentiment-flow signal but subtract the
    # most crowded short-reversal / jc5 direction.
    add(
        "repair_1553_residual",
        "group_neutralize(ts_decay_linear(vector_neut(multiply(rank(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry)), rank(ts_corr(volume, returns, 40))), rank(ts_delta(close, 5))), 7), subindustry)",
    )
    add(
        "repair_1553_residual",
        "group_neutralize(ts_decay_linear(vector_neut(multiply(rank(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry)), rank(ts_corr(volume, returns, 60))), rank(ts_zscore(mdl53_jc5_3year, 10))), 10), subindustry)",
    )

    # Adjacent repair around #1561/#1564: gate rather than add/multiply with jc5.
    add(
        "repair_1561_gate",
        "trade_when(rank(ts_corr(volume, returns, 40)), group_neutralize(ts_decay_linear(group_rank(ts_zscore(mdl53_jc5_3year, 10), subindustry), 7), subindustry), 0)",
    )
    add(
        "repair_1561_gate",
        "trade_when(rank(ts_zscore(scl12_sentiment, 20)), group_neutralize(ts_decay_linear(group_rank(ts_zscore(mdl53_jc5_3year, 10), subindustry), 7), subindustry), 0)",
    )

    # Alternative sentiment fields, still paired with volume-flow not return
    # reversal, to avoid the dominant submitted clusters.
    add(
        "repair_alt_sentiment_flow",
        "group_neutralize(ts_decay_linear(multiply(rank(ts_zscore(scl12_sentiment, 20)), rank(ts_corr(volume, returns, 40))), 7), subindustry)",
    )
    add(
        "repair_alt_sentiment_flow",
        "group_neutralize(ts_decay_linear(multiply(rank(ts_delta(daily_equity_mood_indicator, 5)), rank(ts_corr(volume, returns, 40))), 7), subindustry)",
    )
    add(
        "repair_alt_sentiment_flow",
        "ts_decay_linear(multiply(rank(ts_corr(snt_buzz_ret, returns, 60)), rank(ts_corr(volume, returns, 40))), 7)",
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
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.db, args.limit, args.dry_run)

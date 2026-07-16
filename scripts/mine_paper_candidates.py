from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime

from alphagen_agent.db import expression_skeleton


NOW = datetime.now().isoformat()


def candidates() -> list[tuple[str, str]]:
    """Small paper-inspired batch using fields already seen in this workspace."""
    rows: list[tuple[str, str]] = []

    def add(theme: str, expr: str) -> None:
        rows.append((theme, expr))

    # 101 Formulaic Alphas / AutoAlpha: price-volume short horizon reversals.
    for n in (3, 5, 7, 10):
        add("101_price_shock_reversal", f"ts_decay_linear(reverse(group_zscore(ts_delta(close, {n}), subindustry)), 5)")
        add("101_price_shock_reversal", f"group_neutralize(ts_decay_linear(reverse(rank(ts_delta(close, {n}))), 5), subindustry)")
    for n in (10, 20):
        add("101_av_diff_reversal", f"ts_decay_linear(reverse(group_rank(ts_av_diff(close, {n}), subindustry)), 5)")
        add("101_av_diff_reversal", f"group_neutralize(winsorize(reverse(group_rank(ts_av_diff(close, {n}), subindustry)), std=4), subindustry)")
    add(
        "101_volume_gated_reversal",
        "ts_decay_linear(multiply(reverse(group_rank(ts_delta(close, 5), subindustry)), rank(divide(volume, adv20))), 5)",
    )

    # Attention-factor style: residualized or industry-neutral reversal legs.
    add(
        "residual_reversal_attention",
        "ts_decay_linear(group_neutralize(add(reverse(group_zscore(ts_delta(close, 7), subindustry)), reverse(group_rank(ts_av_diff(close, 20), subindustry))), subindustry), 5)",
    )
    add(
        "residual_reversal_attention",
        "ts_decay_linear(add(reverse(group_zscore(ts_delta(close, 7), subindustry)), reverse(group_rank(ts_zscore(ts_mean(returns, 60), 40), subindustry))), 5)",
    )
    add(
        "residual_reversal_attention",
        "group_neutralize(ts_decay_linear(add(reverse(rank(ts_zscore(ts_sum(returns, 5), 20))), reverse(group_rank(ts_av_diff(close, 20), subindustry))), 5), subindustry)",
    )

    # Sentiment/buzz interaction: inspired by AutoAlpha-style composition, using local strong fields.
    for corr_window in (40, 60):
        add(
            "sentiment_flow_interaction",
            f"ts_decay_linear(multiply(rank(group_neutralize(ts_mean(signed_power(mdl177_5shortsentimentfactor_dmd_supply, 0.5), 20), subindustry)), rank(ts_corr(snt_buzz, returns, {corr_window}))), 10)",
        )
        add(
            "sentiment_flow_interaction",
            f"group_neutralize(ts_decay_linear(multiply(rank(winsorize(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry), std=4)), rank(ts_corr(snt_buzz_ret, returns, {corr_window}))), 10), subindustry)",
        )

    # Intangible / quality-adjacent low-turnover legs available in the local history.
    for short, long in ((15, 90), (20, 100), (25, 140)):
        add(
            "quality_intangible_proxy",
            f"group_neutralize(add(rank(subtract(ts_decay_linear(fscore_bfl_value, {short}), ts_decay_linear(fscore_bfl_value, {long}))), rank(winsorize(mdl77_divcov, std=4))), subindustry)",
        )
        add(
            "quality_intangible_proxy",
            f"ts_decay_linear(group_neutralize(add(rank(subtract(ts_decay_linear(fscore_bfl_value, {short}), ts_decay_linear(fscore_bfl_value, {long}))), rank(mdl77_divcov)), subindustry), 5)",
        )

    # Analyst revision / earnings momentum variants, nudged away from plain rank(ts_zscore(field, 63)).
    for field in (
        "earnings_momentum_analyst_score",
        "earnings_revision_magnitude",
        "mdl177_earningmomentumfactor_sue",
        "mdl177_earningmomentumfactor_epsrm",
    ):
        add("revision_momentum", f"group_neutralize(ts_decay_linear(rank(ts_delta({field}, 20)), 10), subindustry)")
        add("revision_momentum", f"ts_decay_linear(group_rank(ts_zscore({field}, 63), subindustry), 10)")

    return rows


def main(db_path: str, limit: int, dry_run: bool) -> None:
    con = sqlite3.connect(db_path)
    existing_exprs = {r[0] for r in con.execute("SELECT expression FROM alphas")}
    existing_skeletons = {expression_skeleton(expr) for expr in existing_exprs}

    selected: list[tuple[str, str]] = []
    seen_skeletons = set(existing_skeletons)
    for theme, expr in candidates():
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
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.db, args.limit, args.dry_run)

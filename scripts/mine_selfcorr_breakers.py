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
    """Variants for paper_v3 HIGH alphas that failed WQ self-correlation.

    The two source alphas were strong but crowded:
    - #1523: sentiment supply + short price reversal, self-corr 0.9127
    - #1545: mdl53_jc5_3year + sentiment supply, self-corr 0.9769

    These candidates keep the same economic themes while changing the PnL
    shape via interaction, gating, residualization, nonlinear compression, and
    substitute attention/quality fields.
    """
    rows: list[tuple[str, str]] = []

    def add(theme: str, expr: str) -> None:
        rows.append((theme, expr))

    # Breakers for #1523: avoid add(sentiment, short-reversal) as the main shape.
    add(
        "selfcorr_breaker_1523_interaction",
        "group_neutralize(ts_decay_linear(multiply(rank(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry)), reverse(group_rank(ts_delta(close, 7), subindustry))), 5), subindustry)",
    )
    add(
        "selfcorr_breaker_1523_interaction",
        "ts_decay_linear(multiply(signed_power(rank(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry)), 0.5), reverse(rank(ts_zscore(ts_sum(returns, 5), 20)))), 7)",
    )
    add(
        "selfcorr_breaker_1523_attention_gate",
        "trade_when(rank(ts_zscore(snt_buzz, 20)), group_neutralize(ts_decay_linear(reverse(group_rank(ts_delta(close, 5), subindustry)), 5), subindustry), 0)",
    )
    add(
        "selfcorr_breaker_1523_attention_gate",
        "trade_when(rank(ts_corr(snt_buzz_ret, returns, 40)), group_neutralize(ts_decay_linear(rank(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry)), 5), subindustry), 0)",
    )
    add(
        "selfcorr_breaker_1523_residual",
        "group_neutralize(ts_decay_linear(vector_neut(rank(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry)), rank(ts_delta(close, 7))), 7), subindustry)",
    )
    add(
        "selfcorr_breaker_1523_residual",
        "ts_decay_linear(vector_neut(rank(ts_zscore(scl12_sentiment, 20)), reverse(rank(ts_delta(close, 5)))), 5)",
    )
    add(
        "selfcorr_breaker_1523_volume_flow",
        "group_neutralize(ts_decay_linear(multiply(rank(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry)), rank(ts_corr(volume, returns, 20))), 5), subindustry)",
    )
    add(
        "selfcorr_breaker_1523_volume_flow",
        "ts_decay_linear(multiply(rank(ts_corr(snt_buzz, volume, 20)), reverse(rank(ts_av_diff(close, 20)))), 5)",
    )
    add(
        "selfcorr_breaker_1523_substitute_sentiment",
        "group_neutralize(ts_decay_linear(multiply(rank(ts_delta(scl12_sentiment_fast_d1, 5)), reverse(group_rank(ts_delta(close, 5), subindustry))), 5), subindustry)",
    )
    add(
        "selfcorr_breaker_1523_substitute_sentiment",
        "ts_target_tvr_decay(multiply(rank(ts_zscore(daily_equity_mood_indicator, 20)), reverse(rank(ts_delta(close, 5)))), target_tvr=0.18)",
    )

    # Breakers for #1545: avoid add(rank(jc5), rank(sentiment_supply)).
    add(
        "selfcorr_breaker_1545_interaction",
        "group_neutralize(ts_decay_linear(multiply(rank(ts_zscore(mdl53_jc5_3year, 10)), rank(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry))), 5), subindustry)",
    )
    add(
        "selfcorr_breaker_1545_interaction",
        "ts_decay_linear(multiply(rank(ts_delta(mdl53_jc5_3year, 20)), signed_power(rank(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry)), 0.5)), 10)",
    )
    add(
        "selfcorr_breaker_1545_residual_quality",
        "group_neutralize(ts_decay_linear(vector_neut(rank(ts_zscore(mdl53_jc5_3year, 10)), rank(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry))), 7), subindustry)",
    )
    add(
        "selfcorr_breaker_1545_residual_quality",
        "ts_decay_linear(vector_neut(rank(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry)), rank(ts_zscore(mdl53_jc5_3year, 10))), 7)",
    )
    add(
        "selfcorr_breaker_1545_attention_gate",
        "trade_when(rank(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry)), ts_decay_linear(group_rank(ts_zscore(mdl53_jc5_3year, 10), subindustry), 5), 0)",
    )
    add(
        "selfcorr_breaker_1545_attention_gate",
        "trade_when(rank(ts_corr(snt_buzz_ret, returns, 40)), ts_decay_linear(rank(ts_zscore(mdl53_jc5_3year, 10)), 5), 0)",
    )
    add(
        "selfcorr_breaker_1545_substitute_quality",
        "group_neutralize(ts_decay_linear(multiply(rank(ts_zscore(mdl53_jc5_1year, 10)), rank(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry))), 5), subindustry)",
    )
    add(
        "selfcorr_breaker_1545_substitute_quality",
        "group_neutralize(ts_decay_linear(multiply(rank(fscore_bfl_quality), rank(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry))), 10), subindustry)",
    )
    add(
        "selfcorr_breaker_1545_substitute_sentiment",
        "group_neutralize(ts_decay_linear(multiply(rank(ts_zscore(mdl53_jc5_3year, 10)), rank(ts_zscore(scl12_sentiment, 20))), 5), subindustry)",
    )
    add(
        "selfcorr_breaker_1545_substitute_sentiment",
        "ts_target_tvr_decay(multiply(rank(ts_zscore(mdl53_jc5_3year, 10)), rank(ts_corr(snt_buzz_ret, returns, 40))), target_tvr=0.16)",
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

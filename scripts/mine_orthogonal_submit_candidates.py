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
    """Candidates designed against the known submitted-correlation basis.

    Submitted clusters are dominated by:
    - short/medium return reversal
    - sentiment supply level and buzz/return correlation
    - jc5 quality + sentiment
    - fscore_bfl_value + mdl77_divcov
    - IV + reversal composites

    The best clean candidate so far (#1594) uses vector_neut on a
    sentiment-flow signal vs short price delta. This batch generalizes that
    pattern across several raw signals and orthogonalization bases.
    """
    rows: list[tuple[str, str]] = []

    def add(theme: str, expr: str) -> None:
        rows.append((theme, expr))

    # Sentiment-flow residuals: stay close to #1594 but use different flow
    # windows/bases so we do not just make a clone.
    for corr_window, decay in ((30, 7), (60, 10), (80, 10)):
        raw = (
            f"multiply(rank(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry)), "
            f"rank(ts_corr(volume, returns, {corr_window})))"
        )
        add(
            "orth_sentiment_flow_price",
            f"group_neutralize(ts_decay_linear(vector_neut({raw}, rank(ts_delta(close, 5))), {decay}), subindustry)",
        )
        add(
            "orth_sentiment_flow_jc5",
            f"group_neutralize(ts_decay_linear(vector_neut({raw}, rank(ts_zscore(mdl53_jc5_3year, 10))), {decay}), subindustry)",
        )

    # Substitute sentiment fields: lower chance of matching submitted sentiment
    # supply PnL exactly.
    add(
        "orth_alt_sentiment_flow",
        "group_neutralize(ts_decay_linear(vector_neut(multiply(rank(ts_zscore(scl12_sentiment, 20)), rank(ts_corr(volume, returns, 40))), rank(ts_delta(close, 5))), 7), subindustry)",
    )
    add(
        "orth_alt_sentiment_flow",
        "group_neutralize(ts_decay_linear(vector_neut(multiply(rank(ts_delta(daily_equity_mood_indicator, 5)), rank(ts_corr(volume, returns, 40))), rank(ts_delta(close, 5))), 7), subindustry)",
    )
    add(
        "orth_alt_sentiment_flow",
        "ts_decay_linear(vector_neut(multiply(rank(ts_corr(snt_buzz_ret, returns, 60)), rank(ts_corr(volume, returns, 40))), rank(ts_delta(close, 5))), 7)",
    )

    # IV-flow residuals: submitted IV composites are price-reversal heavy; use
    # volume-flow interaction and neutralize the reversal component.
    add(
        "orth_iv_flow",
        "group_neutralize(ts_decay_linear(vector_neut(multiply(reverse(group_rank(implied_volatility_mean_60, subindustry)), rank(ts_corr(volume, returns, 40))), rank(ts_delta(close, 5))), 7), subindustry)",
    )
    add(
        "orth_iv_flow",
        "group_neutralize(ts_decay_linear(vector_neut(multiply(reverse(group_rank(implied_volatility_mean_skew_30, subindustry)), rank(ts_corr(volume, returns, 40))), rank(ts_zscore(ts_mean(returns, 60), 63))), 10), subindustry)",
    )
    add(
        "orth_iv_flow",
        "ts_decay_linear(vector_neut(multiply(reverse(rank(ts_zscore(implied_volatility_mean_60, 63))), rank(ts_corr(snt_buzz_ret, returns, 40))), rank(ts_delta(close, 5))), 7)",
    )

    # Revision-flow residuals: prior revision alpha was too weak, but gating it
    # by flow and neutralizing price may produce a distinct path.
    add(
        "orth_revision_flow",
        "group_neutralize(ts_decay_linear(vector_neut(multiply(rank(snt1_d1_earningsrevision), rank(ts_corr(volume, returns, 40))), rank(ts_delta(close, 5))), 10), subindustry)",
    )
    add(
        "orth_revision_flow",
        "group_neutralize(ts_decay_linear(vector_neut(multiply(group_rank(snt1_d1_netrecpercent, subindustry), rank(ts_corr(volume, returns, 40))), rank(ts_zscore(mdl53_jc5_3year, 10))), 10), subindustry)",
    )
    add(
        "orth_revision_flow",
        "ts_decay_linear(vector_neut(multiply(rank(ts_delta(snt1_d1_netearningsrevision, 20)), rank(ts_corr(volume, returns, 40))), rank(ts_delta(close, 20))), 10)",
    )

    # Quality-flow residuals: avoid direct fscore_bfl_value + divcov submitted
    # family by interacting with volume-flow and neutralizing that family.
    add(
        "orth_quality_flow",
        "group_neutralize(ts_decay_linear(vector_neut(multiply(rank(fscore_bfl_quality), rank(ts_corr(volume, returns, 40))), rank(add(rank(subtract(ts_decay_linear(fscore_bfl_value, 20), ts_decay_linear(fscore_bfl_value, 120))), rank(mdl77_divcov)))), 10), subindustry)",
    )
    add(
        "orth_quality_flow",
        "group_neutralize(ts_decay_linear(vector_neut(multiply(rank(fscore_bfl_profitability), rank(ts_corr(volume, returns, 40))), rank(ts_delta(close, 5))), 10), subindustry)",
    )
    add(
        "orth_quality_flow",
        "ts_decay_linear(vector_neut(multiply(rank(winsorize(mdl77_divcov, std=4)), rank(ts_corr(volume, returns, 40))), rank(subtract(ts_decay_linear(fscore_bfl_value, 20), ts_decay_linear(fscore_bfl_value, 120)))), 10)",
    )

    # Market-regime residuals: lower beta/correlation paths can be distinct if
    # stripped of the dominant return reversal.
    add(
        "orth_market_regime",
        "group_neutralize(ts_decay_linear(vector_neut(multiply(reverse(group_rank(beta_last_60_days_spy, subindustry)), rank(ts_corr(volume, returns, 40))), rank(ts_delta(close, 5))), 10), subindustry)",
    )
    add(
        "orth_market_regime",
        "group_neutralize(ts_decay_linear(vector_neut(multiply(reverse(group_rank(correlation_last_60_days_spy, subindustry)), rank(ts_corr(volume, returns, 40))), rank(ts_zscore(ts_mean(returns, 60), 63))), 10), subindustry)",
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
    parser.add_argument("--limit", type=int, default=22)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.db, args.limit, args.dry_run)

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
    """Diverse candidate batch after sentiment/reversal self-correlation hits.

    This batch deliberately avoids the two crowded winning shapes:
    - add(sentiment_supply, short_price_reversal)
    - add(jc5_quality, sentiment_supply)

    It explores orthogonal IV/reversal, quality-intangible carry, analyst drift
    repairs, beta/correlation low-risk overlays, and non-additive newsflow legs.
    """
    rows: list[tuple[str, str]] = []

    def add(theme: str, expr: str) -> None:
        rows.append((theme, expr))

    # IV / price residual: close to prior winners, but use residualization,
    # gating, skew, and volume-flow legs rather than pure return reversal.
    add(
        "v4_iv_residual_reversal",
        "group_neutralize(ts_decay_linear(vector_neut(reverse(group_rank(ts_delta(close, 7), subindustry)), rank(ts_zscore(implied_volatility_mean_60, 63))), 7), subindustry)",
    )
    add(
        "v4_iv_residual_reversal",
        "ts_decay_linear(multiply(reverse(rank(ts_zscore(implied_volatility_mean_60, 63))), rank(ts_corr(volume, returns, 20))), 10)",
    )
    add(
        "v4_iv_residual_reversal",
        "trade_when(reverse(rank(ts_zscore(implied_volatility_mean_skew_30, 63))), group_neutralize(ts_decay_linear(reverse(group_rank(ts_delta(close, 5), subindustry)), 5), subindustry), 0)",
    )
    add(
        "v4_iv_residual_reversal",
        "group_neutralize(ts_decay_linear(multiply(reverse(group_rank(implied_volatility_mean_30, subindustry)), rank(ts_corr(snt_buzz_ret, returns, 40))), 7), subindustry)",
    )

    # Quality / intangible carry: the fscore_bfl_value + mdl77_divcov family had
    # medium results with low turnover; push it with interaction and price/flow.
    add(
        "v4_quality_intangible_carry",
        "ts_decay_linear(multiply(rank(subtract(ts_decay_linear(fscore_bfl_value, 20), ts_decay_linear(fscore_bfl_value, 120))), rank(winsorize(mdl77_divcov, std=4))), 10)",
    )
    add(
        "v4_quality_intangible_carry",
        "group_neutralize(ts_decay_linear(multiply(rank(fscore_bfl_quality), rank(subtract(ts_decay_linear(fscore_bfl_value, 15), ts_decay_linear(fscore_bfl_value, 90)))), 10), subindustry)",
    )
    add(
        "v4_quality_intangible_carry",
        "ts_decay_linear(vector_neut(rank(winsorize(mdl77_divcov, std=4)), rank(ts_delta(close, 20))), 10)",
    )
    add(
        "v4_quality_intangible_carry",
        "group_neutralize(ts_decay_linear(multiply(rank(fscore_bfl_profitability), reverse(group_rank(ts_av_diff(close, 20), subindustry))), 7), subindustry)",
    )

    # Analyst/revision drift repairs: #1543 had decent sharpe but low
    # sub-universe sharpe; try stronger neutralization and confirmation.
    add(
        "v4_revision_drift_repair",
        "group_neutralize(ts_decay_linear(multiply(group_rank(snt1_d1_earningsrevision, subindustry), rank(ts_delta(mdl53_jc5_3year, 20))), 10), subindustry)",
    )
    add(
        "v4_revision_drift_repair",
        "ts_decay_linear(vector_neut(rank(ts_delta(snt1_d1_netearningsrevision, 20)), rank(ts_delta(close, 20))), 10)",
    )
    add(
        "v4_revision_drift_repair",
        "group_neutralize(ts_decay_linear(multiply(rank(snt1_d1_earningssurprise), reverse(group_rank(ts_delta(close, 5), subindustry))), 7), subindustry)",
    )
    add(
        "v4_revision_drift_repair",
        "trade_when(rank(snt1_d1_fundamentalfocusrank), ts_decay_linear(group_rank(snt1_d1_netrecpercent, subindustry), 10), 0)",
    )

    # Beta / correlation low-risk overlays: many pure return-reversal shapes are
    # crowded; add market-risk fields to change the PnL path.
    add(
        "v4_beta_corr_lowrisk",
        "ts_decay_linear(multiply(reverse(rank(ts_zscore(beta_last_60_days_spy, 60))), reverse(rank(ts_delta(close, 5)))), 10)",
    )
    add(
        "v4_beta_corr_lowrisk",
        "group_neutralize(ts_decay_linear(add(reverse(group_rank(correlation_last_60_days_spy, subindustry)), reverse(group_rank(beta_last_30_days_spy, subindustry))), 10), subindustry)",
    )
    add(
        "v4_beta_corr_lowrisk",
        "ts_decay_linear(vector_neut(reverse(rank(ts_zscore(ts_sum(returns, 20), 60))), rank(correlation_last_90_days_spy)), 7)",
    )
    add(
        "v4_beta_corr_lowrisk",
        "group_neutralize(ts_decay_linear(multiply(reverse(group_rank(beta_last_90_days_spy, subindustry)), rank(ts_corr(volume, returns, 20))), 7), subindustry)",
    )

    # Newsflow non-additive legs: avoid simply adding sentiment to reversal;
    # use correlation, deltas, compression, and residualization.
    add(
        "v4_newsflow_nonadditive",
        "ts_decay_linear(multiply(rank(ts_delta(scl12_buzz_fast_d1, 5)), signed_power(reverse(rank(ts_delta(close, 5))), 0.5)), 5)",
    )
    add(
        "v4_newsflow_nonadditive",
        "group_neutralize(ts_decay_linear(vector_neut(rank(ts_zscore(scl12_sentiment, 20)), rank(ts_corr(snt_buzz, returns, 40))), 7), subindustry)",
    )
    add(
        "v4_newsflow_nonadditive",
        "ts_target_tvr_decay(multiply(rank(ts_corr(snt_buzz_ret, returns, 60)), reverse(rank(ts_av_diff(close, 20)))), target_tvr=0.18)",
    )
    add(
        "v4_newsflow_nonadditive",
        "group_neutralize(ts_decay_linear(multiply(rank(ts_delta(daily_equity_mood_indicator, 5)), rank(ts_corr(volume, returns, 20))), 5), subindustry)",
    )

    # Options / put-call flavor: prior raw PCR was weak; combine with trend and
    # IV skew to avoid pure PCR levels.
    add(
        "v4_options_flow",
        "ts_decay_linear(multiply(reverse(rank(ts_zscore(pcr_oi_30, 60))), rank(ts_zscore(ts_sum(returns, 60), 120))), 10)",
    )
    add(
        "v4_options_flow",
        "group_neutralize(ts_decay_linear(multiply(reverse(group_rank(pcr_vol_30, subindustry)), reverse(group_rank(implied_volatility_mean_skew_30, subindustry))), 10), subindustry)",
    )
    add(
        "v4_options_flow",
        "ts_decay_linear(vector_neut(reverse(rank(ts_zscore(pcr_oi_10, 40))), rank(ts_delta(close, 5))), 7)",
    )
    add(
        "v4_options_flow",
        "group_neutralize(ts_decay_linear(multiply(rank(call_breakeven_30), reverse(group_rank(ts_delta(close, 5), subindustry))), 5), subindustry)",
    )

    # Price-volume structures that differ from plain ts_delta reversal.
    add(
        "v4_microstructure_alt",
        "ts_decay_linear(reverse(rank(ts_corr(rank(open), rank(volume), 10))), 5)",
    )
    add(
        "v4_microstructure_alt",
        "group_neutralize(ts_decay_linear(multiply(reverse(group_rank(ts_delta(vwap, 5), subindustry)), rank(divide(volume, adv20))), 5), subindustry)",
    )
    add(
        "v4_microstructure_alt",
        "ts_decay_linear(vector_neut(reverse(rank(ts_corr(high, volume, 10))), rank(ts_delta(close, 5))), 5)",
    )
    add(
        "v4_microstructure_alt",
        "group_neutralize(ts_decay_linear(multiply(reverse(group_rank(ts_av_diff(vwap, 20), subindustry)), rank(ts_corr(volume, returns, 20))), 5), subindustry)",
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
    parser.add_argument("--limit", type=int, default=28)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.db, args.limit, args.dry_run)

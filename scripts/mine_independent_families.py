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
    rows: list[tuple[str, str]] = []

    def add(theme: str, expr: str) -> None:
        rows.append((theme, expr))

    # JC5 is strong on Sharpe but weak on fitness by itself. Pair it with
    # earnings information and strip out short-sentiment exposure.
    for window, decay in ((8, 5), (10, 7), (15, 7)):
        jc5 = f"group_rank(ts_zscore(mdl53_jc5_3year, {window}), subindustry)"
        add(
            "jc5_revision",
            f"group_neutralize(ts_decay_linear(vector_neut(multiply(rank({jc5}), rank(snt1_d1_earningsrevision)), rank(group_neutralize(mdl177_5shortsentimentfactor_dmd_supply, subindustry))), {decay}), subindustry)",
        )
        add(
            "jc5_surprise",
            f"group_neutralize(ts_decay_linear(vector_neut(multiply(rank({jc5}), rank(snt1_d1_earningssurprise)), rank(ts_delta(close, 5))), {decay}), subindustry)",
        )

    # Mood changes carry a different source from the submitted supply-level
    # family. Volume interaction and cross-sectional grouping reduce weight
    # concentration seen in the earlier mood candidate.
    for mood_window, corr_window in ((3, 20), (5, 40), (10, 60)):
        raw = (
            f"multiply(group_rank(ts_delta(daily_equity_mood_indicator, {mood_window}), subindustry), "
            f"group_rank(ts_corr(volume, returns, {corr_window}), subindustry))"
        )
        add(
            "mood_flow",
            f"group_neutralize(ts_decay_linear(vector_neut({raw}, rank(ts_delta(close, 5))), 7), subindustry)",
        )
        add(
            "mood_flow",
            f"ts_decay_linear(group_rank(vector_neut({raw}, rank(ts_zscore(ts_sum(returns, 20), 60))), subindustry), 10)",
        )

    # Analyst estimate spread and recommendation changes are independent of
    # the existing quality/value and sentiment-supply submissions.
    spread = (
        "rank(divide(anl4_fs_detail_estimate_1qf_v4_nd_totgw_high, "
        "anl4_fs_detail_estimate_1qf_v4_nd_totgw_low))"
    )
    for rec_field in ("snt1_d1_netrecpercent", "snt1_d1_buyrecpercent"):
        add(
            "analyst_revision",
            f"group_neutralize(ts_decay_linear(vector_neut(multiply({spread}, group_rank({rec_field}, subindustry)), rank(ts_delta(close, 5))), 10), subindustry)",
        )
        add(
            "analyst_revision",
            f"ts_decay_linear(group_rank(multiply({spread}, rank(ts_delta({rec_field}, 20))), subindustry), 7)",
        )
        add(
            "analyst_revision",
            f"group_neutralize(ts_decay_linear(multiply(rank(ts_zscore({spread}, 40)), rank({rec_field})), 10), subindustry)",
        )

    # Profitability/earnings momentum residuals avoid the submitted
    # fscore_bfl_value + mdl77_divcov family.
    for quality in (
        "fscore_bfl_profitability",
        "fscore_bfl_quality",
        "mdl177_earningmomentumfactor_sue",
        "mdl177_earningmomentumfactor_epsrm",
    ):
        add(
            "quality_risk",
            f"group_neutralize(ts_decay_linear(vector_neut(multiply(rank({quality}), reverse(group_rank(systematic_risk_last_30_days, subindustry))), rank(add(rank(fscore_bfl_value), rank(mdl77_divcov)))), 10), subindustry)",
        )
        add(
            "quality_risk",
            f"group_neutralize(ts_decay_linear(multiply(group_rank({quality}, subindustry), reverse(group_rank(correlation_last_90_days_spy, subindustry))), 10), subindustry)",
        )

    # Recommendation and earnings-revision breadth without price reversal.
    add(
        "revision_breadth",
        "group_neutralize(ts_decay_linear(multiply(group_rank(snt1_d1_netearningsrevision, subindustry), group_rank(snt1_d1_fundamentalfocusrank, subindustry)), 10), subindustry)",
    )
    add(
        "revision_breadth",
        "ts_decay_linear(vector_neut(add(rank(snt1_d1_earningsrevision), rank(snt1_d1_netrecpercent)), rank(ts_zscore(mdl53_jc5_3year, 10))), 7)",
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
        skeleton = expression_skeleton(expr)
        if expr in existing_exprs or skeleton in seen_skeletons:
            continue
        selected.append((theme, expr))
        seen_skeletons.add(skeleton)
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
    parser.add_argument("--limit", type=int, default=28)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.db, args.limit, args.dry_run)

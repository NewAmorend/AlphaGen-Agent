from __future__ import annotations


# WQ Brain 提交检查里的关键项。完整顾问权限下，未达标项有时返回
# WARNING 而非 FAIL，因此两者都必须视为未通过。
CRITICAL_CHECKS = {
    "LOW_SHARPE",
    "LOW_FITNESS",
    "LOW_TURNOVER",
    "HIGH_TURNOVER",
    "CONCENTRATED_WEIGHT",
    "LOW_SUB_UNIVERSE_SHARPE",
    "IS_LADDER_SHARPE",
}


def critical_check_failures(checks: list[dict] | None) -> list[str]:
    """Return unique critical checks reported as FAIL or WARNING by WQ."""
    failures: list[str] = []
    for check in checks or []:
        if not isinstance(check, dict):
            continue
        name = str(check.get("name", "")).upper()
        status = str(check.get("result", "")).upper()
        if name in CRITICAL_CHECKS and status in {"FAIL", "WARNING"} and name not in failures:
            failures.append(name)
    return failures

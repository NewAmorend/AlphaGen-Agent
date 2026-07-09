from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class FunctionSpec:
    min_args: int
    max_args: int | None = None


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    issues: tuple[ValidationIssue, ...] = ()


_FUNCTIONS: dict[str, FunctionSpec] = {
    # arithmetic / logical
    "abs": FunctionSpec(1, 1),
    "add": FunctionSpec(2, 3),
    "subtract": FunctionSpec(2, 3),
    "multiply": FunctionSpec(2, 3),
    "divide": FunctionSpec(2, 2),
    "inverse": FunctionSpec(1, 1),
    "reverse": FunctionSpec(1, 1),
    "sign": FunctionSpec(1, 1),
    "sqrt": FunctionSpec(1, 1),
    "log": FunctionSpec(1, 1),
    "power": FunctionSpec(2, 2),
    "signed_power": FunctionSpec(2, 2),
    "min": FunctionSpec(2, None),
    "max": FunctionSpec(2, None),
    "median": FunctionSpec(1, None),
    "densify": FunctionSpec(1, 1),
    "to_nan": FunctionSpec(1, 3),
    "and": FunctionSpec(2, 2),
    "or": FunctionSpec(2, 2),
    "not": FunctionSpec(1, 1),
    "if_else": FunctionSpec(3, 3),
    "is_nan": FunctionSpec(1, 1),
    # time series
    "days_from_last_change": FunctionSpec(1, 1),
    "hump": FunctionSpec(1, 2),
    "jump_decay": FunctionSpec(2, 4),
    "kth_element": FunctionSpec(3, 3),
    "last_diff_value": FunctionSpec(2, 2),
    "ts_arg_max": FunctionSpec(2, 2),
    "ts_arg_min": FunctionSpec(2, 2),
    "ts_argmax": FunctionSpec(2, 2),
    "ts_argmin": FunctionSpec(2, 2),
    "ts_av_diff": FunctionSpec(2, 2),
    "ts_backfill": FunctionSpec(1, 4),
    "ts_corr": FunctionSpec(3, 3),
    "ts_correlation": FunctionSpec(3, 3),
    "ts_covariance": FunctionSpec(3, 3),
    "ts_count_nans": FunctionSpec(2, 2),
    "ts_decay_linear": FunctionSpec(2, 3),
    "decay_linear": FunctionSpec(2, 3),
    "ts_delay": FunctionSpec(2, 2),
    "ts_delta": FunctionSpec(2, 2),
    "ts_max": FunctionSpec(2, 2),
    "ts_mean": FunctionSpec(2, 2),
    "ts_median": FunctionSpec(2, 2),
    "ts_min": FunctionSpec(2, 2),
    "ts_product": FunctionSpec(2, 2),
    "ts_quantile": FunctionSpec(2, 3),
    "ts_rank": FunctionSpec(2, 3),
    "ts_regression": FunctionSpec(3, 5),
    "ts_scale": FunctionSpec(2, 3),
    "ts_skew": FunctionSpec(2, 2),
    "ts_kurtosis": FunctionSpec(2, 2),
    "ts_entropy": FunctionSpec(2, 2),
    "ts_std_dev": FunctionSpec(2, 2),
    "ts_step": FunctionSpec(1, 1),
    "ts_sum": FunctionSpec(2, 2),
    "ts_target_tvr_decay": FunctionSpec(1, 4),
    "ts_zscore": FunctionSpec(2, 2),
    # cross-sectional
    "normalize": FunctionSpec(1, 3),
    "quantile": FunctionSpec(1, 3),
    "rank": FunctionSpec(1, 2),
    "scale": FunctionSpec(1, 3),
    "scale_down": FunctionSpec(1, 2),
    "scale_standardize": FunctionSpec(1, 3),
    "scale_normalize": FunctionSpec(1, 3),
    "winsorize": FunctionSpec(1, 2),
    "zscore": FunctionSpec(1, 1),
    "vector_neut": FunctionSpec(2, 2),
    # vector / group / transforms
    "vec_avg": FunctionSpec(1, 1),
    "vec_max": FunctionSpec(1, 1),
    "vec_min": FunctionSpec(1, 1),
    "vec_sum": FunctionSpec(1, 1),
    "bucket": FunctionSpec(1, 3),
    "combo_a": FunctionSpec(1, 3),
    "generate_stats": FunctionSpec(1, 1),
    "trade_when": FunctionSpec(3, 3),
    "group_backfill": FunctionSpec(3, 4),
    "group_cartesian_product": FunctionSpec(2, 2),
    "group_max": FunctionSpec(2, 2),
    "group_mean": FunctionSpec(3, 3),
    "group_min": FunctionSpec(2, 2),
    "group_neutralize": FunctionSpec(2, 2),
    "group_rank": FunctionSpec(2, 2),
    "group_scale": FunctionSpec(2, 2),
    "group_zscore": FunctionSpec(2, 2),
    # reductions
    "reduce_avg": FunctionSpec(1, 2),
    "reduce_choose": FunctionSpec(2, 3),
    "reduce_count": FunctionSpec(2, 2),
    "reduce_ir": FunctionSpec(1, 1),
    "reduce_kurtosis": FunctionSpec(1, 1),
    "reduce_max": FunctionSpec(1, 1),
    "reduce_min": FunctionSpec(1, 1),
    "reduce_norm": FunctionSpec(1, 1),
    "reduce_percentage": FunctionSpec(1, 2),
    "reduce_powersum": FunctionSpec(1, 3),
    "reduce_range": FunctionSpec(1, 1),
    "reduce_skewness": FunctionSpec(1, 1),
    "reduce_stddev": FunctionSpec(1, 2),
    "reduce_sum": FunctionSpec(1, 1),
    "self_corr": FunctionSpec(1, 1),
}

_LITERAL_IDENTIFIERS = {
    "true", "false", "nan", "inf", "none",
    "gaussian", "uniform", "algo1", "algo2", "dense", "sparse",
    "subindustry", "industry", "sector", "country", "currency", "exchange",
    "universe_size", "nan", "NAN", "ON", "OFF",
}

_KWARG_NAMES = {
    "filter", "driver", "sigma", "std", "range", "buckets", "lookback", "ignore",
    "dense", "sensitivity", "force", "constant", "lambda_min", "lambda_max",
    "target_tvr", "use_std", "usestd", "limit", "rate", "longscale", "shortscale",
    "mode", "nlength", "threshold", "nth", "ignorenan", "ignoreNan", "percentage",
    "precise", "value", "decay", "hump", "lag", "rettype", "k",
}

_IDENT_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")
_CALL_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
_KWARG_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*\s*=")


def validate_fast_expr(
    expr: str,
    *,
    field_ids: set[str] | None = None,
    max_depth: int | None = None,
) -> ValidationResult:
    """Lightweight FastExpr validation before spending a WQ simulation slot.

    This is intentionally conservative about syntax but not a full WorldQuant parser:
    it validates balanced calls, known function names, common arities, and optionally
    that identifiers used as fields are in the fetched data-field pool.
    """
    issues: list[ValidationIssue] = []
    text = expr.strip()
    if not text:
        return ValidationResult(False, (ValidationIssue("empty", "expression is empty"),))
    if ";" in text or "```" in text or "/*" in text or "*/" in text:
        issues.append(ValidationIssue("statement_syntax", "expression contains statement/comment syntax"))

    depth, bad = _paren_depth(text)
    if bad:
        issues.append(ValidationIssue("unbalanced_parens", bad))
    if max_depth is not None and depth > max_depth:
        issues.append(ValidationIssue("nesting_too_deep", f"nesting depth {depth} exceeds {max_depth}"))

    calls = list(_iter_calls(text))
    if not calls:
        issues.append(ValidationIssue("no_function_call", "expression contains no function calls"))

    for name, args_text in calls:
        spec = _FUNCTIONS.get(name)
        if spec is None:
            issues.append(ValidationIssue("unknown_function", f"unknown function: {name}"))
            continue
        argc = len(_split_top_level_args(args_text))
        if argc < spec.min_args:
            issues.append(ValidationIssue("too_few_args", f"{name} expects at least {spec.min_args} args, got {argc}"))
        if spec.max_args is not None and argc > spec.max_args:
            issues.append(ValidationIssue("too_many_args", f"{name} expects at most {spec.max_args} args, got {argc}"))

    if field_ids is not None:
        _validate_identifiers(text, field_ids, issues)

    return ValidationResult(valid=not issues, issues=tuple(issues))


def known_functions() -> set[str]:
    return set(_FUNCTIONS)


def _paren_depth(text: str) -> tuple[int, str | None]:
    depth = 0
    max_depth = 0
    quote: str | None = None
    for ch in text:
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in {'"', "'"}:
            quote = ch
            continue
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return max_depth, "closing parenthesis without matching open"
    if depth != 0:
        return max_depth, "unclosed parenthesis"
    return max_depth, None


def _iter_calls(text: str):
    for match in _CALL_RE.finditer(text):
        name = match.group(1)
        open_idx = match.end() - 1
        close_idx = _matching_paren(text, open_idx)
        if close_idx is None:
            continue
        yield name, text[open_idx + 1:close_idx]


def _matching_paren(text: str, open_idx: int) -> int | None:
    depth = 0
    quote: str | None = None
    for i in range(open_idx, len(text)):
        ch = text[i]
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in {'"', "'"}:
            quote = ch
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def _split_top_level_args(args_text: str) -> list[str]:
    args: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    for i, ch in enumerate(args_text):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in {'"', "'"}:
            quote = ch
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(depth - 1, 0)
        elif ch == "," and depth == 0:
            part = args_text[start:i].strip()
            if part:
                args.append(part)
            start = i + 1
    tail = args_text[start:].strip()
    if tail:
        args.append(tail)
    return args


def _validate_identifiers(text: str, field_ids: set[str], issues: list[ValidationIssue]) -> None:
    lowered_functions = set(_FUNCTIONS)
    normalized_fields = {f.lower() for f in field_ids}
    for match in _IDENT_RE.finditer(text):
        ident = match.group(0)
        ident_lc = ident.lower()
        if ident_lc in lowered_functions or ident in _KWARG_NAMES or ident_lc in _KWARG_NAMES:
            continue
        if ident_lc in {x.lower() for x in _LITERAL_IDENTIFIERS}:
            continue
        if _is_function_name(text, match.end()):
            continue
        if _is_kwarg_name(text, match.end()):
            continue
        if ident_lc not in normalized_fields:
            issues.append(ValidationIssue("unknown_identifier", f"identifier is not in available fields: {ident}"))


def _is_function_name(text: str, end: int) -> bool:
    i = end
    while i < len(text) and text[i].isspace():
        i += 1
    return i < len(text) and text[i] == "("


def _is_kwarg_name(text: str, end: int) -> bool:
    i = end
    while i < len(text) and text[i].isspace():
        i += 1
    return i < len(text) and text[i] == "="

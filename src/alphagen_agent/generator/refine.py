from __future__ import annotations

import random
import re
from typing import Any

from loguru import logger

from ..llm.base import BaseLLMProvider
from .llm import LLMAlphaGenerator, build_proven_wrappers_section


_FAIL_TO_HINT = {
    "LOW_FITNESS": (
        "fitness 差一点点（< 1.0）→ 试**加复合维度**：`add(rank(原表达式), rank(相关信号))`，"
        "或外层加 `winsorize(_, std=4)` 去掉极端值的影响。"
    ),
    "LOW_SHARPE": (
        "sharpe 偏低 → 信号噪声比差。试外层套 `ts_decay_linear(_, 20)` 平滑，"
        "或在中间一层加 `winsorize(_, std=4)`、`group_neutralize(_, subindustry)`。"
    ),
    "LOW_SUB_UNIVERSE_SHARPE": (
        "在 sub-universe（小盘 / 行业子集）上 sharpe 不达标 → 通常需要"
        "**行业中性化**：`group_neutralize(_, subindustry)` 或换成 `group_rank(_, subindustry)`。"
    ),
    "HIGH_TURNOVER": (
        "换手太快（> 70%）→ 外层套 `ts_decay_linear(_, 60)` 或 `hump(_, hump=0.05)` 强力降换手。"
    ),
    "LOW_TURNOVER": (
        "换手低于 1%（信号几乎不动）→ 可能 ts 窗口太长。试缩短到 60 以下，或换 `ts_zscore` 等更动态的算子。"
    ),
    "CONCENTRATED_WEIGHT": (
        "权重过于集中（少数股票主导信号）→ 必须 `rank(_)` 横截面排名拍平分布，"
        "再叠加 `winsorize(_)` 去极值。"
    ),
}


_REFINE_PROMPT = """你是一位顶尖量化研究员。下面这个 alpha 已经有可见的预测信号（{base_quality}），
你的任务是基于它生成 {count} 个变体，目标是把所有 WQ Brain 关键检查推到 PASS（grade=HIGH）。

## 基础表达式

```
{expression}
```

## 历史指标

- fitness = {fitness}
- sharpe = {sharpe}
- turnover = {turnover}
- returns = {returns}
- 失败检查：{failed}  ← **总共 {n_fails} 项失败**

## 针对此失败检查的优化方向

{hints}

## 变体策略（根据失败项数自动选择）

{strategy_section}

## 可以做的变化维度（请在 {count} 个变体里覆盖多种）

1. **调窗口** —— 把数字参数在 5 / 20 / 60 / 120 / 252 之间调整
2. **加中性化** —— 包入 `group_neutralize(_, subindustry)` / `group_rank(_, subindustry)` / `vector_neut(_, ...)`
3. **加平滑/降换手** —— 外层套 `ts_decay_linear(_, 5~20)`、`hump(_, hump=0.01~0.05)`
4. **加预处理** —— 包入 `winsorize(_, std=4)` / `rank(_)` / `zscore(_)` / `ts_backfill(_, lookback=120)`
5. **复合多信号** —— `add(rank(原信号), rank(同向量价信号))` / `multiply(rank(信号A), rank(信号B))`
6. **换同语义字段** —— 用同 dataset 等价字段替换（价格类↔成交类、cashflow_op↔cashflow、ni↔eps）
7. **加非线性** —— 用 `signed_power(_, 0.5)`、`log(1+abs(_))*sign(_)` 压缩极端值（{nonlinear_hint}）
8. **加条件门控** —— `trade_when(ts_std_dev(低波动/流动性字段, 20) < threshold, signal, 0)` 选择性交易降换手提 sharpe

{proven_wrappers_section}

{submitted_skeletons_section}

{exemplars_section}

## 严格规则

- 保留原表达式的**核心信号方向**（动量/反转/价值等大类不能反向）
- **不允许**：与原表达式 verbatim 重复
- **多样性**：{count} 个变体之间不能高度雷同——至少覆盖 4 种以上变化维度
- 每行一个完整表达式，不要解释，不要 `1.` 序号

## 可用字段（已随机抽样 {n_fields_shown} / {n_fields_total} 个，鼓励真换字段）

{fields}

## 可用算子

{operators}

{knowledge_section}

请直接输出 {count} 个变体表达式，每行一个："""


def _build_strategy_section(n_fails: int) -> str:
    """根据失败检查数动态选择 refine 策略——少则微调、多则结构性突变。"""
    if n_fails == 0:
        # base 已经 HIGH。目标不是"修 fail"，是"把 fitness 从 1.x 推到 1.5+"。
        # 关键观察：当 sharpe 和 turnover 都已达标时，fitness 的提升要么靠拉长平滑窗口
        # 进一步压 turnover、要么靠更精细的 neutralization 提 sharpe、要么换更强信号字段。
        return ("**已是 HIGH，无失败项**：目标是把 fitness 推到 1.5+ 拿更好排名。\n"
                "  - 优先**拉长 ts_decay_linear / ts_mean 窗口**（5→10→20→30），常能继续压 turnover 而不损 sharpe\n"
                "  - 加一层 **`group_neutralize(_, subindustry)`** 或换成 **`group_rank(_, subindustry)`**\n"
                "    通常能再提 sharpe 0.1-0.2，对 fitness 是乘性增益\n"
                "  - 尝试换**信号字段**——同语义不同来源（returns ↔ pct_change ↔ ts_delta(close,1)/close）\n"
                "  - 保留外层 wrapper 结构（已被验证有效），变化集中在内层")
    if n_fails == 1:
        return ("**只差 1 项**：以**微调**为主（调窗口/加 winsorize/加 group_neutralize），"
                "保留核心结构，小步迭代。")
    if n_fails == 2:
        return ("**差 2 项**：需要**中等结构调整**。除微调外，至少有 1/3 的变体应该叠加复合信号 "
                "（如 `add(rank(原), rank(同向信号))`）或在原信号外套非线性变换（signed_power）。")
    return ("**差 3 项或更多**：base 距离 HIGH 还有距离，需要**激进重构**。约一半变体应做"
            "结构性调整：复合信号、外层非线性变换、加 trade_when 条件门控、换主算子（rank→zscore、"
            "ts_delta→ts_zscore）。保留信号方向但骨架可以变。")


class RefineAlphaGenerator:
    """对单个 base alpha 生成 K 个微调变体，复用 LLMAlphaGenerator 的 wiki + parse 逻辑。"""

    def __init__(
        self,
        llm: BaseLLMProvider,
        wiki_retriever=None,
        wiki_top_k: int = 3,
        wiki_summary_chars: int = 200,
    ):
        self.llm = llm
        # 复用 LLMAlphaGenerator 的 wiki/parse helpers，避免重复实现
        self._base_gen = LLMAlphaGenerator(
            llm=llm,
            wiki_retriever=wiki_retriever,
            wiki_top_k=wiki_top_k,
            wiki_summary_chars=wiki_summary_chars,
        )

    async def refine(
        self,
        base: dict[str, Any],
        data_fields: list,
        operators: list,
        count: int = 10,
        high_fitness_exemplars: list[dict[str, Any]] | None = None,
        submitted_skeletons: set[str] | None = None,
        extra_exclude_skeletons: set[str] | None = None,
    ) -> list[str]:
        expression = base.get("expression", "")
        if not expression:
            return []

        failed = base.get("failed_checks") or []
        n_fails = len(failed)
        hints = "\n".join(
            f"- {_FAIL_TO_HINT.get(name.upper(), '')}" for name in failed
            if name.upper() in _FAIL_TO_HINT
        ) or "- 无明显失败项，做小幅微调即可"

        # 字段随机抽样 80 个（之前是前 30 个）——base 已用某些字段，让 LLM 真正能换
        sample_n = min(len(data_fields), 80)
        sampled_fields = random.sample(data_fields, sample_n) if len(data_fields) > sample_n else list(data_fields)
        fields_str = "\n".join(
            f"{f.id} ({f.description or 'No description'})" for f in sampled_fields
        )
        operators_str = "\n".join(
            f"{op.name} ({op.description or ''})" for op in operators[: min(len(operators), 40)]
        )

        # 复用 wiki 知识注入（用 base 表达式做 query 检索相关 cookbook）
        knowledge_section = ""
        if self._base_gen.wiki_retriever:
            try:
                hits = await self._base_gen.wiki_retriever.search(
                    f"{expression} {' '.join(failed)} 微调 refine", top_k=3,
                )
                if hits:
                    lines = ["", "## 知识库参考", ""]
                    for h in hits:
                        lines.append(
                            f"- [[{h.page.slug}]] · {h.page.type.value}: "
                            f"{h.page.summary(self._base_gen.wiki_summary_chars)}"
                        )
                    knowledge_section = "\n".join(lines)
            except Exception as exc:
                logger.warning(f"Refine wiki retrieval failed: {exc}")

        def _fmt(v):
            if v is None:
                return "—"
            try:
                return f"{float(v):.3f}"
            except (TypeError, ValueError):
                return str(v)

        grade = (base.get("grade") or "").lower()
        if grade == "high":
            base_quality = (f"HIGH —— 已通过全部 6 项关键检查，fitness={_fmt(base.get('fitness'))} 已是 A 级。"
                            "目标是把 fitness 推到 1.5+ 拿更靠前排名")
            nonlinear_hint = "base 已 HIGH，避免破坏结构，非线性变换大概率把数值搞乱"
        elif grade == "medium":
            base_quality = "MEDIUM —— 只差 1 项关键检查即可提交"
            nonlinear_hint = "若 base 已较精炼，非线性变换通常起反效果，慎用"
        elif grade == "low":
            base_quality = f"LOW —— 信号已可见但有多项检查不达标（fitness={_fmt(base.get('fitness'))} 已超 0.5）"
            nonlinear_hint = "base 是 LOW，鼓励引入非线性来突破当前 fitness 平台"
        else:
            base_quality = f"grade={grade or '?'}"
            nonlinear_hint = "按需引入"

        strategy_section = _build_strategy_section(n_fails)

        # 把 exemplars 段塞进去（复用 LLMAlphaGenerator 的静态方法，行为完全一致）
        exemplars_section = self._base_gen._build_exemplars_section(high_fitness_exemplars or [])
        submitted_skeletons_section = self._base_gen._build_submitted_skeletons_section(submitted_skeletons or set())

        prompt = _REFINE_PROMPT.format(
            count=count,
            base_quality=base_quality,
            n_fails=n_fails,
            nonlinear_hint=nonlinear_hint,
            strategy_section=strategy_section,
            expression=expression,
            fitness=_fmt(base.get("fitness")),
            sharpe=_fmt(base.get("sharpe")),
            turnover=_fmt(base.get("turnover")),
            returns=_fmt(base.get("returns")),
            failed=", ".join(failed) if failed else "—",
            hints=hints,
            fields=fields_str,
            n_fields_shown=sample_n,
            n_fields_total=len(data_fields),
            operators=operators_str,
            knowledge_section=knowledge_section,
            proven_wrappers_section=build_proven_wrappers_section(),
            submitted_skeletons_section=submitted_skeletons_section,
            exemplars_section=exemplars_section,
        )

        logger.info(
            f"Refining alpha #{base.get('alpha_id')} grade={grade} n_fails={n_fails} "
            f"({expression[:50]}...)"
        )
        # 温度从 0.4 提到 0.55——之前变体高度雷同，需要更大多样性
        content = await self.llm.generate(prompt, temperature=0.55)
        raw = self._base_gen._parse_response(content)
        # refine 用 8 层深度上限（vs generate 的 4 层）——HIGH base 常有 5-6 层，
        # cap=6 时 11/15 变体被砍（refine #521 实测）；放到 8 给突破留余地。
        field_ids = {f.id for f in data_fields}
        cleaned = self._base_gen._clean_expressions(raw, max_depth=8, valid_field_ids=field_ids)

        # 去掉跟 base 完全相同的（regex 友好的简单 dedup）
        base_norm = re.sub(r"\s+", "", expression)
        deduped = [e for e in cleaned if re.sub(r"\s+", "", e) != base_norm]
        # 兜底过滤：骨架命中"已存在/已知低分"集合的也丢——WQ 上已存在或已知差，回测纯浪费
        exclude = set(submitted_skeletons or ()) | set(extra_exclude_skeletons or ())
        if exclude:
            from .. import db as _db_mod
            before = len(deduped)
            deduped = [e for e in deduped if _db_mod.expression_skeleton(e) not in exclude]
            if before != len(deduped):
                logger.info(
                    f"Dropped {before - len(deduped)} variants matching existing/known-low skeleton"
                )
        logger.info(
            f"Refine produced {len(deduped)} variants from {len(raw)} raw "
            f"(dropped {len(cleaned) - len(deduped)} duplicates of base)"
        )
        return deduped[:count]

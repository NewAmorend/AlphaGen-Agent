from __future__ import annotations

import re
import random
from typing import Any, TYPE_CHECKING

from loguru import logger

from ..models import WQDataField, WQOperator
from ..llm.base import BaseLLMProvider
from .base import BaseAlphaGenerator

if TYPE_CHECKING:
    from ..wiki.retrieve.hybrid import HybridRetriever

_ALPHA_PROMPT_TEMPLATE = """Generate {count} WorldQuant Brain alpha expressions.

你是一位顶尖的量化因子研究员，专注于创造具有卓越预测能力的Alpha表达式。请基于金融经济学原理和市场微观结构知识，设计{count}个高质量的WorldQuant Brain表达式。

## 核心设计理念

**因子经济学逻辑**：
- 动量效应：价格趋势的持续性
- 均值回归：价格的回复特性
- 波动率特征：风险与收益的关系
- 流动性溢价：成交量蕴含的信息
- 市场情绪：价量关系的异常表现

Grammar Rules
/* */ - Block comments for multiple lines
; - Statement separator (not needed for last line)
Last statement is the Alpha expression for BRAIN simulator
Alpha expression cannot be assigned to a variable
No classes, objects, pointers, or functions allowed

**统计有效性准则**：
- 信息比率最大化：信号与噪声比
- 因子稳定性：在不同市场环境下保持有效
- 单调性：因子值与预期收益的单调关系
- 可解释性：基于扎实的金融逻辑

## 技术实现要求

**智能参数匹配**：
- 动量类：短周期[10,20,30]捕捉近期趋势
- 反转类：中周期[40,50,60]识别过度反应
- 波动类：长周期[120,250]确保统计稳定性
- 流动性类：多周期对比分析异常变化

**噪声处理技术**：
- 移动平均线：平滑噪声，突出趋势
- 标准偏差：测量波动程度
- 归一化：将不同时间窗口的因子值调整到同一尺度

**嵌套结构逻辑**：
- 单层：直接有效的原始信号
- 双层：信号提炼与降噪处理
- 三层：多维度特征融合与增强

**操作符汇总

**算术运算符 (Arithmetic):
abs(x) - 绝对值
add(x, y, filter=false) - 加法 (x + y)
densify(x) - 分组字段稠密化
divide(x, y) - 除法 (x / y)
inverse(x) - 倒数 (1/x)
log(x) - 自然对数
max(x, y, ..) - 最大值
min(x, y, ..) - 最小值
multiply(x, y, filter=false) - 乘法 (x * y)
power(x, y) - 幂运算 (x^y)
reverse(x) - 取反 (-x)
sign(x) - 符号函数
signed_power(x, y) - 保留符号的幂运算
sqrt(x) - 平方根
subtract(x, y, filter=false) - 减法 (x - y)
to_nan(x, value=0, reverse=false) - 值与NaN转换

**逻辑运算符 (Logical):
and(input1, input2) - 逻辑与
if_else(input1, input2, input3) - 条件判断
input1 < input2 - 小于比较
input1 <= input2 - 小于等于
input1 == input2 - 等于比较
input1 > input2 - 大于比较
input1 >= input2 - 大于等于
input1 != input2 - 不等于
is_nan(input) - 是否为NaN
not(x) - 逻辑非
or(input1, input2) - 逻辑或

**时间序列运算符 (Time Series):
days_from_last_change(x) - 上次变化天数
hump(x, hump=0.01) - 限制变化幅度
jump_decay(x, d, sensitivity=0.5, force=0.1) - 跳跃衰减
kth_element(x, d, k) - 第K个值
last_diff_value(x, d) - 最后一个不同值
ts_arg_max(x, d) - 最大值相对索引
ts_arg_min(x, d) - 最小值相对索引
ts_av_diff(x, d) - 与均值的差
ts_backfill(x, lookback=d, k=1, ignore="NAN") - 回填
ts_corr(x, y, d) - 时间序列相关性
ts_count_nans(x, d) - NaN计数
ts_covariance(y, x, d) - 协方差
ts_decay_linear(x, d, dense=false) - 线性衰减
ts_delay(x, d) - 延迟值
ts_delta(x, d) - 差值 (x - 延迟值)
ts_max(x, d) - 时间序列最大值
ts_mean(x, d) - 时间序列均值
ts_min(x, d) - 时间序列最小值
ts_product(x, d) - 时间序列乘积
ts_quantile(x, d, driver="gaussian") - 分位数
ts_rank(x, d, constant=0) - 时间序列排名
ts_regression(y, x, d, lag=0, rettype=0) - 回归分析
ts_scale(x, d, constant=0) - 时间序列缩放
ts_std_dev(x, d) - 时间序列标准差
ts_step(1) - 天数计数器
ts_sum(x, d) - 时间序列求和
ts_target_tvr_decay(x, lambda_min=0, lambda_max=1, target_tvr=0.1) - 目标换手率衰减
ts_zscore(x, d) - 时间序列Z分数

**横截面运算符 (Cross Sectional):
normalize(x, useStd=false, limit=0.0) - 标准化
quantile(x, driver=gaussian, sigma=1.0) - 分位数转换
rank(x, rate=2) - 排名
scale(x, scale=1, longscale=1, shortscale=1) - 缩放
scale_down(x, constant=0) - 按比例缩放
vector_neut(x, y) - 向量中性化
winsorize(x, std=4) - 缩尾处理
zscore(x) - Z分数

**向量运算符 (Vector):
vec_avg(x) - 向量均值
vec_max(x) - 向量最大值
vec_min(x) - 向量最小值
vec_sum(x) - 向量求和

**变换运算符 (Transformational):
bucket(rank(x), range="0,1,0.1" or buckets="2,5,6,7,10") - 分桶
generate_stats(alpha) - 生成统计量
trade_when(x, y, z) - 条件交易

**分组运算符 (Group):
combo_a(alpha, nlength=250, mode='algo1') - 组合Alpha
group_backfill(x, group, d, std=4.0) - 分组回填
group_cartesian_product(g1, g2) - 笛卡尔积分组
group_max(x, group) - 分组最大值
group_mean(x, weight, group) - 分组均值
group_min(x, group) - 分组最小值
group_neutralize(x, group) - 分组中性化
group_rank(x, group) - 分组排名
group_scale(x, group) - 分组缩放
group_zscore(x, group) - 分组Z分数

**特殊运算符 (Special):
in - 包含判断
self_corr(input) - 自相关性
universe_size - 宇宙大小

**归约运算符 (Reduce):
reduce_avg(input, threshold=0) - 平均值归约
reduce_choose(input, nth, ignoreNan=true) - 选择归约
reduce_count(input, threshold) - 计数归约
reduce_ir(input) - IR归约
reduce_kurtosis(input) - 峰度归约
reduce_max(input) - 最大值归约
reduce_min(input) - 最小值归约
reduce_norm(input) - 范数归约
reduce_percentage(input, percentage=0.5) - 百分比归约
reduce_powersum(input, constant=2, precise=false) - 幂和归约
reduce_range(input) - 范围归约
reduce_skewness(input) - 偏度归约
reduce_stddev(input, threshold=0) - 标准差归约
reduce_sum(input) - 求和归约

## 卓越因子特征
每个表达式应体现：
1. **经济学直觉**：基于公认的市场异象
2. **计算鲁棒性**：避免过拟合和极端值敏感
3. **交易可行性**：考虑实际执行成本
4. **创新性**：在经典因子上进行合理改进

**表达式不要太复杂

可用字段：{fields}
可用运算符：{operators}
{forbidden_section}
{proven_wrappers_section}
{exemplars_section}
{knowledge_section}
{previous_results_section}

**重要规则（必须严格遵守）：**
1. **CRITICAL：必须使用提供的「可用字段」列表中的完整字段ID**
2. **CRITICAL：绝对禁止使用 close、volume、high、low、open 等不完整的字段名**
3. **必须使用提供的「可用运算符」列表中的运算符**
4. **每个表达式必须是独立的，不能有赋值或其他语法结构**
5. **请严格按照 operator(field_id, numeric_parameter) 的格式编写**
6. **CRITICAL：每个表达式最多嵌套 4 层括号**。超过 4 层会被自动丢弃。
   合格示例（3 层）：`group_neutralize(rank(ts_delta(fnd6_assets, 60)), subindustry)`
   不合格（5 层）：`quantile(normalize(ts_decay_linear(ts_rank(ts_delta(fnd6_assets, 60)), 20)))`
   **简单 + 经济直觉清晰** 远胜复杂嵌套。深度嵌套是过拟合的信号。
7. **如果违反以上规则，生成的表达式将被系统拒绝！**

**错误示例（禁止使用）：**
- ts_corr(rank(close), rank(volume), 10)
- ts_sum(returns, 20) / ts_std_dev(returns, 20)

**正确示例（使用格式）：**
- ts_corr(rank(fnd72_pit_or_cr_q_oper_inc_to_tot_debt), rank(fnd72_pit_or_cr_q_oper_inc_to_tot_debt), 10)
- divide(ts_sum(fnd72_pit_or_cr_q_oper_inc_to_tot_debt, 20), ts_std_dev(fnd72_pit_or_cr_q_oper_inc_to_tot_debt, 20))

请深入思考每个表达式的金融逻辑基础、预期表现特征以及在实际投资中的适用性。设计完成后，直接输出{count}个表达式，每行一个，规则不允许有其他解释。
Generate {count} expressions:"""


# ============================================================================
# 实测高 Fitness Wrapper 模式（QuantGPT 文档 + WorldQuant 社区经验沉淀）
#
# 这些 wrapper 模式不是 LLM 想出来的，是社区/比赛实测验证过的"信号外套"。
# 项目自己的 exemplars 段是数据驱动的（来自 db），这个 PROVEN_WRAPPERS 是经验
# 驱动的（来自外部沉淀），互补——前者教 LLM "你做过什么 work"，后者教 "整个领
# 域里什么 work"。
#
# 解决我们卡在 0.7 fitness 平台的两个核心问题：
#   - LOW_SHARPE   ← ts_decay_linear 平滑信号 + winsorize 去极值
#   - HIGH_TURNOVER ← trade_when 选择性交易 + ts_decay_linear 让信号惯性
# ============================================================================
PROVEN_WRAPPERS_SECTION = """
## 🎯 实测高 Fitness Wrapper 模式（必看，强烈推荐使用）

WorldQuant 社区 + 比赛实测验证有效的"信号外套"——把你设计的原始信号 `<signal>` 套进
下面任一 wrapper，几乎总能把 Fitness 推到 1.0+。**这不是建议，是数据。**

### 单层 wrapper（最简单，先试这些）
- `ts_decay_linear(rank(<signal>), 5)` — 实测 Fitness ~2.86（VWAP 偏离类）
- `-1 * rank(ts_zscore(<signal>, 63))` — 实测 Fitness ~1.26-1.70（估值/基本面类）
- `rank(ts_rank(<signal>, 40))` — 实测 Fitness ~1.42-1.58（动量类，双重排名）

### 控换手专用（如果 turnover > 50%）
- `trade_when(ts_std(returns,20) < 0.03, rank(<signal>), 0)` — **最强降换手算子**
- `hump(rank(<signal>), hump=0.05)` — 限制日内变化幅度

### 复合 + 过滤（信号好但 sharpe 不够时）
- `-ts_av_diff(<signal>, 50) * ts_corr(<signal>, volume, 50)` — 实测 Fitness ~1.70
- `ts_decay_linear(group_neutralize(rank(<signal>), subindustry), 10)` — 行业中性化后平滑

### 关键观察
1. **`ts_decay_linear` 是最广谱的 sharpe 提升器**——几乎所有信号外套一层都涨
2. **`trade_when` + `ts_std(returns)` 阈值**是 turnover 杀手，能从 60% 降到 20%
3. **`-1 * rank(ts_zscore(_, 63))`** 对所有"水平类"因子（pe、roe、cap 等）都好用
4. 切忌嵌套 5 层以上——wrapper 加一层就行，加两层会把信号磨平
"""


class LLMAlphaGenerator(BaseAlphaGenerator):
    def __init__(
        self,
        llm: BaseLLMProvider,
        wiki_retriever: "HybridRetriever | None" = None,
        wiki_top_k: int = 5,
        wiki_summary_chars: int = 200,
    ):
        self.llm = llm
        self.wiki_retriever = wiki_retriever
        self.wiki_top_k = wiki_top_k
        self.wiki_summary_chars = wiki_summary_chars

    async def generate(
        self,
        data_fields: list[WQDataField],
        operators: list[WQOperator],
        previous_results: list[dict[str, Any]] | None = None,
        count: int = 18,
        forbidden_fields: list[dict[str, Any]] | None = None,
        high_fitness_exemplars: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        fields_str = [f"{f.id} ({f.description or 'No description'})" for f in data_fields]
        operators_by_cat: dict[str, list[str]] = {}
        for op in operators:
            cat = op.category or "Other"
            operators_by_cat.setdefault(cat, []).append(
                f"{op.name} ({op.description or 'No description'})"
            )
        operators_str: list[str] = []
        for cat, ops in operators_by_cat.items():
            sampled = random.sample(ops, min(len(ops), max(1, len(ops) // 2)))
            operators_str.extend(sampled)

        previous_section = self._build_previous_results_section(previous_results)

        knowledge_section = await self._build_knowledge_section(data_fields, operators)

        forbidden_section = self._build_forbidden_section(forbidden_fields)

        exemplars_section = self._build_exemplars_section(high_fitness_exemplars)

        prompt = _ALPHA_PROMPT_TEMPLATE.format(
            count=count,
            fields="\n".join(fields_str),
            operators="\n".join(operators_str),
            forbidden_section=forbidden_section,
            proven_wrappers_section=PROVEN_WRAPPERS_SECTION,
            exemplars_section=exemplars_section,
            knowledge_section=knowledge_section,
            previous_results_section=previous_section,
        )

        content = await self.llm.generate(prompt, temperature=0.3)
        raw_expressions = self._parse_response(content)
        cleaned = self._clean_expressions(raw_expressions)
        logger.info(f"LLM generated {len(cleaned)} valid expressions from {len(raw_expressions)} raw")
        return cleaned

    @staticmethod
    def _build_previous_results_section(previous_results: list[dict[str, Any]] | None) -> str:
        """根据 db 返回的最近回测结果，构造可读 + 可指导的反馈段。

        给 LLM 看的不只是 fitness 数字，而是分级 + 失败 check 名 + 具体改进策略。
        """
        if not previous_results:
            return ""

        lines = ["", "## 历史回测反馈（最近 {} 个）".format(len(previous_results)), ""]
        lines.append("| # | 表达式 | fitness | sharpe | turnover | 评级 | 失败检查 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")

        # 失败模式聚类用
        sharpe_low_count = 0
        fitness_low_count = 0
        turnover_high_count = 0
        sign_flip_candidates: list[str] = []   # fitness 接近 0 且为负 → 加 reverse
        near_miss: list[str] = []              # MEDIUM = 仅一项差 → 微调有戏

        for i, r in enumerate(previous_results, 1):
            expr = r.get("alpha", "")
            perf = r.get("performance") or {}
            fit = perf.get("fitness")
            shp = perf.get("sharpe")
            tov = perf.get("turnover")
            grade = perf.get("grade") or "?"
            failed = r.get("failed_checks") or []
            failed_str = ", ".join(failed[:3]) if failed else "—"

            def _fmt(v):
                if v is None:
                    return "—"
                try:
                    return f"{float(v):.3f}"
                except (TypeError, ValueError):
                    return str(v)

            lines.append(
                f"| {i} | `{expr[:60]}` | {_fmt(fit)} | {_fmt(shp)} | {_fmt(tov)} | "
                f"{grade} | {failed_str} |"
            )

            if isinstance(fit, (int, float)):
                if fit < 0 and abs(fit) >= 0.5:
                    sign_flip_candidates.append(expr)
                if grade == "medium":
                    near_miss.append(expr)
            for name in failed:
                up = name.upper()
                if up == "LOW_SHARPE":
                    sharpe_low_count += 1
                elif up == "LOW_FITNESS":
                    fitness_low_count += 1
                elif up == "HIGH_TURNOVER":
                    turnover_high_count += 1

        lines += ["", "## 基于以上反馈的迭代要求（严格遵守）", ""]
        lines.append(
            "1. **不要重复**上表里出现过的表达式 —— 哪怕窗口数稍微改一下也行，"
            "禁止 verbatim 复制。"
        )
        if sign_flip_candidates:
            lines.append(
                f"2. 上表里有 {len(sign_flip_candidates)} 个 fitness 显著为负的因子 —— "
                "这种**信号方向反了**，对它们外层加 `reverse()` 或 `-1*` 通常能直接翻成正 fitness。"
            )
        if near_miss:
            lines.append(
                f"3. 上表里有 {len(near_miss)} 个 MEDIUM 评级的因子 —— 它们**只差一项硬指标**。"
                "对这些表达式做**微调变体**（换窗口 5↔20↔60↔252、加 `group_neutralize(_, subindustry)`、"
                "外层套 `ts_decay_linear(_, 10)`），通常能从 MEDIUM 推到 HIGH。"
            )
        if fitness_low_count >= 3:
            lines.append(
                f"4. 有 {fitness_low_count} 个 fitness 偏低 —— 单一弱信号撑不起 alpha，"
                "请尝试**复合**：`add(rank(信号A), rank(信号B))` 两个弱相关信号叠加。"
            )
        if sharpe_low_count >= 3:
            lines.append(
                f"5. 有 {sharpe_low_count} 个 sharpe 偏低 —— 信号噪声比差，"
                "尝试 `winsorize(_, std=4)` 或 `ts_decay_linear(_, 20)` 平滑。"
            )
        if turnover_high_count >= 3:
            lines.append(
                f"6. 有 {turnover_high_count} 个 turnover 超 70% —— 换手太快，"
                "外层套 `ts_decay_linear(_, 60)` 或 `hump(_, hump=0.05)` 控换手。"
            )
        lines.append(
            "7. 从未尝试过的因子族（看 wiki 的 [[concepts/quality-factors]]、"
            "[[concepts/liquidity-microstructure]]、[[concepts/analyst-revisions]] 等）也要纳入新一批。"
        )
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _build_forbidden_section(forbidden_fields: list[dict[str, Any]] | None) -> str:
        """已被 orchestrator 从 fields 列表里删掉的字段—— LLM 看不到这条信息就会反复幻觉出来。

        把 top-N 高失败次数的字段显式列在 prompt 里，并附上失败原因和别名建议（如果能推断），
        让 LLM 主动避开而不是事后过滤。对应 QuantGPT llm_service.py 的 "禁止使用的变量" 段。
        """
        if not forbidden_fields:
            return ""
        top = forbidden_fields[:15]
        lines = ["", "## 🚨 禁止使用的字段（历史 WQ 模拟反复报错）", ""]
        for row in top:
            fid = row.get("field_id", "?")
            n = row.get("fail_count", 0)
            reason = row.get("last_reason", "")
            lines.append(f"- `{fid}` (失败 {n} 次, 原因: {reason})")
        lines.append("")
        lines.append("⚠️ 出现在以上列表的字段一律视为非法 ——"
                     " 即使你认为它「应该」存在（如 close、volume、pe_ratio 等简写）。"
                     "必须使用「可用字段」列表里出现的完整 field_id。")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _build_exemplars_section(exemplars: list[dict[str, Any]] | None) -> str:
        """把项目自己 DB 里的高 fitness 历史 alpha 当模板锚点喂给 LLM。

        对应 QuantGPT llm_service.py 的 "高 Fitness 模板" 段，但我们的来源是真实回测过的本地历史，
        不是手写常量，所以会随着 db 累积自动迭代。
        """
        if not exemplars:
            return ""
        top = exemplars[:5]
        lines = ["", "## 📈 本项目历史高 fitness 模板（参考结构，不要原样照抄）", ""]
        lines.append("| fitness | sharpe | turnover | 表达式 |")
        lines.append("| --- | --- | --- | --- |")
        for row in top:
            fit = row.get("fitness")
            shp = row.get("sharpe")
            tov = row.get("turnover")
            expr = (row.get("expression") or "").replace("|", "\\|")[:120]

            def _fmt(v):
                if v is None:
                    return "—"
                try:
                    return f"{float(v):.2f}"
                except (TypeError, ValueError):
                    return str(v)
            lines.append(f"| {_fmt(fit)} | {_fmt(shp)} | {_fmt(tov)} | `{expr}` |")
        lines.append("")
        lines.append("以上是本账号过去真实回测到 fitness ≥ 阈值的因子。"
                     "请参考其**结构组合方式**（算子层级、字段+算子搭配、外层标准化），"
                     "但**必须换用不同字段或不同窗口**避免重复。")
        lines.append("")
        return "\n".join(lines)

    async def _build_knowledge_section(
        self,
        data_fields: list[WQDataField],
        operators: list[WQOperator],
    ) -> str:
        if not self.wiki_retriever:
            return ""
        sampled_fields = random.sample(data_fields, min(len(data_fields), 5))
        sampled_ops = random.sample(operators, min(len(operators), 5))
        query_parts = [
            "高 fitness alpha 设计",
            "动量 反转 波动率 流动性 质量",
            *[f.id for f in sampled_fields],
            *[op.name for op in sampled_ops],
        ]
        query = " ".join(query_parts)
        try:
            hits = await self.wiki_retriever.search(query, top_k=self.wiki_top_k)
        except Exception as exc:
            logger.warning(f"Wiki retrieval failed: {exc}")
            return ""
        if not hits:
            logger.info("Wiki retrieval: 0 hits, prompt unchanged")
            return ""
        hit_summary = " | ".join(
            f"{h.page.title[:30]}({h.score:.2f},{'+'.join(h.sources)})" for h in hits
        )
        logger.info(f"Wiki injected {len(hits)} pages: {hit_summary}")
        lines = ["", "## 知识库参考", "以下是从内部 Quant Wiki 检索到的相关知识，请充分吸收并应用：", ""]
        for h in hits:
            summary = h.page.summary(self.wiki_summary_chars)
            tags = ", ".join(h.page.tags[:6])
            lines.append(f"- [[{h.page.slug}]] · {h.page.type.value} · 标签: {tags}")
            lines.append(f"  - {summary}")
        lines.append("")
        return "\n".join(lines)

    def _parse_response(self, content: str) -> list[str]:
        if "antml:thinking>" in content or "</thinking>" in content:
            parts = content.split("</thinking>")
            if len(parts) > 1:
                content = parts[-1].strip()
            parts2 = content.split("antml:thinking>")
            if len(parts2) > 1:
                content = parts2[-1].strip()

        expressions: list[str] = []
        skip_prefixes = (
            "Hmm,", "Okay,", "I need", "Let me", "First,", "Then,", "Finally,",
            "The expression", "Here is", "This is", "Let me think", "I think",
            "Based on", "Using the", "Combining", "Creating", "Generating",
            "python", "def ", "import ", "Expression:", "Answer:", "Result:", "Output:",
            "#", "*", "Comment:",
        )

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            if any(line.startswith(p) for p in skip_prefixes):
                continue
            line = line.replace("`", "")
            if ". " in line and line[0].isdigit():
                line = line.split(". ", 1)[1]
            if "(" in line and ")" in line:
                expressions.append(line)

        return expressions

    MAX_NESTING_DEPTH = 4   # 超过 4 层嵌套的表达式直接丢弃，太复杂调起来不可能

    @staticmethod
    def _nesting_depth(expr: str) -> int:
        depth = 0
        max_depth = 0
        for ch in expr:
            if ch == "(":
                depth += 1
                max_depth = max(max_depth, depth)
            elif ch == ")":
                depth = max(depth - 1, 0)
        return max_depth

    def _clean_expressions(self, ideas: list[str], max_depth: int | None = None) -> list[str]:
        # max_depth=None 时用类常量 MAX_NESTING_DEPTH（默认 4）；refine 可以传 6 给已经
        # 复杂的 base 留出叠 wrapper 的余地。
        depth_cap = max_depth if max_depth is not None else self.MAX_NESTING_DEPTH
        cleaned: list[str] = []
        rejected_complexity = 0
        valid_funcs = {
            "ts_mean", "divide", "subtract", "add", "multiply", "zscore",
            "ts_rank", "ts_std_dev", "rank", "log", "sqrt", "ts_sum", "ts_min", "ts_max",
            "ts_median", "ts_correlation", "ts_corr", "ts_covariance", "ts_regression",
            "ts_decay_linear", "ts_argmax", "ts_argmin", "ts_product", "ts_skew",
            "ts_kurtosis", "ts_entropy", "scale", "scale_standardize", "scale_normalize",
            "rank_corr", "rank_kendall", "rank_percentile", "correlation", "covariance",
            "regression", "min", "max", "median", "abs", "inverse", "sign", "power",
            "signed_power", "reverse", "ts_delta", "ts_delay", "ts_zscore", "ts_scale",
            "ts_quantile", "ts_av_diff", "ts_backfill", "ts_count_nans", "ts_arg_max",
            "ts_arg_min", "ts_step", "normalize", "quantile", "scale_down", "vector_neut",
            "winsorize", "group_neutralize", "group_rank", "group_zscore", "group_mean",
            "group_min", "group_max", "group_scale", "group_backfill",
            # 之前漏掉的 WQ 算子（prompt 里推荐 LLM 用，但 validator 不认会误丢）：
            # 逻辑 / 条件
            "and", "or", "not", "is_nan", "if_else", "to_nan", "trade_when",
            # 时序 / 控换手
            "hump", "jump_decay", "days_from_last_change", "kth_element",
            "last_diff_value", "ts_target_tvr_decay",
            # 分组 / 向量 / 归约
            "densify", "bucket", "combo_a", "generate_stats", "self_corr",
            "group_cartesian_product",
            "vec_avg", "vec_max", "vec_min", "vec_sum",
            "reduce_avg", "reduce_choose", "reduce_count", "reduce_ir", "reduce_kurtosis",
            "reduce_max", "reduce_min", "reduce_norm", "reduce_percentage",
            "reduce_powersum", "reduce_range", "reduce_skewness", "reduce_stddev",
            "reduce_sum",
        }
        # 用整词匹配——之前用 substring 匹配 "is" 会误杀 is_nan 算子，"the"/"are" 也会
        # 命中很多合法字段名（如 fnd*_assets 含 "are"... 实际是反过来其它字段含子串）。
        # 整词匹配只在真正出现 "the/is/are/it" 等英文散字时才丢。
        common_words = {"it", "the", "is", "are", "captures", "provides", "measures", "represents", "indicates"}
        common_word_re = re.compile(r"\b(" + "|".join(common_words) + r")\b", re.IGNORECASE)

        # 按拒绝原因分桶，跑完后 debug 一行看到底是哪条规则砍掉了大头
        reject_counts: dict[str, int] = {
            "numeric_or_word_only": 0,
            "contains_english_word": 0,
            "no_func_no_arith": 0,
            "missing_parens": 0,
            "nesting_too_deep": 0,
        }

        for idea in ideas:
            fixed = self._fix_syntax(idea)
            if re.match(r"^\d+\.?$|^[a-zA-Z]+$", fixed):
                reject_counts["numeric_or_word_only"] += 1
                continue
            found = common_word_re.findall(fixed)
            if found:
                reject_counts["contains_english_word"] += 1
                logger.debug(f"Rejected (english word {found}): {fixed[:80]}")
                continue
            has_func = any(f in fixed for f in valid_funcs)
            has_arith = any(op in fixed for op in ["+", "-", "*", "/"])
            if not has_func and not has_arith:
                reject_counts["no_func_no_arith"] += 1
                continue
            if "(" not in fixed or ")" not in fixed:
                reject_counts["missing_parens"] += 1
                continue
            if self._nesting_depth(fixed) > depth_cap:
                reject_counts["nesting_too_deep"] += 1
                rejected_complexity += 1
                continue
            cleaned.append(fixed)

        total_rejected = sum(reject_counts.values())
        if total_rejected:
            breakdown = ", ".join(f"{k}={v}" for k, v in reject_counts.items() if v)
            logger.info(
                f"Validation: kept {len(cleaned)}/{len(ideas)}, rejected {total_rejected} "
                f"({breakdown}; depth_cap={depth_cap})"
            )
        return cleaned

    @staticmethod
    def _fix_syntax(expr: str) -> str:
        expr = re.sub(r"\[(\d+)\]", r"\1", expr)
        expr = re.sub(r"\[(\d+),\s*\d+\]", r"\1", expr)
        expr = re.sub(r"\[([^\]]+)\]", r"\1", expr)
        expr = re.sub(r",\s*\)", ")", expr)
        for kw in ("lookback_days", "lambda_max", "window", "period"):
            expr = re.sub(rf"{kw}=(\d+)", r"\1", expr)
        expr = re.sub(r"(\w+)\s+(\w+)\s+(\d+)\)", r"\1, \2, \3)", expr)
        return expr.strip()

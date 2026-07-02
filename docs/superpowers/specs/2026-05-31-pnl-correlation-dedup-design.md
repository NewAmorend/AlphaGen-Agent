# PnL 收益相关性防重复 — 设计文档

日期：2026-05-31
状态：已批准，待写实现计划

## 背景与动机

项目当前的去重是**结构骨架级**的（`expression_skeleton`：字段→FIELD、数字→N）。
但 WorldQuant Brain 真正拒收因子靠的是**收益相关性**（self_correlation）：两个结构
不同的因子，收益曲线也可能高度相关。

实例（2026-05-30 实跑）：refine 产出的 `#617`（HIGH, fitness 1.16）与已提交的
`#606` 结构不同（`group_neutralize(...)` vs `rank(...)`），但收益相关 **0.93**，
提交时被 WQ 拒：「Self-correlation 0.9294 is above cutoff of 0.7 and Sharpe not
better by 10.0% or more.」骨架去重抓不到这种，白白浪费了 30 分钟 refine + 提交尝试。

本功能在**本地**用 PnL 收益向量算相关性，**提交前 / refine 前**就识别这类重复。

## 范围

- **Level 1 — 提交前 gate**：HIGH 候选若与已提交因子收益相关超阈值（且未明显更优）
  → 判重，踢出可提交列表。
- **Level 2 — refine 阶段过滤**：选 refine 候选时跳过"与已提交因子同收益血脉"的，
  避免浪费 refine。
- **不含** Level 3（生成阶段 steering）——超范围。

## 关键设计决策

| 决策点 | 选择 |
| --- | --- |
| 方法 | **纯本地 PnL 相关性**（不依赖 WQ 提交时才算的 self_corr；未提交候选也能算）|
| 参考集 | **已提交（硬 gate）+ HIGH（软提示）** 分级 |
| PnL 拉取 | **懒加载 + 缓存**，只对参与比较的 alpha 拉，自动跳过 REJECT 垃圾 |
| 判重规则 | **照搬 WQ**：`corr > 0.7` 且 `候选 sharpe 没比撞的高 10%` |

## 核心架构：复用现有 "SELF_CORRELATION FAIL" 约定

项目已有一套约定，三处都认 `backtest_results.checks` 里的 `SELF_CORRELATION=FAIL`：
- `list_submittable_alphas` 排除它们
- `get_blacklisted_skeletons` 把它们的骨架拉黑
- `batch_produce._is_redundant` 在 refine 时跳过它们

**本功能不另造过滤，而是"自动算出判决并写进这个约定"**——下游三处全部免费复用。
（2026-05-30 手动给 #617 写 SELF_CORRELATION FAIL 后这三件事就自动生效，验证了该思路。）

## 组件

### 1. WQ client：`get_pnl(wq_alpha_id) -> PnlSeries`
- 调 `/alphas/{id}/recordsets/pnl`，取累计 PnL 时间序列。
- **diff 成每日收益向量**（相关性用日收益，不是累计 PnL）。
- 返回 `{dates: list[str], returns: list[float]}` 或等价结构。
- ⚠️ **实现第一步需验证**：该端点确切返回结构（字段名/格式）尚未实跑确认。
  先拿一个真 `wq_alpha_id` 打一次确认，再定 parser。结构不同只调 parser，不影响设计。

### 2. db：PnL 缓存表 + 存取
```sql
CREATE TABLE IF NOT EXISTS alpha_pnl (
    alpha_id INTEGER PRIMARY KEY REFERENCES alphas(id),
    wq_alpha_id TEXT,
    returns TEXT NOT NULL,        -- JSON: 每日收益向量
    start_date TEXT,
    end_date TEXT,
    fetched_at TIMESTAMP NOT NULL
);
```
- 幂等建表（同 wiki_pages 风格）。
- `get_cached_pnl(alpha_id)` / `upsert_pnl(...)`。

### 3. `CorrelationScreener`（新模块，`engine/correlation.py`）
- `pearson(a, b) -> float`
- `align(series_a, series_b) -> (vec_a, vec_b)`：按日期取重叠段，只用两边都非 NaN 的日。
- `max_correlation(candidate, reference_set) -> (max_corr, ref_alpha_id)`
- `ensure_pnl(alpha_id)`：缓存命中直接返回；否则用 `wq_alpha_id` 拉取 → 存缓存 → 返回。
- **硬 gate**：取候选对已提交集的**最大相关** `(max_corr, ref)`；若
  `max_corr > THRESHOLD` 且 `候选.sharpe < (1+MARGIN) * ref.sharpe`（ref = 最大相关
  命中的那个已提交因子）→ 判重 → `update_backtest_checks` 写入
  `{name: SELF_CORRELATION, result: FAIL, value: max_corr, source: local_pnl}`。
  （候选与参考集均为 HIGH/已提交，sharpe 实际恒为正，无需处理负 sharpe 翻转。）
- **软提示**：`corr(vs 未提交 HIGH) > THRESHOLD` → **只日志/展示标注，不写 FAIL**
  （避免误拉黑还没决定提交的 HIGH）。

### 4. 接入点
- **refine 前**（Level 2）：`batch_produce` 选候选前调 `screen(candidates)`。
- **submittable 列举前**（Level 1）：`screen(high_candidates)`。
- **CLI**：`alphagen-agent screen-corr [--candidates | --high | --all]` 手动跑。

## 数据流

1. 回测产出 alpha + `wq_alpha_id`（已有）。
2. screen 时：对候选 + 参考集成员 `ensure_pnl()`（懒加载）。
3. 算候选 vs 已提交（硬）、vs 未提交 HIGH（软）的 max 相关。
4. 硬判重 → 写 SELF_CORRELATION FAIL（本地）→ 提交 gate / refine 跳过 / 骨架黑名单
   三处自动生效。软 → 仅标注。

## 配置（新 settings）

| 名称 | 默认 | 含义 |
| --- | --- | --- |
| `SELF_CORR_THRESHOLD` | 0.7 | 相关性 cutoff（对齐 WQ）|
| `SELF_CORR_SHARPE_MARGIN` | 0.10 | sharpe 超越豁免线（对齐 WQ）|
| `SELF_CORR_MIN_OVERLAP` | 60 | 两向量重叠不足此天数则跳过（判"未知"）|

## 错误处理（核心原则：fail-open）

- PnL 拉取失败（404 / 无 wq_alpha_id / WQ 抽风 / 5-retry 耗尽）→ 跳过该 alpha 的
  检查，记日志，**绝不崩**。数据缺失 = 未知 = **当不重复放行**。
- 向量重叠 < `MIN_OVERLAP` → 跳过（判未知）。
- 参考集为空 → no-op。
- PnL 含 NaN（缺失日）→ 对齐时只用两边都有值的日。

## 测试（全程假 WQ client，不依赖真网络）

- `pearson`：完全正相关→1、反相关→-1、独立→≈0
- 对齐/重叠：不同日期范围、重叠不足→跳过
- **硬 gate 规则**（最关键）：
  - corr>0.7 且 sharpe 没更好 → 判重
  - corr>0.7 但 sharpe 高 15% → **不判重**（WQ 会收）
  - corr<0.7 → 不判重
- 软提示 vs 未提交 HIGH
- screener 写 SELF_CORRELATION FAIL 进 db（假 PnL 跑 round-trip）
- 懒加载缓存：第二次调用不重拉（假 client 调用计数）

## 不做（YAGNI）

- Level 3 生成 steering
- PnL 向量的 embedding / ANN 索引（参考集小，直接 Pearson 即可）
- 与 WQ 自身 self_corr 端点的混合校验（先纯本地；发现口径偏差再升级）

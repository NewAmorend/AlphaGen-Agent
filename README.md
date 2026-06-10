# wq-agent

[![CI](https://github.com/NewAmorend/wq-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/NewAmorend/wq-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

WorldQuant Alpha 生成与回测 Agent Harness。

通过 LLM（Kimi / DeepSeek）、模板与因子挖掘三种策略批量生成 alpha 表达式，调用 WorldQuant Brain Simulator 进行回测，并按 fitness / Sharpe / turnover / returns 阈值自动评级、入库 SQLite。

> 本项目用于研究自动化与工程实验，不构成投资建议。使用者需要自行确认 WorldQuant Brain、数据源、LLM 服务和研究内容的使用条款与发布权限。

## 功能特性

- **三种生成策略**
  - `llm` — 调用 LLM（默认 Kimi `kimi-k2.6`，可切 DeepSeek）生成符合 FastExpr 语法的表达式
  - `template` — 基于 `templates/alpha_templates.yaml` 的模板组合
  - `factor_mining` — 因子挖掘式遍历
- **WQ Brain 客户端**：自动登录、拉取 datafields/operators、并发提交 simulation、轮询结果
- **回测评估**：按 `MIN_FITNESS / MIN_SHARPE / MAX_TURNOVER / MIN_RETURNS` 划分 HIGH / MEDIUM / LOW / REJECT
- **持久化**：所有 alpha 表达式与回测结果落地到 `wq_agent.db`（SQLite, aiosqlite）
- **Rich 终端界面**：高质量 alpha 表格展示，分级统计

## 安装

需要 Python 3.11+。

```bash
# 创建虚拟环境并安装
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

在新的研究工作区初始化运行所需资源：

```bash
wq-agent init
```

该命令会创建空的 `wiki/` 目录，并复制 `templates/` 和 `.env.example` 到当前目录；已有模板/配置文件默认跳过，如需覆盖可加 `--overwrite`。公开仓不预置 wiki 内容，避免把第三方资料或私人研究记录混入发行包。

## 配置

复制 `.env.example` 为 `.env` 并填入凭证：

```bash
cp .env.example .env
```

关键变量：

| 变量 | 说明 |
| --- | --- |
| `WQ_USERNAME` / `WQ_PASSWORD` | WorldQuant Brain 账号 |
| `LLM_PROVIDER` | `kimi` 或 `deepseek` |
| `KIMI_API_KEY` / `DEEPSEEK_API_KEY` | LLM 服务密钥 |
| `WQ_REGION` / `WQ_UNIVERSE` / `WQ_DELAY` / `WQ_NEUTRALIZATION` | 回测参数 |
| `MIN_FITNESS` / `MIN_SHARPE` / `MAX_TURNOVER` / `MIN_RETURNS` | 评级阈值 |
| `WQ_MAX_CONCURRENT` | Simulation 并发数 |
| `LLM_GEN_TEMPERATURE` | 主生成采样温度（默认 0.5，调高增大多样性减少重复） |
| `DEDUP_FITNESS_FLOOR` | 同骨架历史最佳 fitness 始终低于此值则从生成里排除（默认 0.3，0 关闭） |

## 使用

安装后会注册 `wq-agent` 命令：

```bash
# 全流程：生成 → 回测 → 评估 → 展示
wq-agent run --strategy llm --count 18 --batches 1

# 交互式 TUI 工作台
wq-agent tui

# 仅生成（不回测）
wq-agent generate --strategy template -n 20 --no-backtest

# 用自然语言研究想法驱动 LLM 生成
wq-agent generate --idea "分析师盈利上修叠加低换手约束，做行业中性 alpha" -n 20
wq-agent run --idea-file ideas/analyst_revision.txt -n 20 -b 3

# 对待回测的 alpha 跑回测
wq-agent backtest --pending --concurrent 5
wq-agent backtest --ids 1,2,3

# 查看高质量 alpha
wq-agent list --quality high --min-fitness 0.6

# 统计概览
wq-agent status

# 重复度报告：按 wrapper 家族（outer-2）看库里结构集中度
wq-agent diversity
```

### TUI 工作台

`wq-agent tui` 提供一个键盘优先的终端工作台：左侧设置 strategy/count/batches/idea，右侧查看任务日志和最近 alpha。快捷键：`g` 仅生成、`r` 全流程、`f` refine、`b` 回测 pending、`Ctrl+R` 刷新、`q` 退出。

### 减少 alpha 重复性

生成侧有多道去重，越往后越细：

1. **批内去重** — 同一次生成里同骨架（仅换字段/窗口）的表达式只保留第一个
2. **历史低分骨架排除**（`DEDUP_FITNESS_FLOOR`）— 同骨架历史最佳 fitness 始终低于阈值的结构不再重测
3. **已提交 / self_correlation FAIL 骨架黑名单** — 注入 prompt 并二次过滤
4. **Exemplar 三级多样化** — 防止高 fitness 模板单一栽培（同 wrapper 霸屏）
5. **Wrapper 样例轮换** — 每批从更大的 wrapper 池抽样展示（含 decay+rank 之外的变体），
   不再每次都把 LLM 往同几个外壳上推
6. **结构饱和反馈** — 用 `diversity` 的家族分布，把库里已饱和的 wrapper 家族喂回 prompt
   提示 LLM 这批换别的结构
7. **`LLM_GEN_TEMPERATURE`** — 调高采样温度直接增大结构/字段多样性

用 `wq-agent diversity` 观察 wrapper 家族集中度：`family/alpha ratio` 越低说明结构越单一。

加 `-v / --verbose` 输出 DEBUG 级日志，日志同时写入 `wq_agent.log`。

## 项目结构

```
src/wq_agent/
├── cli.py              # Typer CLI 入口（generate / backtest / list / run / status / wiki ...）
├── config.py           # pydantic-settings 配置加载
├── models.py           # AlphaRecord / BacktestResult / 枚举
├── db.py               # aiosqlite 持久化层（含 wiki 表）
├── agent/
│   └── orchestrator.py # 主流程编排（生成 → 回测 → 评估 → 自学习写 wiki）
├── generator/          # 三种 alpha 生成策略
│   ├── llm.py          # 注入 wiki 检索结果到 prompt
│   ├── template.py
│   └── factor.py
├── llm/                # LLM 适配（Kimi / DeepSeek）
├── wq/                 # WQ Brain 客户端 + 鉴权
├── engine/
│   ├── backtest.py     # Simulation 提交与轮询
│   └── evaluator.py    # 多指标评级
└── wiki/               # Quant Wiki：三通道混合检索 + 私有自动沉淀
    ├── schema.py       # frontmatter / Page 数据类
    ├── store.py        # public/private 扫盘、wikilink 解析、断链检测
    ├── tokenize.py     # FMM + 同义词扩展
    ├── embeddings.py   # Volcengine / zhipu / NoOp
    ├── index.py        # 全量/增量索引器
    ├── auto_record.py  # backtest 后写 entries / lessons
    └── retrieve/
        ├── grep.py     # ripgrep + IDF/Coverage
        ├── vector.py   # sqlite-vec 余弦
        ├── graph.py    # NetworkX + Louvain + PageRank
        └── hybrid.py   # priority + 加权 RRF + 图扩展
templates/
└── alpha_templates.yaml
wiki/                   # 本地知识库工作区；公开仓默认为空，按需导入/沉淀
private_wiki/           # 私有自动沉淀：真实 alpha entries / lessons，默认 gitignore
```

## Quant Wiki（三通道混合检索）

参考 [cnblogs.com/jtuki/p/19861920](https://www.cnblogs.com/jtuki/p/19861920) 的 AI Agent 结构化知识层架构，给 alpha 生成提供领域知识 + 历史教训。公开仓默认不附带 wiki 内容；`wiki/` 是你的本地知识库工作区，可以通过导入命令、手写 markdown 或自动沉淀逐步填充。架构：

- **存储**：`wiki/` 放本地可检索 markdown；`private_wiki/` 放真实 alpha entries / lessons。两者都默认 gitignore，不随公开发行包发布，避免泄露私人研究或第三方资料。
- **词法通道**：FMM 分词（可选 `wiki/dictionary/base.txt` + `synonyms.yaml`）+ 页面正文/slug/tags/frontmatter 元数据 + IDF/Coverage 评分，纯 `0.6 * 加权 IDF 覆盖率 + 0.25 * 原始词条覆盖率`（上限 0.85）
- **向量通道**：sqlite-vec 表 + Volcengine / zhipu Embedding，余弦排名
- **图通道**：NetworkX 构建 wikilink + 共享标签 + 共享来源边，Louvain 社区检测 + PageRank，邻居扩展
- **融合**：所有原始词条命中或明确命中页面身份（slug/path/operator_name/field_id/dataset_id）→ priority；其余走加权 RRF（`k=60, grep:vec=7:3`）+ 图扩展

### 使用

```bash
# 索引（在 wiki 目录有变动后跑一次）
wq-agent wiki index                # 全量
wq-agent wiki index --incremental  # 仅 hash 变化的页重新嵌入

# 调试检索
wq-agent wiki search "动量 反转" -k 5

# 离线评估检索质量（先准备 wiki/bench/retrieval_golden.yml；否则使用内置示例查询）
wq-agent wiki eval --top-k 5

# 本地回归门：低于阈值会以 exit code 2 失败
wq-agent wiki eval --top-k 5 --min-hit-at-k 0.8 --min-mrr 0.6

# 输出 JSON，便于 CI 或脚本记录趋势
wq-agent wiki eval --top-k 5 --json

# 知识库管理员 sub-agent：审查断链、TODO、lesson 归并、bench 覆盖
wq-agent wiki curate
wq-agent wiki curate --since 2026-06-01 --json

# 只应用低风险动作：补 retrieval bench 覆盖 + 写 curation_report.json
wq-agent wiki curate --apply

# 看统计
wq-agent wiki stats
```

`wq-agent generate / run` 启动时会自动构建索引并把 top-K 命中以 `## 知识库参考` section 注入 LLM prompt（目录缺失或 embedding 失败时静默降级）。

### 导入官方文档与论文

WQ Brain 平台的 `/learn/documentation` 是 SPA + 需要登录态，社区论坛走 Cloudflare 也拿不到。所以走两条路：

**1. 用项目的 WQ Brain 鉴权 API 拉官方元数据**（这是 platform UI 上"Operators / Datasets / Datafields"页面背后的数据源）：

```bash
# 拉所有 operator + dataset + field（当前 region/universe/delay），渲染成 wiki/operators/ wiki/datasets/ wiki/fields/
wq-agent wiki import-wq

# 覆盖 region / 限制字段数
wq-agent wiki import-wq --region CHN --universe TOP2000 --limit-per-dataset 100

# 默认只拉 operators + datasets，跳过几千条 field 页；如需字段页再加 --with-fields
wq-agent wiki import-wq
```

importer 写入的页都带 `<!-- managed by wq-agent wiki import-wq -->` 标记，**重跑只覆盖标记还在的页**，你手改过的不动。

**2. 导入研究论文**：

```bash
# arxiv（用 export API，无需登录）
wq-agent wiki import-paper --url https://arxiv.org/abs/2401.12345 --tags momentum

# SSRN（抓 abstract 页 meta）
wq-agent wiki import-paper --url "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=987654"

# 手动模式（WQ 社区 / Cloudflare 后面的论文）
wq-agent wiki import-paper --manual \
    --title "Returns to Buying Winners" \
    --authors "Jegadeesh,Titman" --year 1993 \
    --url https://example.com/paper.pdf \
    --abstract "We document momentum..."
```

### 自学习

`WIKI_AUTO_RECORD=true` 时，每次 `wq-agent run` 的 backtest 完成后，真实研究记录默认写入 `WIKI_AUTO_RECORD_DIR=./private_wiki`：

- HIGH / MEDIUM 结果 → `private_wiki/entries/{date}-alpha-{id}.md`
- REJECT 结果按"失败原因"聚类 → `private_wiki/lessons/{date}-batch-N.md`

下次生成时，`wiki/` 与 `private_wiki/` 会合并检索，所以本地知识和私有 entries / lessons 仍会喂给 LLM；这些目录默认不进公开仓。

### 选向量后端

`EMBEDDING_PROVIDER` 支持 `local` / `volcengine` / `zhipu` / `none`。

**推荐：local**（离线、无 API 费用、无限流）：

```bash
pip install -e ".[local-embed]"   # 安装 fastembed（onnxruntime, ~50MB）

# .env
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5   # 95MB, 512 dim, 中文为主
# 想多语言更强（中英 paper 混合）用 BAAI/bge-m3 (2.2GB, 1024 dim)
EMBEDDING_DIM=0                                # 0 = 自动检测
```

首次 `wq-agent wiki index` 会从 HuggingFace 下载模型并缓存到 `~/.cache/fastembed/`，之后离线运行。

**API 后端**（如需）：

```bash
EMBEDDING_PROVIDER=volcengine
EMBEDDING_MODEL=doubao-embedding-text-240715
EMBEDDING_API_KEY=          # 留空则复用 KIMI_API_KEY
EMBEDDING_BASE_URL=https://ark.cn-beijing.volces.com/api/v3/embeddings
EMBEDDING_DIM=2048
```

`none` 关闭向量通道，只用 grep + 图。

## 开源安全

本仓库按“public code + local/private research”设计：

- 可以公开：`src/`、`tests/`、`templates/`、`.env.example`、项目治理和流程文档。
- 默认不公开：`wiki/` 内容、`private_wiki/`、`.env`、`wq_agent.db`、`*.log`、`.claude/`、`.codex/`。
- 公开前跑：`python scripts/open_source_preflight.py`。该检查会拦截被跟踪的 wiki 内容、私有路径、生成产物和明显的密钥/提交编号痕迹。

## 开源协作流程

本仓库现在按常规开源项目流程维护：

- 贡献入口：[CONTRIBUTING.md](CONTRIBUTING.md)
- 行为准则：[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- 安全披露：[SECURITY.md](SECURITY.md)
- 治理模型：[GOVERNANCE.md](GOVERNANCE.md)
- 支持信息：[SUPPORT.md](SUPPORT.md)
- 变更记录：[CHANGELOG.md](CHANGELOG.md)
- 完整流程：[docs/OPEN_SOURCE_PROCESS.md](docs/OPEN_SOURCE_PROCESS.md)

GitHub 侧包含 issue 模板、PR 模板、CI、tag release 构建和 Dependabot 配置。普通贡献流程是：开 issue 或关联现有 issue -> 从 `main` 建分支 -> 补测试/文档 -> 跑本地检查 -> 开 PR -> CI 通过后由 maintainer review/merge。

## 开发

```bash
ruff check src tests
pytest
python -m build
```

## 许可证

[MIT](LICENSE) © wq-agent contributors

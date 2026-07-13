# 首次运行指南

本文面向第一次从源码运行 AlphaGen Agent 的用户，介绍从安装、初始化研究工作区到完成第一次生成的最短路径，以及常见问题的排查方式。

> AlphaGen Agent 会连接 WorldQuant Brain，并可能调用外部 LLM 或 Embedding 服务。使用前请确认你有权访问相关服务，并遵守其使用条款。

## 1. 环境要求

- Python 3.11 或更高版本
- Git
- 可用的 WorldQuant Brain 账号
- 使用 `llm` 生成或 `refine` 时，还需要兼容的 LLM 服务

先确认 Python 版本：

```bash
python3 --version
```

如果系统中的命令是 `python`，后续可以将示例里的 `python3` 替换为 `python`。

## 2. 从源码安装

```bash
git clone https://github.com/NewAmorend/AlphaGen-Agent.git
cd AlphaGen-Agent

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Windows PowerShell 激活虚拟环境时使用：

```powershell
.venv\Scripts\Activate.ps1
```

安装后先检查命令是否可用：

```bash
alphagen-agent --help
```

## 3. 初始化独立研究工作区

建议将代码仓库与本地研究数据分开。以下命令会创建一个新的工作区，并复制模板与配置示例：

```bash
alphagen-agent init ../alphagen-workspace
cd ../alphagen-workspace
cp .env.example .env
```

工作区初始化后通常包含：

```text
alphagen-workspace/
├── .env                  # 本地配置与凭证，不要提交
├── .env.example          # 配置示例
├── templates/            # alpha 模板
└── wiki/                 # 初始为空的本地知识库
```

默认情况下，后续生成的数据库、日志和私有研究记录也会写入当前工作区。因此，请在运行命令前确认终端位于正确的工作区。

## 4. 配置最小凭证

编辑工作区中的 `.env`，至少填写 WorldQuant Brain 凭证：

```env
WQ_USERNAME=your_username
WQ_PASSWORD=your_password
```

不同工作流需要的配置如下：

| 工作流 | WorldQuant Brain | LLM | Embedding |
| --- | --- | --- | --- |
| `template` / `factor_mining` 生成 | 必需 | 不需要 | 可选 |
| `llm` 生成 | 必需 | 必需 | 可选 |
| `refine` | 必需 | 必需 | 可选 |
| 回测 | 必需 | 不需要 | 可选 |
| Wiki 纯词法检索 | 不需要 | 不需要 | 不需要 |

### LLM 配置

使用 OpenAI-compatible 服务时：

```env
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your_api_key
LLM_MODEL=your_model
```

使用 Anthropic Messages API 时：

```env
LLM_PROVIDER=anthropic
LLM_BASE_URL=https://api.anthropic.com
LLM_API_KEY=your_anthropic_key
LLM_MODEL=your_model
```

如果暂时不使用向量检索，可以先关闭 Embedding：

```env
EMBEDDING_PROVIDER=none
```

需要本地向量检索时，再安装额外依赖并切换配置：

```bash
cd /path/to/AlphaGen-Agent
python -m pip install -e ".[local-embed]"
```

```env
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_DIM=0
```

## 5. 第一次运行

第一次建议使用模板策略、小批量且不回测，以便先验证 WorldQuant Brain 登录与字段加载：

```bash
alphagen-agent generate --strategy template --count 3 --no-backtest
```

确认生成成功后，再对待回测记录运行：

```bash
alphagen-agent backtest --pending --concurrent 2
```

查看统计和结果：

```bash
alphagen-agent status
alphagen-agent list --limit 20
```

启动交互式终端工作台：

```bash
alphagen-agent tui
```

使用 LLM 生成前，先完成 LLM 配置，然后运行小批量请求：

```bash
alphagen-agent generate \
  --strategy llm \
  --idea "分析师盈利上修叠加低换手约束" \
  --count 3 \
  --no-backtest
```

## 6. 本地文件与隐私边界

默认工作区会产生以下本地文件：

| 路径 | 内容 | 是否应提交到公开仓库 |
| --- | --- | --- |
| `.env` | 账号、API Key 与本地配置 | 否 |
| `alphagen_agent.db` | alpha 与回测结果 | 否 |
| `alphagen_agent.log` | 运行日志 | 否 |
| `private_wiki/` | 自动沉淀的真实研究记录 | 否 |
| `wiki/` | 导入或手写的本地知识库 | 默认否 |
| `templates/` | 可公开的模板定义 | 视内容授权而定 |

分享日志或提交 issue 前，应删除凭证、账号标识、真实 alpha、提交编号和私人研究内容。准备公开贡献时运行：

```bash
cd /path/to/AlphaGen-Agent
python scripts/open_source_preflight.py
```

## 7. 常见问题

### `alphagen-agent: command not found`

确认已经激活安装项目时创建的虚拟环境：

```bash
source /path/to/AlphaGen-Agent/.venv/bin/activate
python -m pip show alphagen-agent
```

### Python 版本不满足要求

运行 `python3 --version`。如果低于 3.11，请安装较新的 Python，再重新创建 `.venv`。

### WorldQuant Brain 登录失败

检查 `.env` 中的 `WQ_USERNAME` 和 `WQ_PASSWORD`，并确认命令是在包含该 `.env` 的工作区运行。不要把真实凭证粘贴到 issue 或日志中。

### LLM 请求失败

依次检查：

1. `LLM_PROVIDER` 是否与服务协议一致。
2. `LLM_BASE_URL` 是否指向 API root。
3. `LLM_API_KEY` 和 `LLM_MODEL` 是否有效。
4. 本地服务是否使用 `localhost` 或 `127.0.0.1`。
5. 代理不支持 Responses API 时，可将 `LLM_WIRE_API` 设为 `chat_completions`。

使用 `-v` 获取更详细的本地日志：

```bash
alphagen-agent generate --strategy llm --count 1 --no-backtest -v
```

分享错误信息前请先脱敏。

### 缺少本地 Embedding 依赖

如果不需要向量检索，将 `EMBEDDING_PROVIDER` 设为 `none`。如果需要本地向量检索，按上文安装 `local-embed` 可选依赖。

### `wiki/` 是空目录

这是正常行为。公开仓库不会预置第三方资料或私人研究记录。可以通过 `wiki import-wq`、`wiki import-paper` 或手写 Markdown 逐步填充，随后执行：

```bash
alphagen-agent wiki index
alphagen-agent wiki stats
```

## 下一步

- 完整命令与配置说明：[README](../README.md)
- 贡献指南：[CONTRIBUTING](../CONTRIBUTING.md)
- 支持信息：[SUPPORT](../SUPPORT.md)
- 开源数据边界：[OPEN_SOURCE_PROCESS](OPEN_SOURCE_PROCESS.md)

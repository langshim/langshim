# LangShim

> 一个轻量级本地 LLM 网关，用来统一路由多家模型服务，并集中处理鉴权和用量记录。

`LangShim` 是一个基于 Rust 的 API 网关，位于本地客户端和上游大模型服务之间。它对外暴露 OpenAI / Anthropic / Gemini 兼容接口，在需要时做协议转换，并统一处理网关鉴权、用量记录。

![](docs/slogan.jpg)

## ✨ 核心特性

- 🧩 支持 Anthropic Messages、OpenAI Chat Completions、OpenAI Responses、Google Gemini 之间的协议转换
- 🔀 多 Provider 路由，支持 OpenAI、Anthropic、OpenRouter、DeepSeek、Aliyun、Moonshot、Amazon、Google 等
- 🌊 支持 SSE 流式转发与用量统计

![](docs/claude-code-gpt.png)

![](docs/codex-claude.png)

![](docs/gemini-gpt.png)

## 🚦 快速开始

### 安装

通过 Shell 脚本安装预构建二进制（Linux / macOS）：

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/langshim/langshim/releases/download/v0.1.0/langshim-installer.sh | sh
```

通过 PowerShell 脚本安装预构建二进制（Windows）：

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://github.com/langshim/langshim/releases/download/v0.1.0/langshim-installer.ps1 | iex"
```

### 配置

创建 `~/.langshim/models.json`（或 `$LANGSHIM_HOME/models.json`），建议直接从示例复制后修改：

```bash
mkdir -p ~/.langshim
cp models.example.json ~/.langshim/models.json
```

每个模型条目都包含定价信息和 transport 配置。示例：

```json
{
  "transport": {
    "provider": "google",
    "protocol": "gemini",
    "model_id": "gemini-2.5-flash",
    "base_url": "https://generativelanguage.googleapis.com",
    "api_key": "AIza..."
  }
}
```

检查配置

```
$ langshim doctor 
```

### 运行

```bash
langshim serve
```

服务默认监听 `127.0.0.1:3000`，API_KEY 默认为 secret。
默认情况下，USD/CNY 汇率固定使用 `6.9`，不会联网拉取实时汇率。

也支持命令行参数：

```bash
langshim --api-key secret --port 3000 serve
langshim --live-exchange serve
langshim usage --month 2026-03
```

如果同时提供命令行参数和环境变量，以命令行参数为准。

### Harness 配置参考

配合 github.com/farion1231/cc-switch 使用体验更好，不过 cc-switch 可能未及时跟进 codex 的快速迭代，导致部分配置无法生效。

#### Claude Code

经过 2.1.92 (Claude Code) 测试，建议使用 cc-switch 配置

```
$ cat ~/.claude/settings.json
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "secret",
    "ANTHROPIC_BASE_URL": "http://localhost:3000",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "gpt-5.4-mini",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "gpt-5.4",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "gpt-5.4",
    "ANTHROPIC_MODEL": "gpt-5.4",
    "ENABLE_TOOL_SEARCH": false
  }
}
```

#### Codex

经过 codex-cli 0.118.0 测试

```
$ cat ~/.codex/config.toml
model = "claude-sonnet-4.6"
model_provider = "langshim"

[model_providers.langshim]
name = "LangShim"
wire_api = "responses"
requires_openai_auth = true
supports_websockets = false
base_url = "http://localhost:3000/v1"
```

#### Gemini

经过 gemini-cli 0.36.0 测试，建议使用 cc-switch 配置

```
$ cat ~/.gemini/.env
GEMINI_API_KEY=secret
GEMINI_MODEL=gpt-5.4
GOOGLE_GEMINI_BASE_URL=http://localhost:3000
```

### 从源码编译

需要 Rust `1.75+`：

```bash
git clone https://github.com/langshim/langshim.git
cd langshim
cargo build --release
```

## ⚙️ 配置说明

### 环境变量

| 变量 | 必填 | 说明 | 默认值 |
|------|------|------|--------|
| `LANGSHIM_API_KEY` | 否 | 网关入站鉴权 token | `secret` |
| `LANGSHIM_HOME` | 否 | 本地数据目录，默认 `~/.langshim` | `~/.langshim` |
| `LANGSHIM_PORT` | 否 | HTTP 监听端口 | `3000` |

`--live-exchange` 会在启动时和之后每天从 Frankfurter 拉取 USD/CNY 实时汇率。

### 用量文件

- 请求完成后会把消费明细写到 `~/.langshim/usage/YYYY-MM-DD.jsonl`
- 每天一个 JSONL 文件，便于归档、同步和按时间范围查询
- 可以通过 `LANGSHIM_HOME` 改数据根目录
- 模型路由配置从 `~/.langshim/models.json` 加载
- `usage` 会按 `周期 + model` 聚合

### 用量查询

```bash
# 默认查看最近 15 天，按天聚合
langshim usage

# 指定时间范围
langshim usage daily --from 2026-03-01 --to 2026-03-15

# 查看某个月
langshim usage monthly --month 2026-03
```

### `models.json`

模型路由由 `~/.langshim/models.json`（或 `$LANGSHIM_HOME/models.json`）驱动：

- `models` 存放实际的定价和 transport 定义
- 请求里的 `model` 会直接匹配 `models` 里的 key
- `transport.protocol` 决定该模型走哪套请求路径和适配逻辑

Bedrock 示例：

```json
{
  "transport": {
    "provider": "amazon",
    "protocol": "bedrock",
    "model_id": "global.anthropic.claude-sonnet-4-5-v1:0",
    "base_url": "https://bedrock-runtime.us-west-2.amazonaws.com",
    "api_key": "bedrock-api-key-optional"
  }
}
```

对于 `transport.protocol = bedrock`，网关会把 `transport.api_key` 传给 AWS SDK 的 bearer token provider；当 `transport.base_url` 命中标准 Bedrock 域名时，会自动推断 region。

## 🏗️ 工作方式

![](docs/architecture.jpg)

## 🔌 API 接口

### Anthropic 兼容接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/v1/messages` | `POST` | 发送消息并获取回复 |
| `/v1/messages/count_tokens` | `POST` | 统计消息 token 数量 |

### OpenAI 兼容接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | `POST` | Chat Completions 接口 |
| `/v1/responses` | `POST` | OpenAI Responses API |

### Google Gemini 兼容接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/v1beta/models/{model}:generateContent` | `POST` | 生成内容（非流式） |
| `/v1beta/models/{model}:streamGenerateContent` | `POST` | 生成内容（流式） |

## 🧠 支持的传输协议

`transport.protocol` 当前支持：

- `bedrock`
- `gemini`
- `anthropic`
- `openai`
- `openai-responses`

### 支持矩阵

| 客户端端点 | 请求格式 | `bedrock` 后端 | `gemini` 后端 | `anthropic` 后端 | `openai` 后端 | `openai-responses` 后端 |
|----------|----------|------------------|------------------|------------------|---------------|-------------------------|
| `/v1/messages` | Anthropic Messages | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| `/v1/chat/completions` | OpenAI Chat Completions | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| `/v1/responses` | OpenAI Responses | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| `/v1beta/models/{model}:generateContent`<br>`/v1beta/models/{model}:streamGenerateContent` | Google Gemini | ✅ | ✅ | ✅ | ✅ | ✅ |

### 兼容性说明

- `bedrock` 在 `/v1/messages` 上支持原生直通；`/v1/chat/completions` 和 `/v1/responses` 会通过 Anthropic 兼容转换链处理
- `google` 会调用 Gemini REST API：`/v1beta/models/{model}:generateContent`、`/v1beta/models/{model}:streamGenerateContent?alt=sse`、`/v1beta/models/{model}:countTokens`
- `openai-responses` 目前只在 `/v1/responses` 上支持原生直通
- 文档中的“支持”表示当前存在代码路径
- tools、reasoning、图片输入及厂商扩展字段等细节能力，仍可能因上游 provider 不同而存在差异

## 🧪 测试

```bash
cargo test
```

说明：

- 单元测试覆盖了部分协议约束和适配器转换逻辑

## 📌 项目说明

- 这个项目开箱即可使用本地 API Key 鉴权、模型路由、用量记录能力
- 入站鉴权会把本地 `API_KEY` 与 `Authorization: Bearer ...`、`x-api-key`、`x-goog-api-key` 做比对
- 如果你想把它当作一个通用公开网关来直接使用，通常还需要按自己的多租户、权限和部署模型做一层改造

## 🤝 贡献

欢迎提 Issue 和 PR。如果你准备新增 provider 或协议映射，建议同时补充请求 / 响应兼容边界说明，这会让项目更容易被外部开发者接入。

## 📄 许可证

MIT，见 [LICENSE](LICENSE)。

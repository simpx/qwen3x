# q3x

qwen3x 的轻量命令行 agent，用于内部测试、教学和简单日常任务。使用 TypeScript + Bun，
独立构建和运行，通过标准 OpenAI Chat Completions HTTP 接口连接模型。推理引擎保持 C++；
q3x 不链接 engine/runtime，也不启动或下载模型。

## 构建与运行

开发和构建使用 [Bun](https://bun.com/)（CI 固定 1.3.14）。`agent/` 可以单独复制和构建：

```sh
cd agent
bun install --frozen-lockfile
bun run check
bun test
bun run build
./build/q3x --help
```

也可以从仓库根目录运行 `make q3x`，产物是根目录的 `build/q3x`；`make q3x-test`
运行类型检查和测试。首次测试前仍需在 `agent/` 中安装开发依赖。`BUN=/path/to/bun`
可指定构建工具。直接从源码运行使用 `bun run main.ts`。

发布文件只有 `q3x`，包含 Bun 运行时和应用代码，用户无需安装 Bun、Node 或 npm。
支持 Linux、macOS；Windows 使用 WSL。bash 工具需要系统提供 bash，以及任务实际用到的
git、编译器等程序。没有运行时 npm 依赖；TypeScript 和 Bun 类型定义仅用于开发。
Linux/macOS CI 分别构建本机产物并用该产物运行 CLI/PTY 测试。

先在另一个终端启动模型服务，例如在仓库根目录运行 `make serve-9b`，然后进入工作项目：

```sh
cd /path/to/project
/path/to/qwen3x/build/q3x --base-url http://127.0.0.1:8000/v1 --thinking off
```

默认地址为 `http://127.0.0.1:8000/v1`。省略 `--model` 时查询 `/models`；服务器必须只
公布一个模型，否则明确指定 `--model`。`--base-url` 是 API 前缀，通常以 `/v1` 结尾，
不是完整的 `/chat/completions` 地址。`--thinking on|off` 是可选的 Qwen 扩展；省略时
不发送 `chat_template_kwargs`，保持普通服务器兼容性。
`--temperature 0.2` 可固定采样温度；省略时沿用服务器设置。thinking 和温度会影响任务质量，
本地对照结果见 [TESTING.md](TESTING.md)，不能用一次成功代表稳定性。

```sh
export Q3X_BASE_URL=http://127.0.0.1:8000/v1
export Q3X_MODEL=qwen3.5-9b
export Q3X_API_KEY=...
q3x
q3x -p '检查当前项目并运行测试'
printf '解释当前目录的项目结构' | q3x
```

地址和 key 也接受 `OPENAI_BASE_URL`、`OPENAI_API_KEY`；key 最后回退到 `QWEN_API_KEY`。
命令行参数优先，随后是 `Q3X_*`。不自动加载 `.env` 或 `bunfig.toml`。

`-p` 和管道模式完成一项任务后退出。stdout 输出 assistant 文本；stderr 输出思考内容、
bash 输出和状态。成功退出码为 0，失败或达到上限为 1，SIGTERM 为 143。
`--help` 和 `--version` 不创建会话、不读取 AGENTS.md、不连接模型。

## 交互与 bash

裸命令在终端进入交互。行末输入 `\` 继续多行任务，最后一行按 Enter 提交。
支持 bracketed paste 的终端中，多行粘贴显示为 `[paste 1: N lines]`，按 Enter 后才作为
一项完整任务提交；粘贴上限为 1 MiB。可以编辑占位符周围的文字。

| 命令 | 行为 |
| --- | --- |
| `/compact` | 手动压缩有效上下文 |
| `/continue` | 不增加用户消息，继续上次中断、失败或达到轮数上限的任务 |
| `/status` | 显示模型、上下文估算和当前会话文件 |
| `/help` | 显示帮助 |
| `/exit` | 退出 |

Ctrl-C 在任务中取消模型请求或 bash，回到提示符；在提示符退出。
只读取工作目录中的 `AGENTS.md`，在新会话开始时加入 system message。
不递归查找、不向父目录查找其他指令文件。

bash 是唯一工具，在 `tools/bash.ts` 实现。每次调用启动一个新 shell，以会话工作目录
为 cwd；`cd` 和环境变量修改不跨调用保留。工具执行具有当前用户权限，默认直接执行，
没有内置权限审批或沙箱。

命令 stdin 关闭，stdout/stderr 实时显示并合并为工具结果，默认只保留前 32768 bytes。
超出部分继续消费但不显示、不进入模型上下文，结果注明省略字节数。退出码非零也作为工具
结果交给模型处理。默认超时 120 秒，可用 `--command-timeout` 和 `--max-output` 调整。
工具按顺序执行；一次模型响应中的所有调用必须完整接收后才开始执行。

工具是前台命令：通过 POSIX 进程组管理后代，超时或取消先 SIGTERM，随后 SIGKILL；
shell 结束时清理同组后台进程。因此不适合通过工具启动需要永久保留的服务。
主动脱离进程组的程序不在这个清理机制内。终端输出中的控制字符会被过滤，换行和 tab 保留。
如果 q3x 被 SIGKILL 强杀，进程无法运行清理逻辑，bash 后代可能继续运行；恢复会话不会
自动重放结果未知的命令，继续前需要检查这些命令的实际状态。

## 会话 JSONL

```sh
q3x -o session.jsonl
q3x -r session.jsonl
q3x -r session.jsonl -o experiment.jsonl
```

- 无 `-o/-r`：会话只在内存中。
- `-o`：创建新文件，拒绝覆盖已有文件。
- `-r`：恢复并追加同一文件，采用原会话工作目录；指定不同 `--cwd` 时拒绝恢复。
- `-r` 加 `-o`：复制历史到新文件，后续追加到新文件，源文件不变。

恢复保留原 system message 和 AGENTS.md 快照。服务器、key、模型及运行上限仍由本次参数
或环境变量指定，因此可以换服务器继续。API key 不写入会话；会话包含用户输入、代码、
工具结果和模型输出，默认以 0600 创建。

格式是每行一个 JSON 对象，首行为 `session`（`version: 1`、`cwd`），其他记录包括：

| type | 内容 |
| --- | --- |
| `message` | 发给模型的 system/user/assistant/tool 消息 |
| `tool_start` | 调用 ID；在执行前持久化 |
| `usage` | 服务端 token 计数及对应请求的本地估算 |
| `compact` | 新的有效上下文快照，原始历史保留 |
| `response_error` | 错误和未完成的流式文本，不作为完整 assistant 回放 |
| `turn_end` | completed/failed/cancelled |
| `recovered_tail` | 崩溃留下的不完整末行；`raw` 便于阅读，`bytes_base64` 保留原始字节 |

每条记录追加后 fsync。`.lock` 防止并发写同一会话；退出时删除，恢复时可回收已退出 PID
留下的锁。锁面向同一主机上的本地文件，不支持多主机共享会话文件。
过期锁回收由短暂的 `.lock.reclaim` 文件串行化。若在这个步骤被强杀，后续会明确拒绝
回收；确认没有其他恢复进程运行后，手动删除提示中的 `.lock.reclaim` 再试。
完整 JSON 但缺少末尾换行会补换行；只有最后一条未完成 JSON 可以恢复，原文保存在
`recovered_tail` 中，文件中间损坏则拒绝加载。
末行修复通过同目录临时文件、fsync 和原子 rename 完成，避免先截断文件再记录损坏内容。

如果恢复时存在没有结果的工具调用，q3x 为它添加“执行结果未知”的 tool 消息，
不会重新执行旧命令。用户可先检查状态，再用新指令或 `/continue` 继续。

## 上下文与上限

`--context` 默认 32768，应该与服务器实际容量一致；`--max-tokens` 默认 4096，作为
每个请求的输出额度并从上下文预算中预留。`--max-turns` 默认 32，限制一项任务的 agent
轮数（不含压缩摘要请求）；达到后保留会话，可用 `/continue` 继续。HTTP 整个请求默认限时 300 秒。

q3x 不携带 tokenizer。首次按 UTF-8 字节和消息开销估算，收到服务端 usage 后使用其
prompt token 数校准后续增量。`/status` 中的 `~` 表示估算，不能保证跨模型精确计数。

默认在估算超过可用输入预算的 85% 时自动压缩。`--no-auto-compact` 改为手动模式，
达到输入预算后暂停并提示 `/compact`。如果服务器以 context overflow 拒绝估算内的请求，
自动模式会压缩后重试一次。压缩请求也可能因服务器容量、网络或输出截断失败，此时原会话
保持不变；应调整参数后重试。

压缩通过同一个模型完成，把历史拆成有界片段，逐段合并摘要，保留目标、约束、已验证结果
与待办。system message 保留，最后一条尚待处理的 user 消息原文保留。摘要确实缩短上下文
后才写入 `compact` 记录；压缩失败或取消不会替换有效历史。新的请求使用摘要，JSONL 中的
原始记录仍然完整可查。
最近一组较短的完整工具调用和结果也会原文保留，减少压缩后重复执行刚完成命令的机会；
较大的工具结果仍需摘要，不能保证模型永不重复操作。摘要输出额度使用 `--max-tokens`
（不超过 context 的三分之一）。显式设置 Qwen thinking 时，摘要请求关闭 thinking，
后续任务仍沿用原设置；未指定扩展的标准服务器请求保持不变。

## 阅读代码与验证

```text
main.ts                 参数、终端交互、单次任务
  → agent.ts            请求模型 → 执行工具 → 回传结果
      → client.ts       HTTP / SSE / 流式 tool_calls
      → tools/bash.ts   子进程、取消、超时和输出限制
      → session.ts      消息、JSONL、恢复和压缩快照
```

`types.ts` 定义这些文件共享的边界数据，`terminal.ts` 处理终端粘贴输入。没有 agent 框架依赖。

```sh
bun run check
bun test
bun run build
Q3X_TEST_BINARY="$PWD/build/q3x" bun test
```

测试用本地流式 HTTP 服务和真实 bash/PTY，覆盖 UTF-8/SSE 分片、工具参数拼接、断流、
超时、取消与后代清理、JSONL 恢复和分叉、压缩失败保留原文、容量限制，以及命令行和
终端交互。设置 `Q3X_TEST_BINARY` 后 CLI/PTY 用发布产物运行。

测试设计、真实模型任务集、验收结果与已知限制见 [TESTING.md](TESTING.md)。真实任务集在
`eval/` 中，显式运行才会连接模型；普通 `bun test` 不需要模型或 GPU。
macOS 已配置 CI，本次仅在 Linux x86_64 / WSL 验收。

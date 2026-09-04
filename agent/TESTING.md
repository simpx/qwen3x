# q3x 测试与验收

目标是验证三件事：协议和状态处理正确，进程中断后能够理解并恢复现场，真实模型能完成
有明确验收标准的小任务。三者分别计分，不能把模拟服务器通过率当成模型任务成功率。

## 1. 每次修改运行的确定性测试

在 `agent/` 内：

```sh
bun install --frozen-lockfile
bun run check
bun test
bun run build
Q3X_TEST_BINARY="$PWD/build/q3x" bun test
```

| 风险 | 方法与断言 | 测试文件 |
| --- | --- | --- |
| SSE 分片导致乱码或错误执行 | UTF-8、CR/LF、32 种种子分片、交错工具参数；结果必须一致 | `agent.test.ts`, `protocol.test.ts` |
| 半个工具调用就执行 | 截断流、错误 finish reason、无效 JSON、超大事件；完成前不能启动 bash | `agent.test.ts`, `protocol.test.ts` |
| 服务错误被误重试 | 401/429/500、连接拒绝、模型发现异常；普通错误只请求一次 | `protocol.test.ts` |
| 取消后命令仍执行 | 真实子进程组，TERM 不响应的子孙进程，第二个工具取消后第三个不启动 | `reliability.test.ts` |
| 输出阻塞或耗尽内存 | 同时写 16 MB stdout/stderr，实际只保留 4096 bytes，检查省略计数与退出 | `reliability.test.ts` |
| 崩溃重复副作用 | 在副作用前、之后、结果持久化后真实 SIGKILL；恢复不能重放旧命令 | `reliability.test.ts` |
| 会话并发损坏 | 8 个真实进程竞争恢复，重复 5 轮；每轮只有一个写者；回收保护锁中断则拒绝继续 | `reliability.test.ts` |
| 崩溃末行丢字节 | 将含中文和 emoji 的记录截断在每个字节位置，恢复后逐字节核对 | `reliability.test.ts` |
| 压缩破坏历史 | 分段摘要中途失败、取消、为空、变长、输出截断；原有效历史必须完整保留 | `context.test.ts` |
| 长会话重复操作 | 100 项连续任务、反复自动压缩和重新加载；工具效果恰好 100 次 | `context.test.ts` |
| 粘贴意外提交 | 真实 PTY 多行粘贴，Enter 前零请求；标记和 Unicode 的每个切分位置；1 MiB 字节限制 | `reliability.test.ts`, `terminal.test.ts` |
| 交互无法恢复 | 在实际 bash marker 出现后 Ctrl-C，连续 20 轮，每轮能完成下一项任务 | `reliability.test.ts` |
| 分发仍依赖开发环境 | 独立 binary、PATH 无 Bun/Node、只读目录、空格中文路径、不加载 `.env`、缺 bash 报错 | `distribution.test.ts` |
| 退出留下资源 | SIGTERM 退出 143、释放会话锁、清理同组工具进程 | `distribution.test.ts` |

设置 `Q3X_TEST_BINARY` 只将 CLI/PTY 子进程切换到编译产物；模块单元测试仍直接导入源码。
普通测试使用本地模拟 HTTP 服务、真实 bash 和真实 PTY，不使用远程模型。

## 2. 真实模型任务集

先另行启动模型服务。以下命令从 `agent/` 运行；每次 `--out` 必须是新目录，避免覆盖失败证据。

```sh
bun eval/daily.ts --binary ./build/q3x --url http://127.0.0.1:8000/v1 \
  --out ./build/daily-off --runs 3 --thinking off
bun eval/daily.ts --binary ./build/q3x --url http://127.0.0.1:8000/v1 \
  --out ./build/daily-on --runs 3 --thinking on --temperature 0.2
bun eval/continuation.ts --url http://127.0.0.1:8000/v1 --out ./build/continuation
```

`daily.ts` 在隔离任务目录生成小型项目，每项重复三次，共 18 次。通过标准不依赖模型声称
“完成”：读取 JSONL 的最终回答、比较文件哈希、独立执行测试。git review 使用暂存区作为
基线，不创建 commit。可通过 `--only inspect-flow,fix-csv` 缩小复现范围。

| 任务 | 验收 |
| --- | --- |
| 读调用链 | 成功路径 5 个函数顺序、400/404/200 状态码、要求的 JSON 格式、目录内零改动 |
| 读配置优先级 | defaults → JSON → 环境覆盖，精确比对最终配置、零改动 |
| Review 数学函数 | 定位 3 个回归、JSON 数组、零改动；人工核查反例 |
| Review 分页函数 | 定位 2 个回归、JSON 数组、零改动；人工核查反例 |
| 修复重试退避 | 仅改 `retry.py`；agent 自己运行测试，验收程序再独立执行 3 项测试 |
| 修复 CSV 汇总 | 仅改 `totals.py`；小数、带引号逗号、仅表头 3 项测试通过 |

格式判定接受整个回答被 JSON fence 包裹，不接受 JSON 外额外解释。Review 的自动检查只
验证函数名和解释字段，**不能验证反例正确性**。人工应对照 fixture 运行反例，将漏报、
误报、格式失败分别记下；`passed: true` 不能直接当成完整 review 正确。
文件哈希只覆盖任务目录，忽略 `.git` 和 Python cache，不能证明模型没写目录外文件；
还需检查 JSONL 中的命令。本程序不提供文件系统沙箱。

`continuation.ts` 用真实模型先修复 `a.py`、通过测试、向 `actions.log` 追加一次标记；
然后插入明确标注为测试 fixture 的长历史输出，进行三次摘要、关闭/重新加载 Session、
继续任务。每轮检查标识符保留、保护文件字节不变、测试通过、追加不重复。
此测试通过模块 API 重新加载会话，不宣称覆盖进程重启；真正的进程恢复由上一层测试覆盖。

产物包括每项 prompt、stdout/stderr、完整 JSONL、独立测试日志，以及汇总 `results.json`。
汇总记录 binary SHA-256、模型 ID、thinking、温度设置、上下文额度、耗时和 token usage。
usage 为任务请求计数，不包含摘要；估算 token 不是精确 tokenizer 测量。

## 3. 本次发现并修复的问题

- 不完整 UTF-8 末行以前会在解码时丢掉原始字节：现在保留 base64 原字节，原子替换修复文件。
- 多行粘贴以前可能把第一行当作任务提交：现在显示占位符，用户 Enter 后一次性提交；补上
  同一输入块内结束的超长中文粘贴检查。
- thinking 摘要曾耗尽较小输出额度而没有摘要：明确使用文本摘要并给足配置中的输出额度。
- 压缩刚完成的工具交换会诱发重复操作：保留最近较短的工具交换；100 轮测试曾得到
  133 次执行，修复后要求恰好 100 次。
- 模型发现接受空 ID：现在明确拒绝。
- 过期锁回收存在两个恢复进程误删新锁的竞态：串行化回收并重新核对 PID；回收步骤被强杀
  留下保护文件时明确失败，不能为自动恢复而冒险同时写入。

## 4. 本地验收记录

环境：2026-09-04，Linux x86_64 / WSL，Bun 1.3.14，RTX 4080 SUPER，Qwen3.5-9B Q8_0，
32768 context、4096 输出额度，真实任务每项最多 16 轮。本次没有在 macOS 上运行；CI 已配置。

真实任务基线使用旧通用指令、thinking off、服务器默认 temperature=1.0；第二组使用更新
后的通用指令、thinking on、temperature=0.2。三个因素同时变化，**不是单变量实验**，不能
把差异归因于 thinking 或温度，更不能当作跨模型性能结论。

确定性验证：源码 45 pass、1 skip（仅 binary 的分发用例）；指定最终 binary 后 46 pass，
535 个断言。类型检查通过。`agent/` 独立复制到不含 `node_modules` 的目录后，构建与
`--version` 通过。最终 Linux binary 为 94,607,488 bytes（约 90.2 MiB），链接系统 libc
相关库，无 Bun/Node 安装依赖。

| 真实任务 | 基线：thinking off | 对照：thinking on，0.2 |
| --- | --- | --- |
| 调用链 | 0/3 | 2/3 |
| 配置优先级 | 3/3 | 2/3 |
| 数学 review | 1/3 | 0/3 |
| 分页 review | 1/3 | 0/3 |
| retry 修复 | 3/3 | 3/3 |
| CSV 修复 | 3/3 | 3/3 |
| 严格自动验收合计 | 11/18 | 10/18 |
| 任务累计墙钟时间 | 477 秒 | 609 秒 |

这张表包含格式判定。对照组 8 个自动失败均不满足请求的 JSON 格式，不代表 8 个任务内容
全错；但也不能只去掉格式检查就称其可靠。人工检查发现：

- 基线调用链出现不读文件就猜测、漏步骤、最终只回答 `Ready`；其中一次把答案写到
  `/tmp/answer.json`，目录哈希没有捕捉这个目录外副作用，命令日志捕捉到了。
- 基线 `review-math-2` 虽自动通过，却声称 `clamp(4,5,10)` 返回 4、期望 6；独立运行结果
  为 5，正确版本也应为 5。这证明自动 review 分数高估了完整正确性。
- 基线另一次把 `page_count` 的整数参数写成列表，实际会 TypeError，不能作为报告中的
  数值反例。另一次把分页错误泛化成“一律只返回一个元素”，也不成立。
- 对照组数学 review 都定位到了三个回归，但前两次的 `[1,2,3]` 平均值只体现返回类型
  差异，没展示所声称的小数损失；第三次声称 `clamp(5,10,20)` 返回 20，独立结果为 10。
  分页 review 的主要问题与具体反例经人工核查成立，但三次都违反了只返回 JSON 的要求。
- 两组共 12 次修复全部仅修改指定文件，agent 运行测试后，验收程序独立重跑的测试也通过。

因此本次支持“小型、有测试可核对的修复任务可以开始使用”，不支持“9B 已能稳定给出
严格格式或无需核查的代码 review”。开 thinking 和低温采样没有改善总体验收分数。

真实模型连续压缩与恢复 **3/3 通过**。三次压缩前后估算 token 分别为 5818→1065、
4341→1046、4383→1046（继续验证后的上下文会再次增长）。每轮都保留 `task-4827`，
最终回答明确 `b.py` 不在修改范围内，保护文件字节不变，独立测试通过，`actions.log`
始终只有一行 `verified`。这是一条固定约束场景的证据，不代表摘要能无损保留任意历史。

本地产物位于仓库根目录的 `build/`（不提交）：

| 产物 | 用途 |
| --- | --- |
| `q3x-tests-source-final.log` / `q3x-tests-binary-final.log` | 最终测试日志 |
| `q3x-daily-off/results.json` | 基线 18 次任务及逐项证据 |
| `q3x-daily-on/results.json` | 对照 18 次任务及逐项证据 |
| `q3x-daily-manual-examples.json` | 独立运行有疑问反例的实际返回值 |
| `q3x-daily-thinking-before.log` / `q3x-daily-thinking-after.log` | 真实模型摘要截断问题的修复前后记录 |
| `q3x-daily-continuation/` | 真实模型压缩与继续任务记录 |

基线 binary SHA-256 为 `6f49e26dc1ce7c594841a56cc2400e91aefc6f6e00b03ede2dc803e2b7b4a876`；
对照产物 `q3x-candidate` 为 `d2da66de6df40ce924b0808ec464e0ab465988dc1d467f27f6eae4a0a6b71196`。
最终 `build/q3x` 为 `9eaa00335d639239b4ac70951047bd007339797e7fadf0e920dc4c9f593b2b9c`，
比对照产物再增加了粘贴字节上限和过期锁回收保护修复；46 项测试使用最终产物。

## 5. 验收边界与后续门槛

- 可控测试全过只是 harness 正确性的证据。真实模型回答正确、遵守格式、有效反例仍应独立
  评价。小型 fixture 不等价于 SWE-bench，不覆盖大型仓库自主开发。
- SIGKILL 无法执行清理，测试真实观察到了残留 bash 进程；模型重新决定执行同样命令也
  不受“恢复不重放旧调用”保证约束。对有副作用的任务需要核查现场。
- JSONL 测了截断、并发和进程强杀，没有做磁盘断电、磁盘写满、网络文件系统故障注入。
- macOS 发布前需要在真实 Mac 完成同一测试；Linux 产物依赖系统 libc 和任务所需工具，
  “单 binary”表示无需 Bun/Node，不表示静态链接或自带 bash/git/Python。
- 更换模型、修改 system prompt 或采样参数后，重跑真实任务并保留失败样本。若目标变成
  机器消费的严格 JSON，还需另行设计 schema 约束或校验重试；目前不会悄悄重写模型答案。

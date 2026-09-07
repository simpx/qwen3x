# Qwen3.8-27B Q4_0 / Apple Silicon 验收

更新时间：2026-09-07（Asia/Shanghai）。实现对应本文档所在提交，设备为
MacBook Pro（Apple M5 Pro、15 CPU cores、48 GB unified memory），系统为 macOS
26.3.2，Metal compiler 为 32023.883。

## 结论

27B 在本机只保留 Q4_0 + Metal 路径。当前 32K Session 下，128-token prompt 的 warm
prefill 为 `23.61 tok/s`，短上下文持续 decode 为 `13.22 tok/s`；4096-token prompt 的
prefill 降至 `17.69 tok/s`，此时持续 decode 为 `6.89 tok/s`。模型和 Session 的 Metal
allocation 合计约 18.39 GiB，没有超过设备报告的 37.44 GiB recommended working set。

这是一条正确、可运行并通过真实 pi coding fixture 的本地路径。128/512-token 的 prefill
和 decode 均达到 `goal.md` 的最低目标，4096-token case 完整结束。q3x、完整 EvalScope 和
其他模型性能不属于本目标。4096-token TTFT 约 231 秒，是当前实现明确的可用性限制。

## 固定模型与格式

权重来自官方 `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`。只打包 text backbone；vision encoder 和
MTP layer 不进入 model bin。

固定结构为：

```text
V=248320  H=5120  I=17408  N=64
AI=4      AH=24   KVH=4    AD=256  RD=64
KH=16     VH=48   KD=128   VD=128  CK=4
matrix=Q4_0  tied_embeddings=false
```

schema 共 851 个 tensor、26,895,998,464 个参数。当前产物：

```text
build/qwen38-27b-q4_0-model.bin
size:   15,132,820,608 bytes (14.09 GiB)
sha256: 95daa0ced284efaa1f0394fcd162e7b0f385db3177d68858fd497cfe933c6714
```

Q4_0 与 ggml block 布局一致：每 32 个权重共用一个 FP16 scale，随后是 16 bytes packed
nibbles；低 nibble 保存 0..15，高 nibble 保存 16..31，反量化整数范围为 `[-8, 7]`。
model bin 只有一个固定 header 和固定 tensor 顺序，没有 format version、旧格式兼容或运行时
格式探测。

## Metal 实现

Decode 保留一条从 embedding 到 logits 的完整单-token forward。Q4 prefill 固定以四个 token
为一批：

```text
batch embedding
  -> per-layer RMSNorm
  -> batched matrix projections
  -> token-ordered DeltaNet / causal Attention
  -> batched output projection
  -> RMSNorm / FFN
  -> final-token logits
```

矩阵 kernel 让一个 SIMD group 解码一行 Q4 权重，并将同一权重用于四个 token。DeltaNet 的
convolution 和 recurrent state 在 kernel 内仍按 token 顺序更新。Attention 可以先批量写入
KV，但每个 token 只读取到自己的绝对 position，因此 causal 语义不变。

8-token batch 在同一机器上实测比四 token 慢约 3.2%，因此实现固定为四，不增加动态 tuning
或多套兼容路径。

## 正确性

`make metal-test` 在 Apple M5 Pro 真机通过，覆盖 BF16、Q8_0、Q4_0 的完整 forward、
recurrent state、KV cache、不同 batch 尾部、checkpoint restore 和 reset。Q4 合成模型相对
CPU baseline 的最后一步结果为：

```text
logits:      max_abs_error=1.19209e-07
recurrent:   max_abs_error=1.49012e-07
key cache:   max_abs_error=1.43051e-06
value cache: max_abs_error=1.78814e-07
```

真实 27B 使用 15-token prompt 对比 batch prefill 与逐 token forward，完整 248,320 logits
的 `max_abs_error=0`，position 和 argmax 相同。Q4 向量解码改动也与前一 commit 的真实 27B
完整 logits 完全一致。

2026-09-06/07 在当前实现上重新运行以下回归，全部通过：

```sh
make test
make -C tests render-test
make -C tests http-test
MTL_DEBUG_LAYER=1 make metal-test
```

其中 Metal validation layer 明确启用并报告实际设备 `Apple M5 Pro`；没有 Metal error、NaN
或 CPU fallback。官方仓库 API 同时确认固定 revision 的 commit SHA 与 Makefile 中的
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` 一致。

## 内存

只创建模型和一个 32768-token Session 时，Metal 日志为：

```text
Metal ready device=Apple M5 Pro weights=15132820608 allocated=15133786112
  max_buffer=30150672384 recommended_working_set=40200896512
Metal state model=Qwen3.8-27B context=32768 bytes=4612994944
  allocated=19747143680
```

统一 benchmark 前后系统 swap 都是 3,105.38 MiB，变化为零；memory pressure level 始终为
1。进程 peak physical footprint 为 19,665,546,496 bytes。因此这里只能说明测试期间没有
增加 swap 或触发更高压力，不能把系统原有 swap 记作模型自身消耗。

## 性能

命令：

```sh
make metal-library
caffeinate -i python3 scripts/bench_session.py \
  --library build/metal/libqwen3x-metal.dylib \
  --model build/qwen38-27b-q4_0-model.bin \
  --context 32768 --prompts 128 512 4096 --decode 128 --repeats 3 \
  --output build/bench-27b-q4.json
```

一轮 warmup 后三次正式样本中位数：

| prompt / decode | prefill | prefill tok/s | decode | decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| 128 / 128 | 5.422 s | 23.61 | 9.686 s | 13.22 |
| 512 / 128 | 22.975 s | 22.28 | 10.737 s | 11.92 |
| 4096 / 128 | 231.498 s | 17.69 | 18.573 s | 6.89 |

128-token 的三次正式 prefill 为 `23.62/23.61/22.66 tok/s`，decode 为
`13.25/13.22/13.05 tok/s`。

512-token 的三次正式 prefill 为 `22.39/22.28/22.15 tok/s`，decode 为
`12.08/11.92/11.76 tok/s`。4096-token 的三次正式 prefill 为
`17.69/17.70/17.68 tok/s`，decode 为 `6.89/6.89/6.89 tok/s`。模型 load 为 5.94 秒，
Session 创建为 0.011 秒；二者不计入 prefill 或 decode。完整逐轮数据、源码 SHA-256、
内存和系统状态保存在忽略提交的 `build/bench-27b-q4.json`。

## HTTP

使用 `make serve-27b` 启动真实 Q4_0、Metal、32K、单 Session 服务。复验时模型在 4.15 秒内加载，
`/v1/models` 返回唯一模型 `qwen3.8-27b`。

非流式请求关闭 thinking、temperature 设为 0，要求只回答“你好”。服务使用 16-token prompt，
返回一个 completion token“你好”并以 `stop` 结束；prefill 为 1.507 秒，端到端为 1.580 秒。

SSE 请求要求只回答“流式正常”，依次返回 `流`、`式`、`正常` 三个 content chunk，随后返回
带 `finish_reason=stop` 的结束 chunk 和 `[DONE]`。18-token prompt 的 prefill 为 0.997 秒，
端到端为 1.269 秒。错误模型名返回 HTTP 404、`model_not_found`，没有进入 generation。

同一个 16-token 请求连续执行两次时复用同一个 Session ID；第二次返回
`cached_tokens=16`，prefill 从 933 ms 降到 2.8 ms，证明 prefix checkpoint 命中。

每个成功请求都只产生一条 `access started` 和一条 `access completed` 日志；普通日志只记录
token 数、阶段耗时、状态和 request/session ID，没有记录 prompt 或生成文本。完整内容只写入
显式启用的 audit log。

## Pi coding fixture

使用 pi 0.84.4、Node 22.19.0 和 `thinking=low` 驱动真实 27B 服务，在隔离的临时目录中运行
`eval/agent/fixtures/bugfix/repo`。原始 fixture 的三个测试有一个失败：`retry_delays(4, 0.5)`
返回 `[1.0, 2.0, 4.0, 8.0]`，预期为 `[0.5, 1.0, 2.0, 4.0]`。

pi 自主读取代码和测试，将 `retry.py` 中的 `range(1, attempts + 1)` 改为
`range(attempts)`，并运行 `python3 -m unittest -v`。最终只修改了要求范围内的
`retry.py`，pi 内部测试和独立复跑均为 3/3 通过，最终总结准确。

完整过程包含 8 轮流式 generation，约 3 分 54 秒，没有 timeout、terminated 或截断。首轮
1591-token prompt 的 prefill 为 73.542 秒；后续 7 轮都命中 live prefix cache，每轮只需
prefill 46–265 个新 token，对应 2.576–13.612 秒。各轮 decode 为 7.61–9.83 tok/s。

工具轨迹也暴露了一个不影响结果的小问题：pi 把 `grep -c` 输出的 `0` 误读成存在一个回车符，
因此用 `write` 原样重写了一次文件，再次确认测试通过。最终 diff 仍只有上述一行语义改动。
完整 session 轨迹保存在忽略提交的
`build/pi-27b-session/2026-09-06T12-06-02-373Z_01a0769c-7d85-76a9-a629-2b662a0a2a31.jsonl`。

## 使用

```sh
make model-27b
make serve-27b
```

`make model-27b` 生成上述 Q4_0 model bin；`make serve-27b` 固定使用 Metal、单 Session、
32768 context。运行时需要 macOS 的 Metal runtime；只有重新编译 shader 和 executable 时
才需要 Xcode Metal toolchain。

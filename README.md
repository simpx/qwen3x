# qwen3x

一个用于教学和概念验证（PoC）的 Qwen3.5 / Qwen3.8 推理项目。

这个仓库有两条互相配合的路径：

- `from-scratch/` 从可手算的 toy example 开始，逐步解释 Qwen hybrid 架构；
- 仓库根目录是正式实现，提供 C++ Engine/Session、Python 绑定和
  OpenAI-compatible HTTP server。

项目优先级是：

```text
correct -> simple -> readable -> usable -> fast
```

它目前不是通用 Tensor 框架，也不以替代 llama.cpp、vLLM 或生产级 serving
系统为目标。代码首先服务于架构学习、数值验证和真实模型 PoC。

## 数据流

```text
token ids
  -> embedding
  -> N * {
       RMSNorm -> DeltaNet / attention -> residual
       RMSNorm -> SwiGLU FFN          -> residual
     }
  -> final RMSNorm
  -> lm_head
  -> logits
```

Python 负责 tokenizer、chat template、HTTP 和 streaming；C++ 负责模型权重、
Session state 和逐 token forward：

```text
OpenAI client
  -> Python server
  -> C++ SessionManager
  -> C++ Session
  -> C++ Engine + mmap weights
```

## 目录结构

```text
from-scratch/       从 toy math 到真实 Qwen3.5-0.8B forward 的教学课程
reference/          官方 PyTorch/Transformers oracle 与数值对比工具
eval/               EvalScope 评测工具和精简结果
models/             本地 Hugging Face checkpoint，不提交到 Git
tests/              C ABI、Session、sampling、HTTP 和真实权重测试
scripts/            server、权重转换、benchmark 和命令行 client

engine.cpp          Qwen3.5-0.8B CPU Model/State/Work/forward
runtime.cpp         Engine/Session、prefix/checkpoint、sampling 和 LRU
qwen35.h            公共 C ABI
qwen35.py           C ABI 的 ctypes 包装
```

教学入口见 [from-scratch/README.md](from-scratch/README.md)。课程内部仍使用
`00–09` 编号，因为这些文件代表真实的学习顺序；仓库顶层不再使用阶段编号。

## 构建与验证

本地 checkpoint 默认位于 `models/Qwen3.5-0.8B`：

```sh
make sync
make
make weights
make test
make e2e
```

`make test` 包含不加载真实 forward 的单元测试，以及使用本地 packed weights
验证 Engine、SessionManager、cache/checkpoint 和完整词表 logits 的测试。
`make e2e` 进一步经过真实 tokenizer、HTTP server 和 C++ forward。

官方数值 reference 单独保存在 `reference/`。生成 vectors 并比较 C++ Engine：

```sh
make reference
```

评测工具位于 `eval/`：

```sh
# 终端 1
SLOTS=4 CONTEXT=40960 REQUEST_TIMEOUT=7200 make run LOG_LEVEL=info

# 终端 2
make eval-smoke
```

## 启动

```sh
make run
```

默认监听 `127.0.0.1:8000`。需要 Bearer 鉴权时：

```sh
QWEN_API_KEY='换成随机长字符串' make run
```

另一个终端可以发送最小请求：

```sh
make chat
```

或使用多轮 client：

```sh
./scripts/client "你好" "我刚才说了什么？"
```

Runtime 支持 greedy、temperature、top-p、top-k、presence penalty、固定 seed、
streaming，以及基于 Session live/checkpoint state 的 token prefix 复用。

## 核心对象

```text
Engine
  = 一份加载好的只读 Model/weights

Session
  = 一条可变 token 时间线
  = State + Work + tokens + logits

SessionManager
  = 固定数量的预创建 Session
  = prefix lookup + FREE/IDLE/BUSY + LRU
```

Engine、Session 和 SessionManager 定义在 C++。Python 只处理外围协议。
Runtime 启动时预创建固定数量的 Session，请求期间不会重新加载模型。

## Roadmap

这个项目将继续保持“教学 + PoC”的边界，并沿两条模型路线推进：

1. **完整支持 Qwen3.5-0.8B。** 完善 CPU correctness、正式 CUDA backend、
   长上下文、sampling、Session/cache 行为和端到端回归。
2. **完整支持 Qwen3.8-27B。** 固定官方 reference/probe contract，实现
   text-only BF16 GPU forward、独立 lm_head、长上下文 state 和可验证的
   prefill/decode。
3. 在两条模型路径稳定后，再评估量化、性能优化、prefix block sharing、
   batching 和更多 serving 能力。

每一步都先用官方 full logits 和中间 probe 固定正确性，再做性能优化；不会为了
复用代码而提前引入通用 Tensor、Backend hierarchy 或隐藏模型数学的抽象。

## 当前边界

- 当前正式 Engine 是 Qwen3.5-0.8B 的 CPU BF16-weight / FP32-compute 实现。
- 当前只处理 text-only；尚无 vision、MTP、tools、batching 或量化。
- 固定 Session 数量；全部 BUSY 时返回 429。
- 中断只能在 token 边界发现，一次 CPU forward 尚不能抢占。
- Qwen3.8-27B 的完整正式实现仍属于 roadmap。

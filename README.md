# qwen3x

## 1. 仓库目的

这是一个用于学习和验证 Qwen 混合架构推理的教学与 PoC 项目。

仓库包含两部分：

1. **`from-scratch/`**：从可手算的小例子开始，一步一步搭建
   Qwen3.5-0.8B 的基础数据流，用于理解 embedding、RMSNorm、Attention、
   DeltaNet、SwiGLU、state、prefill 和 decode。
2. **主目录**：实现一个极简、本地优先的 C++ 推理引擎。它直接运行真实模型权重，
   以常驻进程提供 token-in、token-out 的本地 Unix socket 服务。

## 2. 原则

- `from-scratch/` 只用于展示模型结构和计算过程，不考虑性能。课程代码允许少量重复，
  让每一步都能独立阅读、编译和手算。
- 主目录尽量用少量、直接的 C++ 文件完成完整数据流。只做不影响阅读的性能优化，
  不引入通用 Tensor 框架、复杂模板或隐藏模型数学的抽象。
- 主目录使用 **C-oriented, exception-free C++17**：模型计算使用普通数组、指针、
  循环和显式 shape，固定 shape 的计算临时量不使用动态容器，CPU/CUDA 数学接口不暴露 STL 类型。
  C++ 用于 namespace、RAII、并发和真正的动态存储；不使用异常、RTTI 或复杂模板。
  可恢复错误显式返回状态码和错误信息，`assert` 只检查程序内部不变量。
  `auto` 和 lambda 可以用，但只用于局部且一眼可读的代码。
- 项目的优先级是：

```text
correct -> simple -> readable -> usable -> fast
```

## 3. 数据流

Python 负责外围服务，C++ 负责 token 级推理和 completion state：

```text
OpenAI request
  -> Python: HTTP / chat template / tokenizer
  -> Unix socket
  -> C++: token ids -> prefill / decode -> logits / next token
  -> C++ Session: 保存模型 state 和 completion token timeline
  -> Python: decode token ids / streaming / response
```

模型内部的数据流是：

```text
token ids
  -> embedding
  -> N * {
       RMSNorm -> DeltaNet / Attention -> residual
       RMSNorm -> SwiGLU FFN          -> residual
     }
  -> final RMSNorm
  -> lm_head
  -> logits
  -> next token
```

## 4. C++ CLI

先构建可执行文件和模型权重：

```sh
make build/qwen35 weights
```

启动常驻 C++ server：

```sh
./build/qwen35 -l /tmp/qwen35.sock \
  -m build/qwen35-0.8b.bin \
  --parallel 4 \
  --context 4096
```

也可以使用 `make run SOCKET=/tmp/qwen35.sock SLOTS=4 CONTEXT=4096`。

C++ client 从标准输入读取 prompt token IDs，将生成的 token IDs 写到标准输出：

```sh
printf '123 456 789\n' | ./build/qwen35 /tmp/qwen35.sock -n 64
```

`-n` 表示最多生成的 token 数。完整协议见 [protocol.md](protocol.md)。

## 5. 目录结构

```text
from-scratch/       从基础数学到真实 0.8B forward 的教学代码

engine.cpp          模型权重、State、Work 和完整 forward
runtime.cpp         Session、sampling 和 cache 生命周期
main.cpp            main、Unix socket 和 token line protocol
protocol.md         Python 与 C++ 之间的 token line protocol

scripts/            Python 外围服务、权重转换和工具
tests/              Session、协议、HTTP 和真实权重回归
reference/          官方 PyTorch/Transformers 数值 reference
eval/               EvalScope 评测工具和结果
models/             本地模型 checkpoint，不提交到 Git
```

教学入口见 [from-scratch/README.md](from-scratch/README.md)。

## 6. Roadmap

1. **完整支持 Qwen3.5-0.8B**
   - 完善 CPU 与 CUDA 推理；
   - 固定完整数值回归和端到端测试；
   - 完善长上下文、sampling、Session 和 cache 行为。
2. **完整支持 Qwen3.8-27B**
   - 建立官方 reference 和中间 probe；
   - 实现 text-only BF16 GPU prefill/decode；
   - 验证长上下文 state、独立 lm_head 和完整生成链路。

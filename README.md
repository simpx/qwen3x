# qwen3x

## 1. 仓库目的

这是一个用于学习和验证 Qwen 混合架构推理的教学与 PoC 项目。

仓库包含两部分：

1. **`from-scratch/`**：从可手算的小例子开始，一步一步搭建
   Qwen3.5-0.8B 的基础数据流，用于理解 embedding、RMSNorm、Attention、
   DeltaNet、SwiGLU、state、prefill 和 decode。
2. **主目录**：实现一个极简、本地优先的 C++ 推理引擎。它直接运行真实模型权重，
   并提供 Session、Python 绑定和 OpenAI-compatible 服务。

## 2. 原则

- `from-scratch/` 只用于展示模型结构和计算过程，不考虑性能。课程代码允许少量重复，
  让每一步都能独立阅读、编译和手算。
- 主目录尽量用少量、直接的 C++ 文件完成完整数据流。只做不影响阅读的性能优化，
  不引入通用 Tensor 框架、复杂模板或隐藏模型数学的抽象。
- 项目的优先级是：

```text
correct -> simple -> readable -> usable -> fast
```

## 3. 数据流

Python 负责外围服务，C++ 负责 token 级推理和 completion state：

```text
OpenAI request
  -> Python: HTTP / chat template / tokenizer
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

## 4. 目录结构

```text
from-scratch/       从基础数学到真实 0.8B forward 的教学代码

engine.cpp          模型权重、State、Work 和完整 forward
runtime.cpp         Engine、Session、sampling 和 cache 生命周期
qwen35.h            C++ runtime 的公共 C ABI
qwen35.py           C ABI 的 Python 绑定

scripts/            server、权重转换、benchmark 和命令行 client
tests/              C ABI、Session、HTTP 和真实权重回归
reference/          官方 PyTorch/Transformers 数值 reference
eval/               EvalScope 评测工具和结果
models/             本地模型 checkpoint，不提交到 Git
```

教学入口见 [from-scratch/README.md](from-scratch/README.md)。

## 5. Roadmap

1. **完整支持 Qwen3.5-0.8B**
   - 完善 CPU 与 CUDA 推理；
   - 固定完整数值回归和端到端测试；
   - 完善长上下文、sampling、Session 和 cache 行为。
2. **完整支持 Qwen3.8-27B**
   - 建立官方 reference 和中间 probe；
   - 实现 text-only BF16 GPU prefill/decode；
   - 验证长上下文 state、独立 lm_head 和完整生成链路。

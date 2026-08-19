# Stage 4：真实文本的端到端封口

Stage 2 和 Stage 3 有意只接受 token ids：这样读 `forward()` 时看见的是模型，不是 UTF-8、聊天
格式或 sampling policy。本目录把它们接回真实文本，但仍然只有一个很薄的 Python 文件：

```text
prompt text
  -> official tokenizer + Qwen chat template       (qwen35_chat.py)
  -> token ids
  -> ../02-cpu-0.8b/qwen35 or ../03-cuda-0.8b/qwen35_cuda
  -> generated token ids
  -> official tokenizer.decode()
  -> text
```

Python、Transformers 和 tokenizer **不是 C++ inference runtime dependency**。这是开发期的外壳：它
先保证输入 ids 与官方模型完全一致。未来 `07-server/` 才会选择一个小的 native tokenizer/HTTP
实现；那也不应修改 C++ 的 `forward(Model, State, token, Work)`。

它使用和 [Stage 1](../01-hf-reference/README.md) 相同的开发依赖；第一次使用前按该目录的
`requirements-dev.txt` 安装即可。

## 自己试一次

先建好 CPU binary/weights：

```sh
make -C ../02-cpu-0.8b qwen35 weights MODEL=../models/Qwen3.5-0.8B
python3 qwen35_chat.py \
  --model ../models/Qwen3.5-0.8B \
  --engine ../02-cpu-0.8b/qwen35 \
  --weights ../02-cpu-0.8b/build/qwen35-0.8b.bin \
  --prompt '用一句话介绍 DeltaNet。' \
  --max-new-tokens 32
```

把 engine/weights 换成 `../03-cuda-0.8b/qwen35_cuda` 和对应 CUDA weights，即是相同的 GPU
调用。C++ 不需要知道这次输入是 chat、普通补全，还是测试向量。

## 验证

```sh
make cpu  MODEL=../models/Qwen3.5-0.8B
make cuda MODEL=../models/Qwen3.5-0.8B CUDA_ARCH=89
# 或两者：make test
```

每个 target 会重新生成对应 device 的官方 Stage 1 vectors，然后检查：

1. wrapper 得到的官方 chat-template input ids 与 `official_chat` vector 完全相同；
2. C++ greedy generated ids 与同一 official vector 完全相同；
3. 官方 tokenizer 能把生成 ids decode 成非空文本。

这不是聊天质量 benchmark，也不是 HTTP server；它只是保证接下来换 27B 时，真实文本入口不会把
已经验证的 token-level model loop 改坏。

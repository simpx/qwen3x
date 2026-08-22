# Qwen3.5-0.8B：独立 plain C++ CPU 版本

这个目录是 `00-lessons/09` 的正式版本，也是一个完整的文字 e2e 闭环：

```text
官方 checkpoint
  -> pack_weights.py                          -> build/qwen35-0.8b.bin

用户文字
  -> chat.py：官方 tokenizer + chat template  -> token ids
  -> qwen35：完整 CPU prefill/decode           -> generated token ids
  -> chat.py：官方 tokenizer.decode            -> 输出文字
```

C++ 模型核心不依赖 PyTorch、Transformers 或其他仓库目录。它固定为 Qwen3.5-0.8B
text backbone、batch=1、BF16 weights + FP32 activations，只使用 C++17 和 POSIX `mmap`。
Python 只用于一次性权重转换、文字 tokenizer 外壳和测试。

## 目录内容

```text
qwen35.cpp       完整 Model + State + Work、prefill、decode、greedy generation
pack_weights.py  官方 safetensors -> 固定顺序 mmap bin，只用 Python 标准库
chat.py          文字/chat e2e 薄层
reference.json   本目录自带的小型官方 CPU 回归契约
test_cpu.py      不依赖其他 stage 的 forward/greedy/state 测试
test_e2e.py      文字 -> token -> CPU -> token -> 文字测试
test_oracle.py   可选的 Stage 1 full-vocabulary logits 重型验证
```

代码仍然只有三类模型数据：

```cpp
struct Model;  // mmap 中的固定权重
struct State;  // 跨 token：position、KV cache、DeltaNet memory、conv history
struct Work;   // 当前 forward 可覆盖的 FP32 临时向量与 logits
```

没有 Tensor、Backend、Session、operator dispatch、通用 config 或 class hierarchy。

`--worker` 是给 `04-runtime` 使用的常驻外围协议：`start` 建立一个请求的 State，`next`
推进一个 token，`reset` 清空请求状态。它不改变 forward，也不把 HTTP/tokenizer 放进 C++。

## 1. 编译和打包

```sh
cd 02-cpu-0.8b
make
make weights MODEL=../models/Qwen3.5-0.8B
```

`weights` 每次都会重新检查 checkpoint config、320 个 tensor 的 dtype/shape/byte size，
再原子写入 packed bin，因此切换 `MODEL` 时不会静默复用旧权重。

## 2. 直接运行 token ids

```sh
./qwen35 --forward build/qwen35-0.8b.bin 248044,198,198
./qwen35 --generate build/qwen35-0.8b.bin 248044,198,198 8
```

`forward()` 每次处理一个 token；`prefill()` 只是连续调用它，`decode()` 使用同一个
`State` 继续调用。C++ 输出 token ids，不在模型数学中混入 tokenizer。

## 3. 真实文字/chat e2e

文字入口需要支持 Qwen3.5 的 Transformers：

```sh
python3 -m pip install -r requirements.txt
python3 chat.py \
  --model ../models/Qwen3.5-0.8B \
  --prompt '用一句话介绍 DeltaNet。' \
  --max-new-tokens 32
```

`chat.py` 默认调用本目录的 `./qwen35` 和 `build/qwen35-0.8b.bin`。它从 checkpoint
读取官方 tokenizer 和 chat template，C++ engine 本身仍然只处理 token ids。

## 4. 独立验证

```sh
make quick-test MODEL=../models/Qwen3.5-0.8B
make test MODEL=../models/Qwen3.5-0.8B
```

`make test` 不进入其他 stage，验证：

1. BF16、argmax、RMSNorm 微型 self-test；
2. 三组固定官方 contract 的 next-token/logit、greedy continuation；
3. `prefill()+decode()` 与连续 `forward()` 留下完全相同的 state/logits；
4. 官方 chat template 的 input ids、CPU generated ids 和最终非空文字。

`reference.json` 是小型、可提交的 smoke contract。需要重新用官方模型验证每一步完整
248,320 词表 logits 时，再运行：

```sh
make oracle-test MODEL=../models/Qwen3.5-0.8B
```

这个重型 target 会显式调用 `../01-hf-reference`；它是额外数值裁判，不是本目录正常运行
或日常测试的依赖。

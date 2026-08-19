# Stage 2：plain C++ 0.8B CPU reference

这里是 `lessons/09` 的可演进版本，而不是第二套模型设计。它固定为官方
Qwen3.5-0.8B text backbone、batch=1、CPU、BF16 weights + FP32 activations。

代码刻意只有三种数据：

```cpp
struct Model;  // mmap 的固定权重
struct State;  // 跨 token：position、KV cache、GDN recurrent state、conv history
struct Work;   // 当前 forward 的临时 FP32 vectors 与 logits
```

`forward()` 是完整模型的一步；`prefill()` 和 `decode()` 只是两个清楚的调用方式，内部仍然
直接调用它。没有 Tensor、Backend、Session、operator dispatch 或通用 config。

```text
prefill(A, B, C, D)  -> State
decode(E)            -> 同一个 State
decode(F)            -> 同一个 State
```

## 权重格式

`pack_weights.py` 只接受一个 checkpoint：Qwen3.5-0.8B。它读取官方 safetensors，按
`qwen35.cpp::Model` 的读取顺序写入对齐的 mmap 文件。格式与 `lessons/09` 暂时相同，方便两个
CPU reference 对照；它不是 GGUF，也不是通用模型转换器。

## 编译与验证

```sh
make
make test MODEL=../models/Qwen3.5-0.8B
```

`make test` 会在缺少 Stage 1 vectors 时先生成它们，随后：

1. 从官方 checkpoint 打包本 stage 的 model bin；
2. 对每个官方 case 运行 `--trace-logits`，比较每一个 token 之后的完整 vocabulary logits；
3. 比较 greedy continuation；
4. 检查 `prefill()+decode()` 与手写连续 `forward()` 得到 bitwise 相同的 state/logits。

运行单条命令：

```sh
./qwen35 --forward build/qwen35-0.8b.bin 248044,198,198
./qwen35 --generate build/qwen35-0.8b.bin 248044,198,198 8
```

tokenizer 仍然不属于本目录；Stage 4 才会用官方 tokenizer/chat template 给这个 token-id CLI
套一层极薄 wrapper。

# Qwen3.5-0.8B CUDA performance path

这里不是第二套教学实现。模型结构、权重格式和可逐行阅读的 CPU forward 都在
[`../00-lessons/09_qwen35_0_8b.cpp`](../00-lessons/09_qwen35_0_8b.cpp)。本目录只回答另一个问题：
同一个固定 Qwen3.5-0.8B，在 NVIDIA GPU 上怎样避免明显的工程浪费。

它仍然不是通用 runtime：只支持这个模型、text、batch=1、单 GPU 和 greedy decoding；没有
Tensor class、operator registry 或模型分发器。不同点是性能必要项可以出现：

- 所有重 linear `W*x` 使用 cuBLAS Tensor Cores；weight 与 linear 输入是 BF16，累加、
  branch 和 recurrent state 是 FP32；
- logits argmax 留在 GPU，正常 decode 不再每 token 下载完整 vocabulary；
- `--serve` 将模型、GPU 权重和 work buffers 常驻，评测的每道题只 reset cache/state，
  不再重新启动进程、上传 checkpoint。

## 编译

~~~
cd qwen35-0.8b
make                     # nvcc + cuBLAS，默认 sm_89；可用 CUDA_ARCH 覆盖
make cuda-test
~~~

在某些 WSL 安装里，系统残留的旧 `libcuda.so` 会盖过 Windows driver；Make targets 会在
启动前把 `/usr/lib/wsl/lib` 放进 `LD_LIBRARY_PATH`。直接运行 binary 时也使用同样前缀：

~~~
LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH ./qwen35_cuda --self-test
~~~

CPU 教学版与权重 packer 在 `../00-lessons/`：

~~~
cd ../00-lessons
make
python3 09_pack_weights.py ../models/Qwen3.5-0.8B ../models/qwen35-0.8b.bin
~~~

回到本目录运行 GPU：

~~~
./qwen35_cuda --forward ../models/qwen35-0.8b.bin 248044,198,198
./qwen35_cuda --generate ../models/qwen35-0.8b.bin 248044,198,198 16
~~~

## 正确性先于性能

每次改 kernel 或 cuBLAS 调用，都先对照第 09 课和官方 checkpoint：

~~~
make cuda-oracle MODEL=../models/Qwen3.5-0.8B

# 需要 Transformers >= 5、torch 和 CUDA；比较完整 vocabulary logits 与 greedy ids。
make official-oracle MODEL=../models/Qwen3.5-0.8B WEIGHTS=../models/qwen35-0.8b.bin
~~~

第一项是快速的 pinned regression；第二项以官方 `Qwen3_5ForCausalLM` FP32 forward 为黄金值。
高性能路径比第 09 课多了每个 linear 输入的 BF16 舍入；因此 CPU 的全词表阈值为 `1e-4`，CUDA
为 `0.1`，但两者都必须保持官方的 greedy tokens。这是明确的性能精度边界，而不是把 GPU
近似结果伪装成 FP32-activation 数值黄金值。

## 持久 worker 与评测

`--serve` 的 stdin 每一行是 `new_tokens<TAB>id,id,...`，输出一行 `generated: ...`。它不是
HTTP server，只是给本目录的 Python evaluator 使用的极小进程协议。`eval_mmlu.py` 和
`eval_mmlu_pro.py` 会自动使用它，因此权重只上传一次：

~~~
pip install transformers datasets duckdb huggingface_hub
make mmlu-eval MODEL=../models/Qwen3.5-0.8B WEIGHTS=../models/qwen35-0.8b.bin \
  SUBJECT=abstract_algebra

make mmlu-pro-eval MODEL=../models/Qwen3.5-0.8B WEIGHTS=../models/qwen35-0.8b.bin LIMIT=20

# 真实生成并计分 3,610/12,032 (30.003%) 道题；RTX 4080 16 GiB 用 5 个常驻 worker。
# 这是短输入 coverage run，不是可以同官方 29.7% 比较的 MMLU-Pro leaderboard run。
make mmlu-pro-fast-coverage MODEL=../models/Qwen3.5-0.8B WEIGHTS=../models/qwen35-0.8b.bin
~~~

MMLU-Pro 的 Qwen model-card non-thinking 参考值为 29.7%。当前脚本复用官方 runner 风格的
5-shot CoT prompt 和答案抽取，但仍是固定抽样而非 leaderboard 复现：官方跑完整 12,032 题，
并具有更复杂的 textual stop/batching。这里的目标是让性能版本能作为实际、可重复的 GPU
runner，同时不把这些 serving 功能带回第 09 课。

### 一小时端到端 coverage

完整官方风格 MMLU-Pro 有 12,032 题、five-shot CoT 和很长的生成；这个 batch=1、单 GPU
教学性能路径不应把它伪装成一个高吞吐 serving engine。`mmlu-pro-fast-coverage` 是另一条明确
命名的质量回归：它用 Qwen 的 text chat framing 关闭 thinking，要求只生成 A--J 的一个选择，
并按每个 category 的原始比例选最短的 3,610 个 prompts。题目、选项、模型 forward、greedy
generation 和答案计分都是真的；没有 logits 对比或 mock。但是短输入选择及不同 prompt 会使
准确率有偏，因此它**不能**与 Qwen model card 的 29.7% 作比较。

在本机 RTX 4080 16 GiB 上，5 个 worker 约占 14.1 GiB 显存；100 题短输入基准为 57.1 秒。
完整命令实际完成了 3,610/12,032（30.003%）题，用时 3,568.1 秒，得分为 845/3,610（23.4%，
0 个格式无效）。因此这个一小时 coverage 上限已真实验证；脚本的
`--time-limit-seconds=3600` 仍会在题目边界停止，若机器被别的程序占用，它会报告实际完成的题数，
而不会假称完成 30%。

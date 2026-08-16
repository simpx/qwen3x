# 真实 Qwen3.5-0.8B 的独立小目录：`make` 只编译 qwen35。

CUDA_ARCH ?= 89

.PHONY: all test oracle-test official-oracle mmlu-eval mmlu-pro-eval cuda cuda-test cuda-oracle-test clean

all: qwen35

qwen35: qwen35.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic qwen35.cpp -o qwen35

test: qwen35
	./qwen35 --self-test

oracle-test: qwen35
	@test -n "$(MODEL)" || (echo "usage: make oracle-test MODEL=/path/to/Qwen3.5-0.8B" >&2; exit 2)
	./test.sh "$(MODEL)"

# 真正调用官方 Transformers FP32 model 的端到端 numerical oracle。它同时检查 CPU、CUDA
# 的完整词表 logits 与 greedy decode；需要本机有支持 qwen3_5 的新 Transformers 和 CUDA。
official-oracle: qwen35 qwen35_cuda
	@test -n "$(MODEL)" || (echo "usage: make official-oracle MODEL=/path/to/Qwen3.5-0.8B [WEIGHTS=out/qwen35-0.8b.bin]" >&2; exit 2)
	python3 test_official.py "$(MODEL)" $(if $(WEIGHTS),--weights "$(WEIGHTS)")

# 生成式 MMLU accuracy：默认 CUDA，subject 默认 abstract_algebra 的完整 100 道 test 题。
mmlu-eval: qwen35_cuda
	@test -n "$(MODEL)" -a -n "$(WEIGHTS)" || (echo "usage: make mmlu-eval MODEL=/path/to/Qwen3.5-0.8B WEIGHTS=out/qwen35-0.8b.bin [SUBJECT=abstract_algebra] [LIMIT=20]" >&2; exit 2)
	python3 eval_mmlu.py "$(MODEL)" "$(WEIGHTS)" --subject "$(or $(SUBJECT),abstract_algebra)" $(if $(LIMIT),--limit "$(LIMIT)")

# 对照 Qwen 模型卡：复用官方 MMLU-Pro 的 5-shot CoT prompt 和答案抽取。
# 教学引擎的 2048 context 小于官方 runner 的 4096，且默认只抽样 20 题；因此这是
# 可复现的合理性检查，不是冒充官方的 12,032 题 leaderboard 复现。
mmlu-pro-eval: qwen35_cuda
	@test -n "$(MODEL)" -a -n "$(WEIGHTS)" || (echo "usage: make mmlu-pro-eval MODEL=/path/to/Qwen3.5-0.8B WEIGHTS=out/qwen35-0.8b.bin [LIMIT=20]" >&2; exit 2)
	python3 eval_mmlu_pro.py "$(MODEL)" "$(WEIGHTS)" $(if $(LIMIT),--limit "$(LIMIT)")

# 可选 GPU 路径。默认 sm_89 对应当前 RTX 4080 SUPER；其他 NVIDIA GPU 可覆盖：
# make cuda CUDA_ARCH=120
cuda: qwen35_cuda

qwen35_cuda: qwen35_cuda.cu qwen35.cpp
	nvcc -O3 -std=c++17 -arch=sm_$(CUDA_ARCH) qwen35_cuda.cu -o qwen35_cuda

cuda-test: qwen35_cuda
	./qwen35_cuda --self-test

cuda-oracle-test: qwen35_cuda qwen35
	@test -n "$(MODEL)" || (echo "usage: make cuda-oracle-test MODEL=/path/to/Qwen3.5-0.8B" >&2; exit 2)
	./test_cuda.sh "$(MODEL)"

clean:
	rm -f qwen35 qwen35_cuda

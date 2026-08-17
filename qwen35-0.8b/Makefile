# 这里不是教学课；真实模型的 CPU reference 在 ../lessons/09_qwen35_0_8b.cpp。
# 这里的唯一 binary 是高性能 CUDA 路径。保持命令直接，避免引入 CMake 或 framework。

CUDA_ARCH ?= 89
CPU_REFERENCE = ../lessons/09_qwen35_0_8b
# WSL 可同时有陈旧的 distro libcuda；在启动进程前优先放 Windows driver 的目录。普通
# Linux 上该目录不存在，后面的原 LD_LIBRARY_PATH / 默认搜索路径仍照常工作。
CUDA_RUN ?= LD_LIBRARY_PATH=/usr/lib/wsl/lib:$$LD_LIBRARY_PATH

.PHONY: all cuda-test cuda-oracle official-oracle mmlu-eval mmlu-pro-eval clean

all: qwen35_cuda

qwen35_cuda: qwen35_cuda.cu ../lessons/09_qwen35_0_8b.cpp
	nvcc -O3 -std=c++17 -arch=sm_$(CUDA_ARCH) qwen35_cuda.cu -lcublas -o qwen35_cuda

$(CPU_REFERENCE): ../lessons/09_qwen35_0_8b.cpp
	$(MAKE) -C ../lessons 09_qwen35_0_8b

cuda-test: qwen35_cuda
	$(CUDA_RUN) ./qwen35_cuda --self-test

# 对照同一权重格式的 CPU 第 09 课；它是 GPU kernel 改动后的最短回归。
cuda-oracle: qwen35_cuda $(CPU_REFERENCE)
	@test -n "$(MODEL)" || (echo "usage: make cuda-oracle MODEL=/path/to/Qwen3.5-0.8B" >&2; exit 2)
	$(CUDA_RUN) ./test_cuda.sh "$(MODEL)"

# 真正调用官方 Transformers FP32 model，比较完整词表 logits 和 greedy token ids。
official-oracle: qwen35_cuda $(CPU_REFERENCE)
	@test -n "$(MODEL)" || (echo "usage: make official-oracle MODEL=/path/to/Qwen3.5-0.8B [WEIGHTS=/path/to/qwen35-0.8b.bin]" >&2; exit 2)
	$(CUDA_RUN) python3 test_official.py "$(MODEL)" $(if $(WEIGHTS),--weights "$(WEIGHTS)") --cpu-engine "$(CPU_REFERENCE)" --cuda-engine ./qwen35_cuda

mmlu-eval: qwen35_cuda
	@test -n "$(MODEL)" -a -n "$(WEIGHTS)" || (echo "usage: make mmlu-eval MODEL=/path/to/Qwen3.5-0.8B WEIGHTS=/path/to/qwen35-0.8b.bin [SUBJECT=abstract_algebra] [LIMIT=20]" >&2; exit 2)
	$(CUDA_RUN) python3 eval_mmlu.py "$(MODEL)" "$(WEIGHTS)" --subject "$(or $(SUBJECT),abstract_algebra)" $(if $(LIMIT),--limit "$(LIMIT)")

mmlu-pro-eval: qwen35_cuda
	@test -n "$(MODEL)" -a -n "$(WEIGHTS)" || (echo "usage: make mmlu-pro-eval MODEL=/path/to/Qwen3.5-0.8B WEIGHTS=/path/to/qwen35-0.8b.bin [LIMIT=20]" >&2; exit 2)
	$(CUDA_RUN) python3 eval_mmlu_pro.py "$(MODEL)" "$(WEIGHTS)" $(if $(LIMIT),--limit "$(LIMIT)")

clean:
	rm -f qwen35_cuda

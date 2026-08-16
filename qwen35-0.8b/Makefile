# 真实 Qwen3.5-0.8B 的独立小目录：`make` 只编译 qwen35。

CUDA_ARCH ?= 89

.PHONY: all test oracle-test cuda cuda-test cuda-oracle-test clean

all: qwen35

qwen35: qwen35.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic qwen35.cpp -o qwen35

test: qwen35
	./qwen35 --self-test

oracle-test: qwen35
	@test -n "$(MODEL)" || (echo "usage: make oracle-test MODEL=/path/to/Qwen3.5-0.8B" >&2; exit 2)
	./test.sh "$(MODEL)"

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

# Stage 3：一个 nvcc 命令，一个固定 0.8B CUDA binary。没有 CMake 或 framework。
NVCC ?= nvcc
PYTHON ?= python3
CUDA_ARCH ?= 89
MODEL ?= ../models/Qwen3.5-0.8B
VECTORS ?= ../01-hf-reference/build/cuda
BUILD ?= build
WEIGHTS ?= $(BUILD)/qwen35-0.8b.bin
CUDA_RUN ?= LD_LIBRARY_PATH=/usr/lib/wsl/lib:$$LD_LIBRARY_PATH

.PHONY: all vectors weights cuda-test test clean

all: qwen35_cuda

qwen35_cuda: qwen35_cuda.cu qwen35_cpu.cpp
	$(NVCC) -O3 -std=c++17 -arch=sm_$(CUDA_ARCH) qwen35_cuda.cu -lcublas -o qwen35_cuda

vectors:
	$(MAKE) -C ../01-hf-reference cuda MODEL=$(abspath $(MODEL))

weights: $(WEIGHTS)

$(WEIGHTS): pack_weights.py
	mkdir -p $(BUILD)
	$(PYTHON) pack_weights.py $(MODEL) $(WEIGHTS)

cuda-test: qwen35_cuda
	$(CUDA_RUN) ./qwen35_cuda --self-test

test: qwen35_cuda vectors weights
	$(CUDA_RUN) $(PYTHON) test_cuda.py --engine ./qwen35_cuda --weights $(WEIGHTS) --vectors $(VECTORS)

clean:
	rm -f qwen35_cuda

# Stage 4：把官方 tokenizer 放在 Python 薄层，C++ core 仍然只接收 token ids。
PYTHON ?= python3
MODEL ?= ../models/Qwen3.5-0.8B
CPU_ENGINE ?= ../02-cpu-0.8b/qwen35
CPU_WEIGHTS ?= ../02-cpu-0.8b/build/qwen35-0.8b.bin
CUDA_ENGINE ?= ../03-cuda-0.8b/qwen35_cuda
CUDA_WEIGHTS ?= ../03-cuda-0.8b/build/qwen35-0.8b.bin
CUDA_ARCH ?= 89
CUDA_RUN ?= LD_LIBRARY_PATH=/usr/lib/wsl/lib:$$LD_LIBRARY_PATH

.PHONY: all cpu cuda test clean

all: test

# Each target builds only the fixed binary it will call and regenerates that device's official
# vector bundle.  There is no package installation, CMake project, server, or model framework.
cpu:
	$(MAKE) -C ../01-hf-reference cpu MODEL=$(abspath $(MODEL))
	$(MAKE) -C ../02-cpu-0.8b qwen35 weights MODEL=$(abspath $(MODEL))
	$(PYTHON) test_e2e.py --model $(MODEL) --engine $(CPU_ENGINE) --weights $(CPU_WEIGHTS) --vectors ../01-hf-reference/build/cpu

cuda:
	$(MAKE) -C ../01-hf-reference cuda MODEL=$(abspath $(MODEL))
	$(MAKE) -C ../03-cuda-0.8b qwen35_cuda weights MODEL=$(abspath $(MODEL)) CUDA_ARCH=$(CUDA_ARCH)
	$(CUDA_RUN) $(PYTHON) test_e2e.py --model $(MODEL) --engine $(CUDA_ENGINE) --weights $(CUDA_WEIGHTS) --vectors ../01-hf-reference/build/cuda

test: cpu cuda

clean:
	@true  # Stage 4 owns no compiled artifact; other stages own their own `make clean`.

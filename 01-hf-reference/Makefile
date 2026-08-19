# Stage 1 只负责官方 oracle。MODEL 是本地 checkpoint，默认不下载任何文件。
PYTHON ?= python3
MODEL ?= ../models/Qwen3.5-0.8B
OUT ?= build
# `dump` / `test` are low-level commands.  The named targets below deliberately place
# CPU and CUDA oracle outputs in separate directories: they are both FP32, but their
# kernels and reduction orders are different and must never be silently mixed.
DEVICE ?= cpu

.PHONY: all dump test cpu cuda clean

all: cpu

cpu:
	$(MAKE) test DEVICE=cpu OUT=build/cpu

cuda:
	$(MAKE) test DEVICE=cuda OUT=build/cuda

dump:
	$(PYTHON) dump_vectors.py --model "$(MODEL)" --out "$(OUT)" --device "$(DEVICE)"

test: dump
	$(PYTHON) test_reference.py --model "$(MODEL)" --vectors "$(OUT)" --device "$(DEVICE)"

clean:
	rm -rf build

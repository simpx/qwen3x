# Stage 2：一份固定模型、一个 plain C++ binary。没有 CMake，也没有 framework。
PYTHON ?= python3
MODEL ?= ../models/Qwen3.5-0.8B
VECTORS ?= ../01-hf-reference/build/cpu
BUILD ?= build
WEIGHTS ?= $(BUILD)/qwen35-0.8b.bin

.PHONY: all vectors weights quick-test test clean

all: qwen35

qwen35: qwen35.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic qwen35.cpp -o qwen35

vectors:
	$(MAKE) -C ../01-hf-reference cpu MODEL=$(abspath $(MODEL))

weights: $(WEIGHTS)

$(WEIGHTS): pack_weights.py
	mkdir -p $(BUILD)
	$(PYTHON) pack_weights.py $(MODEL) $(WEIGHTS)

quick-test: qwen35 weights
	$(PYTHON) test_cpu.py --engine ./qwen35 --weights $(WEIGHTS) --vectors $(VECTORS) --case lesson_short

test: qwen35 vectors weights
	$(PYTHON) test_cpu.py --engine ./qwen35 --weights $(WEIGHTS) --vectors $(VECTORS) --skip-greedy --skip-state
	$(PYTHON) test_cpu.py --engine ./qwen35 --weights $(WEIGHTS) --vectors $(VECTORS) --skip-trace --skip-state
	$(PYTHON) test_cpu.py --engine ./qwen35 --weights $(WEIGHTS) --vectors $(VECTORS) --case lesson_short --skip-trace --skip-greedy

clean:
	rm -f qwen35

# 独立 CPU 目录：打包、C++ engine、固定回归和文字 e2e 都在这里。
PYTHON ?= python3
MODEL ?= ../models/Qwen3.5-0.8B
VECTORS ?= ../01-hf-reference/build/cpu
BUILD ?= build
WEIGHTS ?= $(BUILD)/qwen35-0.8b.bin
REFERENCE ?= reference.json

.PHONY: all weights quick-test test e2e-test vectors oracle-test clean

all: qwen35

qwen35: qwen35.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic qwen35.cpp -o qwen35

weights:
	mkdir -p $(BUILD)
	$(PYTHON) pack_weights.py "$(MODEL)" "$(WEIGHTS)"

quick-test: qwen35 weights
	./qwen35 --self-test
	$(PYTHON) test_cpu.py --engine ./qwen35 --weights "$(WEIGHTS)" --reference "$(REFERENCE)" --case lesson_short

test: qwen35 weights
	./qwen35 --self-test
	$(PYTHON) test_cpu.py --engine ./qwen35 --weights "$(WEIGHTS)" --reference "$(REFERENCE)"
	$(PYTHON) test_e2e.py --model "$(MODEL)" --engine ./qwen35 --weights "$(WEIGHTS)" --reference "$(REFERENCE)"

e2e-test: qwen35 weights
	$(PYTHON) test_e2e.py --model "$(MODEL)" --engine ./qwen35 --weights "$(WEIGHTS)" --reference "$(REFERENCE)"

# 可选重型裁判：重新调用官方 Transformers，并比较每一步完整词表 logits。
vectors:
	$(MAKE) -C ../01-hf-reference cpu MODEL=$(abspath $(MODEL))

oracle-test: qwen35 weights vectors
	$(PYTHON) test_oracle.py --engine ./qwen35 --weights "$(WEIGHTS)" --vectors "$(VECTORS)"

clean:
	rm -f qwen35 "$(WEIGHTS)"

# reference/ 只负责官方 oracle。MODEL 是本地 checkpoint，默认不下载任何文件。
UV ?= uv
PYTHON := $(UV) run --locked python
MODEL ?= ../models/Qwen3.5-0.8B
OUT ?= build
# `dump` / `test` are low-level commands.  The named targets below deliberately place
# CPU and CUDA oracle outputs in separate directories: they are both FP32, but their
# kernels and reduction orders are different and must never be silently mixed.
DEVICE ?= cpu
DTYPE ?= float32
CACHE ?= static
MAX_CONTEXT ?= 40960
CHAT_TEMPLATE ?=
CHAT_TEMPLATE_ARG := $(if $(CHAT_TEMPLATE),--chat-template "$(CHAT_TEMPLATE)",)
PORT ?= 8002
SERVED_MODEL ?= qwen3.5-0.8b-reference

ENGINE_DIR ?= ..
LIBRARY ?= $(ENGINE_DIR)/build/libqwen35.so
BIN ?= $(ENGINE_DIR)/build/qwen35-0.8b.bin
VECTORS ?= build/cpu
REPORT ?= $(ENGINE_DIR)/build/reference-report.json
CXXFLAGS ?=

.PHONY: all sync dump test cpu cuda compare serve clean

all: cpu

sync:
	$(UV) sync --locked

cpu:
	$(MAKE) test DEVICE=cpu OUT=build/cpu

cuda:
	$(MAKE) test DEVICE=cuda OUT=build/cuda

dump:
	$(PYTHON) dump_vectors.py --model "$(MODEL)" --out "$(OUT)" --device "$(DEVICE)" $(CHAT_TEMPLATE_ARG)

test: dump
	$(PYTHON) test_reference.py --model "$(MODEL)" --vectors "$(OUT)" --device "$(DEVICE)"

compare:
	$(MAKE) -C "$(ENGINE_DIR)" reference-library
	PYTHONPATH="$(abspath $(ENGINE_DIR))" $(PYTHON) compare_engine.py \
		--library "$(LIBRARY)" \
		--bin "$(BIN)" \
		--vectors "$(VECTORS)" \
		--model "$(MODEL)" \
		--chat-template "$(CHAT_TEMPLATE)" \
		--report "$(REPORT)" \
		--cxxflags "$(CXXFLAGS)"

serve:
	$(PYTHON) server.py \
		--model "$(MODEL)" \
		--chat-template "$(CHAT_TEMPLATE)" \
		--served-model-name "$(SERVED_MODEL)" \
		--port "$(PORT)" \
		--device "$(DEVICE)" \
		--dtype "$(DTYPE)" \
		--cache "$(CACHE)" \
		--max-context-tokens "$(MAX_CONTEXT)"

clean:
	rm -rf build

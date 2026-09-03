# reference/ 只负责官方 oracle。MODEL 是本地 checkpoint，默认不下载任何文件。
UV ?= uv
PYTHON := $(UV) run --locked python
MODEL ?= ../build/models/Qwen3.5-0.8B
OUT ?= build
# `dump` / `test` are low-level commands.  The named targets below deliberately place
# CPU and CUDA oracle outputs in separate directories: they are both FP32, but their
# kernels and reduction orders are different and must never be silently mixed.
DEVICE ?= cpu
DTYPE ?= float32
CACHE ?= static
MAX_CONTEXT ?= 40960
CHAT_TEMPLATE ?= chat_template.jinja
CHAT_TEMPLATE_ARG := $(if $(CHAT_TEMPLATE),--chat-template "$(CHAT_TEMPLATE)",)
DUMP_ARGS ?=
COMPARE_ARGS ?=
PORT ?= 8002
SERVED_MODEL ?= qwen3.5-0.8b-reference

ENGINE_DIR ?= ..
LIBRARY ?= build/libqwen35.so
CUDA_LIBRARY ?= build/libqwen35-cuda.so
BIN ?= $(ENGINE_DIR)/build/qwen35-0.8b-model.bin
VECTORS ?= build/cpu
REPORT ?= build/reference-report.json
CUDA_REPORT ?= build/cuda-reference-report.json
LLAMA_DEBUG ?= ../build/llama.cpp-b10516/build-cuda/bin/llama-debug
LLAMA_GGUF ?= ../build/models/Unsloth-Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf
QWEN ?= ../build/qwen35-cuda
QWEN_BIN ?= ../build/qwen35-9b-q8_0-model.bin
QWEN_RENDER ?= ../build/qwen35-0.8b-render.bin
CXX ?= c++
CXXFLAGS ?=
CUDA_LIB_DIR ?= /usr/local/cuda/targets/x86_64-linux/lib
ENGINE_CXXFLAGS := -O3 -std=c++17 -fno-exceptions -fno-rtti \
	-Wall -Wextra -Wpedantic -march=native
THIRD_PARTY_FLAGS := -DSPDLOG_COMPILED_LIB -DSPDLOG_NO_EXCEPTIONS \
	-I$(ENGINE_DIR)/third_party/spdlog/include
SPDLOG_SRC := $(wildcard $(ENGINE_DIR)/third_party/spdlog/src/*.cpp)

.PHONY: all sync dump test tool-test llama-smoke-test cpu cuda compare compare-cuda \
	compare-4b compare-cuda-4b llama-smoke llama-smoke-run serve clean

all: cpu

tool-test:
	cd "$(ENGINE_DIR)" && python3 -m unittest -v reference.test_tool_calls

llama-smoke-test:
	cd "$(ENGINE_DIR)" && python3 -m unittest -v reference.test_compare_llama

sync:
	$(UV) sync --locked

cpu:
	$(MAKE) test DEVICE=cpu OUT=build/cpu

cuda:
	$(MAKE) test DEVICE=cuda OUT=build/cuda

dump:
	$(PYTHON) dump_vectors.py --model "$(MODEL)" --out "$(OUT)" \
		--device "$(DEVICE)" --dtype "$(DTYPE)" $(CHAT_TEMPLATE_ARG) $(DUMP_ARGS)

test: dump
	$(PYTHON) test_reference.py --model "$(MODEL)" --vectors "$(OUT)" --device "$(DEVICE)"

compare: $(LIBRARY)
	$(PYTHON) compare_engine.py \
		--library "$(LIBRARY)" \
		--bin "$(BIN)" \
		--vectors "$(VECTORS)" \
		--model "$(MODEL)" \
		--chat-template "$(CHAT_TEMPLATE)" \
		--report "$(REPORT)" \
		--cxxflags "$(CXXFLAGS)" $(COMPARE_ARGS)

compare-cuda: $(CUDA_LIBRARY)
	$(PYTHON) compare_engine.py \
		--library "$(CUDA_LIBRARY)" \
		--bin "$(BIN)" \
		--vectors "$(VECTORS)" \
		--model "$(MODEL)" \
		--chat-template "$(CHAT_TEMPLATE)" \
		--report "$(CUDA_REPORT)" \
		--tolerance-key cuda_max_abs_error \
		--cxxflags "CUDA $(CXXFLAGS)" $(COMPARE_ARGS)

compare-4b:
	$(MAKE) compare LIBRARY=build/libqwen35.so BIN=../build/qwen35-4b-model.bin \
		MODEL=../build/models/Qwen3.5-4B VECTORS=build/4b-mixed-fp32 \
		REPORT=build/4b-cpu-reference-report.json COMPARE_ARGS=--behavior-only

compare-cuda-4b:
	$(MAKE) compare-cuda CUDA_LIBRARY=build/libqwen35-cuda.so BIN=../build/qwen35-4b-model.bin \
		MODEL=../build/models/Qwen3.5-4B VECTORS=build/4b-mixed-fp32 \
		CUDA_REPORT=build/4b-reference-report.json COMPARE_ARGS=--behavior-only

llama-smoke:
	timeout 15m $(MAKE) llama-smoke-run

llama-smoke-run:
	$(MAKE) -C "$(ENGINE_DIR)" cuda
	python3 compare_llama.py \
		--llama-debug "$(LLAMA_DEBUG)" --gguf "$(LLAMA_GGUF)" \
		--qwen "$(QWEN)" --bin "$(QWEN_BIN)" --render "$(QWEN_RENDER)"

$(LIBRARY): $(ENGINE_DIR)/engine.cpp $(ENGINE_DIR)/runtime.cpp \
		$(ENGINE_DIR)/log.cpp $(ENGINE_DIR)/qwen35.h \
		$(ENGINE_DIR)/internal.h $(ENGINE_DIR)/model_config.h \
		$(ENGINE_DIR)/q8.h $(SPDLOG_SRC)
	mkdir -p "$(dir $(LIBRARY))"
	$(CXX) $(ENGINE_CXXFLAGS) $(CXXFLAGS) $(THIRD_PARTY_FLAGS) \
		-fPIC -shared -I$(ENGINE_DIR) $(ENGINE_DIR)/engine.cpp \
		$(ENGINE_DIR)/runtime.cpp $(ENGINE_DIR)/log.cpp $(SPDLOG_SRC) \
		-pthread -o "$(LIBRARY)"

$(CUDA_LIBRARY): $(ENGINE_DIR)/arch/cuda/engine.cu $(ENGINE_DIR)/runtime.cpp \
		$(ENGINE_DIR)/log.cpp $(ENGINE_DIR)/qwen35.h \
		$(ENGINE_DIR)/internal.h $(ENGINE_DIR)/model_config.h \
		$(ENGINE_DIR)/q8.h $(SPDLOG_SRC)
	mkdir -p "$(dir $(CUDA_LIBRARY))"
	nvcc -O3 -std=c++17 -arch=native \
		-Xcompiler=-fPIC,-fno-exceptions,-fno-rtti,-Wall,-Wextra \
		$(THIRD_PARTY_FLAGS) -shared -I$(ENGINE_DIR) \
		$(ENGINE_DIR)/arch/cuda/engine.cu $(ENGINE_DIR)/runtime.cpp \
		$(ENGINE_DIR)/log.cpp $(SPDLOG_SRC) -Xcompiler=-pthread \
		-L$(CUDA_LIB_DIR) -lcublas -Xlinker -rpath -Xlinker $(CUDA_LIB_DIR) \
		-o "$(CUDA_LIBRARY)"

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

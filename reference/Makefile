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
PORT ?= 8002
SERVED_MODEL ?= qwen3.5-0.8b-reference

ENGINE_DIR ?= ..
LIBRARY ?= build/libqwen35.so
CUDA_LIBRARY ?= build/libqwen35-cuda.so
BIN ?= $(ENGINE_DIR)/build/qwen35-0.8b-model.bin
VECTORS ?= build/cpu
REPORT ?= build/reference-report.json
CXX ?= c++
CXXFLAGS ?=
ENGINE_CXXFLAGS := -O3 -std=c++17 -fno-exceptions -fno-rtti \
	-Wall -Wextra -Wpedantic -march=native
THIRD_PARTY_FLAGS := -DSPDLOG_COMPILED_LIB -DSPDLOG_NO_EXCEPTIONS \
	-I$(ENGINE_DIR)/third_party/spdlog/include
SPDLOG_SRC := $(wildcard $(ENGINE_DIR)/third_party/spdlog/src/*.cpp)

.PHONY: all sync dump test cpu cuda compare compare-cuda serve clean

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

compare: $(LIBRARY)
	$(PYTHON) compare_engine.py \
		--library "$(LIBRARY)" \
		--bin "$(BIN)" \
		--vectors "$(VECTORS)" \
		--model "$(MODEL)" \
		--chat-template "$(CHAT_TEMPLATE)" \
		--report "$(REPORT)" \
		--cxxflags "$(CXXFLAGS)"

compare-cuda: $(CUDA_LIBRARY)
	$(PYTHON) compare_engine.py \
		--library "$(CUDA_LIBRARY)" \
		--bin "$(BIN)" \
		--vectors "$(VECTORS)" \
		--model "$(MODEL)" \
		--chat-template "$(CHAT_TEMPLATE)" \
		--report "build/cuda-reference-report.json" \
		--tolerance-key cuda_max_abs_error \
		--cxxflags "CUDA $(CXXFLAGS)"

$(LIBRARY): $(ENGINE_DIR)/engine.cpp $(ENGINE_DIR)/runtime.cpp \
		$(ENGINE_DIR)/log.cpp $(ENGINE_DIR)/qwen35.h \
		$(ENGINE_DIR)/internal.h $(SPDLOG_SRC)
	mkdir -p "$(dir $(LIBRARY))"
	$(CXX) $(ENGINE_CXXFLAGS) $(CXXFLAGS) $(THIRD_PARTY_FLAGS) \
		-fPIC -shared -I$(ENGINE_DIR) $(ENGINE_DIR)/engine.cpp \
		$(ENGINE_DIR)/runtime.cpp $(ENGINE_DIR)/log.cpp $(SPDLOG_SRC) \
		-pthread -o "$(LIBRARY)"

$(CUDA_LIBRARY): $(ENGINE_DIR)/arch/cuda/engine.cu $(ENGINE_DIR)/runtime.cpp \
		$(ENGINE_DIR)/log.cpp $(ENGINE_DIR)/qwen35.h \
		$(ENGINE_DIR)/internal.h $(SPDLOG_SRC)
	mkdir -p "$(dir $(CUDA_LIBRARY))"
	nvcc -O3 -std=c++17 -arch=native \
		-Xcompiler=-fPIC,-fno-exceptions,-fno-rtti,-Wall,-Wextra \
		$(THIRD_PARTY_FLAGS) -shared -I$(ENGINE_DIR) \
		$(ENGINE_DIR)/arch/cuda/engine.cu $(ENGINE_DIR)/runtime.cpp \
		$(ENGINE_DIR)/log.cpp $(SPDLOG_SRC) -Xcompiler=-pthread \
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

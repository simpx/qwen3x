CXX ?= c++
NVCC ?= nvcc
BUILD ?= build
PROGRAM ?= $(BUILD)/qwen35
CUDA_PROGRAM ?= $(BUILD)/qwen35-cuda
METAL_PROGRAM := $(BUILD)/qwen35-metal
METAL_CXX ?= clang++
METAL_DIR := $(BUILD)/metal
METAL_OBJ := $(BUILD)/obj/arch/metal/engine.o
METAL_FLAGS := -fobjc-arc -mmacosx-version-min=13.3
METAL_FRAMEWORKS := -framework Foundation -framework Metal
GPU_BACKEND := $(if $(filter Darwin,$(shell uname -s)),metal,cuda)
GPU_PROGRAM := $(if $(filter metal,$(GPU_BACKEND)),$(METAL_PROGRAM),$(CUDA_PROGRAM))
BUN ?= bun
NATIVE_FLAGS := $(if $(filter arm64 aarch64,$(shell uname -m)),-mcpu=native,-march=native)
PLATFORM_FLAGS := $(if $(filter Darwin,$(shell uname -s)),-mmacosx-version-min=13.3,)

CXXFLAGS ?= -O3 -std=c++17 -fno-exceptions -fno-rtti \
	-Wall -Wextra -Wpedantic $(NATIVE_FLAGS) $(PLATFORM_FLAGS)
THIRD_PARTY_FLAGS := -DSPDLOG_COMPILED_LIB -DSPDLOG_NO_EXCEPTIONS \
	-Ithird_party/spdlog/include
THREAD_FLAGS := -pthread
SPDLOG_SRC := $(wildcard third_party/spdlog/src/*.cpp)
COMMON_SRC := main.cpp runtime.cpp log.cpp parser.cpp render.cpp \
	$(SPDLOG_SRC)
COMMON_OBJ := $(patsubst %.cpp,$(BUILD)/obj/%.o,$(COMMON_SRC))
PROGRAM_OBJ := $(BUILD)/obj/engine.o $(COMMON_OBJ)
PROGRAM_DEP := $(PROGRAM_OBJ:.o=.d)
CUDA_OBJ := $(BUILD)/obj/arch/cuda/engine.o
CUDA_DEP := $(CUDA_OBJ:.o=.d)
CUDA_ARCH ?= native
CUDA_LIB_DIR ?= /usr/local/cuda/targets/x86_64-linux/lib
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) \
	-Xcompiler=-fno-exceptions,-fno-rtti,-Wall,-Wextra

.DELETE_ON_ERROR:

.PHONY: all cuda metal metal-shaders metal-test metal-library metal-reference \
	metal-smoke-vectors metal-smoke-9b-vectors metal-smoke metal-smoke-9b \
	model-4b model-9b serve-4b serve-9b \
	serve-eval-4b serve-eval-9b test cuda-test llama-smoke clean

all: $(PROGRAM)

.PHONY: q3x q3x-test
q3x:
	$(MAKE) -C agent build BUN="$(BUN)" BUILD="$(abspath $(BUILD))"

q3x-test:
	$(MAKE) -C agent test BUN="$(BUN)"

cuda: $(CUDA_PROGRAM)

metal: $(METAL_PROGRAM)

metal-shaders: $(METAL_DIR)/kernels.metallib

metal-test: $(BUILD)/metal-test
	$(BUILD)/metal-test

metal-library: $(METAL_DIR)/libqwen35-metal.dylib

metal-reference: metal-library
	$(MAKE) -C reference compare-metal METAL_LIBRARY="$(abspath $(METAL_DIR)/libqwen35-metal.dylib)"

metal-smoke-vectors:
	$(MAKE) -C reference build/libqwen35.so
	python3 tests/backend_smoke.py dump --library reference/build/libqwen35.so \
		--model $(BUILD)/qwen35-0.8b-model.bin --vectors $(BUILD)/metal-smoke-0.8b

metal-smoke-9b-vectors:
	$(MAKE) -C reference build/libqwen35.so
	python3 tests/backend_smoke.py dump --library reference/build/libqwen35.so \
		--model $(BUILD)/qwen35-9b-q8_0-model.bin --vectors $(BUILD)/metal-smoke-9b

metal-smoke: metal-library
	python3 tests/backend_smoke.py check --library $(METAL_DIR)/libqwen35-metal.dylib \
		--model $(BUILD)/qwen35-0.8b-model.bin --vectors $(BUILD)/metal-smoke-0.8b

metal-smoke-9b: metal-library
	python3 tests/backend_smoke.py check --library $(METAL_DIR)/libqwen35-metal.dylib \
		--model $(BUILD)/qwen35-9b-q8_0-model.bin --vectors $(BUILD)/metal-smoke-9b

model-4b:
	$(MAKE) -C scripts model-4b render

model-9b:
	$(MAKE) -C scripts model-9b render

serve-4b: $(GPU_BACKEND)
	test -f "$(BUILD)/qwen35-4b-model.bin" || { echo "run: make model-4b"; exit 1; }
	test -f "$(BUILD)/qwen35-0.8b-render.bin" || { echo "run: make model-4b"; exit 1; }
	$(GPU_PROGRAM) --model "$(BUILD)/qwen35-4b-model.bin" \
		--render "$(BUILD)/qwen35-0.8b-render.bin" --listen \
		--host 127.0.0.1 --port 8000 --session-slots 1 \
		--session-context 40960 --audit-log "$(BUILD)/qwen35-audit.log"

serve-9b: $(GPU_BACKEND)
	test -f "$(BUILD)/qwen35-9b-q8_0-model.bin" || { echo "run: make model-9b"; exit 1; }
	test -f "$(BUILD)/qwen35-0.8b-render.bin" || { echo "run: make model-9b"; exit 1; }
	$(GPU_PROGRAM) --model "$(BUILD)/qwen35-9b-q8_0-model.bin" \
		--render "$(BUILD)/qwen35-0.8b-render.bin" --listen \
		--host 127.0.0.1 --port 8000 --session-slots 1 \
		--session-context 40960 --audit-log "$(BUILD)/qwen35-9b-audit.log"

serve-eval-4b: $(GPU_BACKEND)
	test -f "$(BUILD)/qwen35-4b-model.bin" || { echo "run: make model-4b"; exit 1; }
	test -f "$(BUILD)/qwen35-0.8b-render.bin" || { echo "run: make model-4b"; exit 1; }
	$(GPU_PROGRAM) --model "$(BUILD)/qwen35-4b-model.bin" \
		--render "$(BUILD)/qwen35-0.8b-render.bin" --listen \
		--host 127.0.0.1 --port 8000 --session-slots 1 \
		--session-context 65536 --audit-log "$(BUILD)/qwen35-4b-eval-audit.log"

serve-eval-9b: $(GPU_BACKEND)
	test -f "$(BUILD)/qwen35-9b-q8_0-model.bin" || { echo "run: make model-9b"; exit 1; }
	test -f "$(BUILD)/qwen35-0.8b-render.bin" || { echo "run: make model-9b"; exit 1; }
	$(GPU_PROGRAM) --model "$(BUILD)/qwen35-9b-q8_0-model.bin" \
		--render "$(BUILD)/qwen35-0.8b-render.bin" --listen \
		--host 127.0.0.1 --port 8000 --session-slots 1 \
		--session-context 65536 --audit-log "$(BUILD)/qwen35-9b-eval-audit.log"

$(PROGRAM): $(PROGRAM_OBJ)
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $^ $(THREAD_FLAGS) -o $@

$(CUDA_PROGRAM): $(CUDA_OBJ) $(COMMON_OBJ)
	mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) $^ -L$(CUDA_LIB_DIR) -lcublas \
		-Xlinker -rpath -Xlinker $(CUDA_LIB_DIR) -Xcompiler=-pthread -o $@

$(CUDA_OBJ): arch/cuda/engine.cu internal.h model_config.h q8.h qwen35.h Makefile
	mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -I. -MMD -MP -c $< -o $@

# Shader bytes are embedded: the executable needs no external .metallib file.
$(METAL_DIR)/kernels.metallib: arch/metal/kernels.metal scripts/compile_metal.py Makefile
	python3 scripts/compile_metal.py --output $@

$(METAL_DIR)/kernels_metallib.h: $(METAL_DIR)/kernels.metallib
	cd $(METAL_DIR) && xxd -i kernels.metallib > kernels_metallib.h

$(METAL_OBJ): arch/metal/engine.mm internal.h model_config.h q8.h qwen35.h \
		$(METAL_DIR)/kernels_metallib.h Makefile
	mkdir -p $(dir $@)
	$(METAL_CXX) $(CXXFLAGS) $(METAL_FLAGS) -I. -I$(METAL_DIR) -MMD -MP -c $< -o $@

$(METAL_PROGRAM): $(METAL_OBJ) $(COMMON_OBJ)
	$(METAL_CXX) $(CXXFLAGS) $^ $(THREAD_FLAGS) $(METAL_FRAMEWORKS) -o $@

$(BUILD)/obj/tests/metal_test.o: tests/metal_test.mm engine.cpp arch/metal/engine.mm \
		internal.h model_config.h q8.h qwen35.h $(METAL_DIR)/kernels_metallib.h Makefile
	mkdir -p $(dir $@)
	$(METAL_CXX) $(CXXFLAGS) $(METAL_FLAGS) -I. -I$(METAL_DIR) -MMD -MP -c $< -o $@

$(BUILD)/metal-test: $(BUILD)/obj/tests/metal_test.o $(BUILD)/obj/log.o \
		$(patsubst %.cpp,$(BUILD)/obj/%.o,$(SPDLOG_SRC))
	$(METAL_CXX) $(CXXFLAGS) $^ $(THREAD_FLAGS) $(METAL_FRAMEWORKS) -o $@

$(METAL_DIR)/libqwen35-metal.dylib: $(METAL_OBJ) $(BUILD)/obj/runtime.o $(BUILD)/obj/log.o \
		$(patsubst %.cpp,$(BUILD)/obj/%.o,$(SPDLOG_SRC))
	$(METAL_CXX) $(CXXFLAGS) -dynamiclib $^ $(THREAD_FLAGS) $(METAL_FRAMEWORKS) -o $@

$(BUILD)/obj/%.o: %.cpp Makefile
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(THIRD_PARTY_FLAGS) -I. $(THREAD_FLAGS) \
		-MMD -MP -c $< -o $@

-include $(PROGRAM_DEP) $(CUDA_DEP) $(METAL_OBJ:.o=.d) $(BUILD)/obj/tests/metal_test.d

test: all
	$(MAKE) -C tests test

cuda-test: cuda
	$(MAKE) -C reference compare-cuda PYTHON=python3 VECTORS=build/cpu

llama-smoke:
	$(MAKE) -C reference llama-smoke

clean:
	rm -f "$(PROGRAM)" "$(CUDA_PROGRAM)" $(PROGRAM_OBJ) $(PROGRAM_DEP) \
		$(CUDA_OBJ) $(CUDA_DEP) "$(METAL_PROGRAM)" $(METAL_OBJ) $(METAL_OBJ:.o=.d) \
		$(BUILD)/metal-test $(BUILD)/obj/tests/metal_test.o $(BUILD)/obj/tests/metal_test.d \
		$(METAL_DIR)/kernels.air $(METAL_DIR)/kernels.metallib $(METAL_DIR)/kernels_metallib.h \
		$(METAL_DIR)/libqwen35-metal.dylib
	$(MAKE) -C scripts clean
	$(MAKE) -C tests clean

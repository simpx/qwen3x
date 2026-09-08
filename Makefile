CXX ?= c++
NVCC ?= nvcc
CPU_OPT ?= 1
BUILD ?= build$(if $(filter 0,$(CPU_OPT)),/scalar)
PROGRAM ?= $(BUILD)/qwen3x
CUDA_PROGRAM ?= $(BUILD)/qwen3x-cuda
METAL_PROGRAM := $(BUILD)/qwen3x-metal
METAL_CXX ?= clang++
METAL_DIR := $(BUILD)/metal
METAL_OBJ := $(BUILD)/obj/arch/metal/engine.o
METAL_FLAGS := -fobjc-arc
METAL_FRAMEWORKS := -framework Foundation -framework Metal
GPU_BACKEND := $(if $(filter Darwin,$(shell uname -s)),metal,cuda)
GPU_PROGRAM := $(if $(filter metal,$(GPU_BACKEND)),$(METAL_PROGRAM),$(CUDA_PROGRAM))
BUN ?= bun
NATIVE_FLAGS := $(if $(filter arm64 aarch64,$(shell uname -m)),-mcpu=native,-march=native)
PLATFORM_FLAGS := $(if $(filter Darwin,$(shell uname -s)),-mmacosx-version-min=26.3,)

CXXFLAGS ?= -O3 -std=c++17 -fno-exceptions -fno-rtti \
	-Wall -Wextra -Wpedantic $(NATIVE_FLAGS) $(PLATFORM_FLAGS)
THIRD_PARTY_FLAGS := -DSPDLOG_COMPILED_LIB -DSPDLOG_NO_EXCEPTIONS \
	-Ithird_party/spdlog/include
THREAD_FLAGS := -pthread
SPDLOG_SRC := $(wildcard third_party/spdlog/src/*.cpp)
COMMON_SRC := main.cpp runtime.cpp log.cpp parser.cpp render.cpp \
	$(SPDLOG_SRC)
COMMON_OBJ := $(patsubst %.cpp,$(BUILD)/obj/%.o,$(COMMON_SRC))
CPU_OBJ := $(BUILD)/obj/arch/cpu.o
PROGRAM_OBJ := $(BUILD)/obj/engine.o $(CPU_OBJ) $(COMMON_OBJ)
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
	model-4b model-9b model-27b model-35b cpu-35b-smoke cuda-35b-smoke \
	serve-4b serve-9b serve-27b \
	serve-eval-4b serve-eval-9b test cuda-test llama-smoke clean

all: $(PROGRAM)

# Optional MLX dependency; the default CPU build does not need it.
CMAKE ?= $(if $(wildcard $(BUILD)/mlx-venv/bin/cmake),$(abspath $(BUILD)/mlx-venv/bin/cmake),cmake)
MLX_SOURCE := $(BUILD)/mlx-src
MLX_DEP_BUILD := $(BUILD)/mlx-static-build
MLX_ROOT ?= $(abspath $(BUILD)/mlx-install)
MLX_REVISION := 1f8e74e3f12f31365464a6867c6579f0e9b29d85
MLX_BUILD_JOBS ?= 6

.PHONY: mlx-deps
mlx-deps:
	@test "$$(uname -s)" = Darwin && test "$$(uname -m)" = arm64 || \
		{ echo 'MLX requires Apple Silicon macOS.' >&2; exit 1; }
	xcrun --toolchain Metal -sdk macosx metal --version
	@if [ ! -d "$(MLX_SOURCE)/.git" ]; then \
		mkdir -p "$(dir $(MLX_SOURCE))"; \
		git clone --depth 1 --branch v0.32.2 https://github.com/ml-explore/mlx.git "$(MLX_SOURCE)"; \
	fi
	@test "$$(git -C "$(MLX_SOURCE)" rev-parse HEAD)" = "$(MLX_REVISION)" || \
		{ echo 'MLX source revision differs from the tested revision; leaving it untouched.' >&2; exit 1; }
	TOOLCHAINS=Metal "$(CMAKE)" -S "$(MLX_SOURCE)" -B "$(MLX_DEP_BUILD)" \
		-DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=26.3 \
		-DCMAKE_INSTALL_PREFIX="$(MLX_ROOT)" -DBUILD_SHARED_LIBS=OFF \
		-DMLX_METAL_JIT=OFF -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF \
		-DMLX_BUILD_BENCHMARKS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF -DMLX_BUILD_GGUF=OFF
	TOOLCHAINS=Metal "$(CMAKE)" --build "$(MLX_DEP_BUILD)" -j "$(MLX_BUILD_JOBS)"
	"$(CMAKE)" --install "$(MLX_DEP_BUILD)"

# MLX is an optional C++20 backend; shared product sources remain C++17.
MLX_LIBS := $(MLX_ROOT)/lib/libmlx.a $(MLX_ROOT)/lib/libjaccl.a
MLX_FRAMEWORKS := -framework Metal -framework Foundation -framework QuartzCore -framework Accelerate
MLX_OBJ := $(BUILD)/obj/arch/mlx/engine.o
MLX_FLAGS := $(filter-out -std=c++17 -fno-exceptions,$(CXXFLAGS)) -std=c++20 -fexceptions
.PHONY: mlx mlx-library mlx-test
mlx: $(BUILD)/qwen3x-mlx
mlx-library: $(BUILD)/mlx/libqwen3x-mlx.dylib
mlx-test: $(BUILD)/mlx-test
	$(BUILD)/mlx-test
	python3 tests/test_mlx_errors.py $(BUILD)/mlx-test
	python3 tests/test_pack_mlx.py

$(BUILD)/mlx-test: tests/mlx_test.cpp arch/mlx/engine.cpp internal.h model_config.h \
		$(BUILD)/obj/log.o $(patsubst %.cpp,$(BUILD)/obj/%.o,$(SPDLOG_SRC)) $(MLX_LIBS)
	$(CXX) $(MLX_FLAGS) -I. -I$(MLX_ROOT)/include \
		-DQ3X_MLX_METALLIB='"$(MLX_ROOT)/lib/mlx.metallib"' $< \
		$(filter %.o %.a,$^) $(THREAD_FLAGS) $(MLX_FRAMEWORKS) -o $@

$(MLX_OBJ): arch/mlx/engine.cpp internal.h model_config.h Makefile
	mkdir -p $(dir $@)
	$(CXX) $(MLX_FLAGS) -I. -I$(MLX_ROOT)/include \
		-DQ3X_MLX_METALLIB='"$(MLX_ROOT)/lib/mlx.metallib"' -MMD -MP -c $< -o $@

$(BUILD)/qwen3x-mlx: $(MLX_OBJ) $(COMMON_OBJ) $(MLX_LIBS)
	$(CXX) $(CXXFLAGS) $^ $(THREAD_FLAGS) $(MLX_FRAMEWORKS) -o $@

$(BUILD)/mlx/libqwen3x-mlx.dylib: $(MLX_OBJ) $(BUILD)/obj/runtime.o $(BUILD)/obj/log.o \
		$(patsubst %.cpp,$(BUILD)/obj/%.o,$(SPDLOG_SRC)) $(MLX_LIBS)
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -dynamiclib $^ $(THREAD_FLAGS) $(MLX_FRAMEWORKS) -o $@

-include $(MLX_OBJ:.o=.d)

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

metal-library: $(METAL_DIR)/libqwen3x-metal.dylib

metal-reference: metal-library
	$(MAKE) -C reference compare-metal METAL_LIBRARY="$(abspath $(METAL_DIR)/libqwen3x-metal.dylib)"

metal-smoke-vectors:
	$(MAKE) -C reference build/libqwen3x.so
	python3 tests/backend_smoke.py dump --library reference/build/libqwen3x.so \
		--model $(BUILD)/qwen35-0.8b-model.bin --vectors $(BUILD)/metal-smoke-0.8b

metal-smoke-9b-vectors:
	$(MAKE) -C reference build/libqwen3x.so
	python3 tests/backend_smoke.py dump --library reference/build/libqwen3x.so \
		--model $(BUILD)/qwen35-9b-q8_0-model.bin --vectors $(BUILD)/metal-smoke-9b

metal-smoke: metal-library
	python3 tests/backend_smoke.py check --library $(METAL_DIR)/libqwen3x-metal.dylib \
		--model $(BUILD)/qwen35-0.8b-model.bin --vectors $(BUILD)/metal-smoke-0.8b

metal-smoke-9b: metal-library
	python3 tests/backend_smoke.py check --library $(METAL_DIR)/libqwen3x-metal.dylib \
		--model $(BUILD)/qwen35-9b-q8_0-model.bin --vectors $(BUILD)/metal-smoke-9b

model-4b:
	$(MAKE) -C scripts model-4b render

model-9b:
	$(MAKE) -C scripts model-9b render

model-27b:
	$(MAKE) -C scripts model-27b render

model-35b:
	$(MAKE) -C scripts model-35b render

cpu-35b-smoke: all model-35b
	$(MAKE) -C reference build/libqwen3x.so
	mkdir -p "$(BUILD)/qwen36-35b-cpu-smoke"
	/usr/bin/time -v -o "$(BUILD)/qwen36-35b-cpu-smoke/time.txt" \
		python3 tests/qwen36_smoke.py dump \
		--library reference/build/libqwen3x.so \
		--model "$(BUILD)/qwen36-35b-a3b-q4_0-model.bin" \
		--vectors "$(BUILD)/qwen36-35b-cpu-smoke" \
		--handoff "$(BUILD)/qwen36-35b-metal-smoke"
	$(PROGRAM) --model "$(BUILD)/qwen36-35b-a3b-q4_0-model.bin" \
		--render "$(BUILD)/qwen3x-render.bin" --prompt "Hello" \
		--session-context 128 --max-tokens 4

cuda-35b-smoke: cuda model-35b
	$(MAKE) -C reference build/libqwen3x-cuda.so
	mkdir -p "$(BUILD)/qwen36-35b-cuda-smoke"
	/usr/bin/time -v -o "$(BUILD)/qwen36-35b-cuda-smoke/time.txt" \
		python3 tests/qwen36_smoke.py check \
		--library reference/build/libqwen3x-cuda.so \
		--model "$(BUILD)/qwen36-35b-a3b-q4_0-model.bin" \
		--vectors "$(BUILD)/qwen36-35b-cpu-smoke" \
		--output "$(BUILD)/qwen36-35b-cuda-smoke" --context 8192
	$(CUDA_PROGRAM) --model "$(BUILD)/qwen36-35b-a3b-q4_0-model.bin" \
		--render "$(BUILD)/qwen3x-render.bin" --prompt "Hello" \
		--session-context 8192 --max-tokens 4 --log-level info

serve-4b: $(GPU_BACKEND)
	test -f "$(BUILD)/qwen35-4b-model.bin" || { echo "run: make model-4b"; exit 1; }
	test -f "$(BUILD)/qwen3x-render.bin" || { echo "run: make model-4b"; exit 1; }
	$(GPU_PROGRAM) --model "$(BUILD)/qwen35-4b-model.bin" \
		--render "$(BUILD)/qwen3x-render.bin" --listen \
		--host 127.0.0.1 --port 8000 --session-slots 1 \
		--session-context 40960 --audit-log "$(BUILD)/qwen3x-audit.log"

serve-9b: $(GPU_BACKEND)
	test -f "$(BUILD)/qwen35-9b-q8_0-model.bin" || { echo "run: make model-9b"; exit 1; }
	test -f "$(BUILD)/qwen3x-render.bin" || { echo "run: make model-9b"; exit 1; }
	$(GPU_PROGRAM) --model "$(BUILD)/qwen35-9b-q8_0-model.bin" \
		--render "$(BUILD)/qwen3x-render.bin" --listen \
		--host 127.0.0.1 --port 8000 --session-slots 1 \
		--session-context 40960 --audit-log "$(BUILD)/qwen35-9b-audit.log"

serve-27b: metal
	test -f "$(BUILD)/qwen38-27b-q4_0-model.bin" || { echo "run: make model-27b"; exit 1; }
	test -f "$(BUILD)/qwen3x-render.bin" || { echo "run: make model-27b"; exit 1; }
	$(METAL_PROGRAM) --model "$(BUILD)/qwen38-27b-q4_0-model.bin" \
		--render "$(BUILD)/qwen3x-render.bin" --listen \
		--host 127.0.0.1 --port 8000 --session-slots 1 \
		--session-context 32768 --audit-log "$(BUILD)/qwen38-27b-audit.log"

serve-eval-4b: $(GPU_BACKEND)
	test -f "$(BUILD)/qwen35-4b-model.bin" || { echo "run: make model-4b"; exit 1; }
	test -f "$(BUILD)/qwen3x-render.bin" || { echo "run: make model-4b"; exit 1; }
	$(GPU_PROGRAM) --model "$(BUILD)/qwen35-4b-model.bin" \
		--render "$(BUILD)/qwen3x-render.bin" --listen \
		--host 127.0.0.1 --port 8000 --session-slots 1 \
		--session-context 65536 \
		--audit-log "$(BUILD)/qwen35-4b-eval-audit.log"

serve-eval-9b: $(GPU_BACKEND)
	test -f "$(BUILD)/qwen35-9b-q8_0-model.bin" || { echo "run: make model-9b"; exit 1; }
	test -f "$(BUILD)/qwen3x-render.bin" || { echo "run: make model-9b"; exit 1; }
	$(GPU_PROGRAM) --model "$(BUILD)/qwen35-9b-q8_0-model.bin" \
		--render "$(BUILD)/qwen3x-render.bin" --listen \
		--host 127.0.0.1 --port 8000 --session-slots 1 \
		--session-context 65536 \
		--audit-log "$(BUILD)/qwen35-9b-eval-audit.log"

$(PROGRAM): $(PROGRAM_OBJ)
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $^ $(THREAD_FLAGS) -o $@

$(CUDA_PROGRAM): $(CUDA_OBJ) $(COMMON_OBJ)
	mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) $^ -L$(CUDA_LIB_DIR) -lcublas \
		-Xlinker -rpath -Xlinker $(CUDA_LIB_DIR) -Xcompiler=-pthread -o $@

$(CUDA_OBJ): arch/cuda/engine.cu internal.h model_config.h q4.h q8.h qwen3x.h Makefile
	mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -I. -MMD -MP -c $< -o $@

# Shader bytes are embedded: the executable needs no external .metallib file.
$(METAL_DIR)/kernels.metallib: arch/metal/kernels.metal scripts/compile_metal.py Makefile
	python3 scripts/compile_metal.py --output $@

$(METAL_DIR)/kernels_metallib.h: $(METAL_DIR)/kernels.metallib
	cd $(METAL_DIR) && xxd -i kernels.metallib > kernels_metallib.h

$(METAL_OBJ): arch/metal/engine.mm internal.h model_config.h q4.h q8.h qwen3x.h \
		$(METAL_DIR)/kernels_metallib.h Makefile
	mkdir -p $(dir $@)
	$(METAL_CXX) $(CXXFLAGS) $(METAL_FLAGS) -I. -I$(METAL_DIR) -MMD -MP -c $< -o $@

$(METAL_PROGRAM): $(METAL_OBJ) $(COMMON_OBJ)
	$(METAL_CXX) $(CXXFLAGS) $^ $(THREAD_FLAGS) $(METAL_FRAMEWORKS) -o $@

$(BUILD)/obj/tests/metal_test.o: tests/metal_test.mm engine.cpp arch/metal/engine.mm \
		internal.h model_config.h q4.h q8.h qwen3x.h $(METAL_DIR)/kernels_metallib.h Makefile
	mkdir -p $(dir $@)
	$(METAL_CXX) $(CXXFLAGS) $(METAL_FLAGS) -I. -I$(METAL_DIR) -MMD -MP -c $< -o $@

$(BUILD)/metal-test: $(BUILD)/obj/tests/metal_test.o $(CPU_OBJ) $(BUILD)/obj/log.o \
		$(patsubst %.cpp,$(BUILD)/obj/%.o,$(SPDLOG_SRC))
	$(METAL_CXX) $(CXXFLAGS) $^ $(THREAD_FLAGS) $(METAL_FRAMEWORKS) -o $@

$(METAL_DIR)/libqwen3x-metal.dylib: $(METAL_OBJ) $(BUILD)/obj/runtime.o $(BUILD)/obj/log.o \
		$(patsubst %.cpp,$(BUILD)/obj/%.o,$(SPDLOG_SRC))
	$(METAL_CXX) $(CXXFLAGS) -dynamiclib $^ $(THREAD_FLAGS) $(METAL_FRAMEWORKS) -o $@

$(CPU_OBJ): arch/cpu.cpp Makefile
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -DQ3X_CPU_OPT=$(CPU_OPT) -I. $(THREAD_FLAGS) \
		-MMD -MP -c $< -o $@

$(BUILD)/obj/%.o: %.cpp Makefile
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(THIRD_PARTY_FLAGS) -I. $(THREAD_FLAGS) \
		-MMD -MP -c $< -o $@

-include $(PROGRAM_DEP) $(CUDA_DEP) $(METAL_OBJ:.o=.d) $(BUILD)/obj/tests/metal_test.d

test: all
	$(MAKE) -C tests test BUILD="$(abspath $(BUILD))" CPU_OPT=$(CPU_OPT)

cuda-test: cuda
	$(MAKE) -C reference compare-cuda PYTHON=python3 VECTORS=build/cpu

llama-smoke:
	$(MAKE) -C reference llama-smoke

clean:
	rm -f "$(PROGRAM)" "$(CUDA_PROGRAM)" $(PROGRAM_OBJ) $(PROGRAM_DEP) \
		$(CUDA_OBJ) $(CUDA_DEP) "$(METAL_PROGRAM)" $(METAL_OBJ) $(METAL_OBJ:.o=.d) \
		$(BUILD)/metal-test $(BUILD)/obj/tests/metal_test.o $(BUILD)/obj/tests/metal_test.d \
		$(BUILD)/qwen3x-mlx $(BUILD)/mlx-test $(BUILD)/mlx/libqwen3x-mlx.dylib \
		$(MLX_OBJ) $(MLX_OBJ:.o=.d) \
		$(METAL_DIR)/kernels.air $(METAL_DIR)/kernels.metallib $(METAL_DIR)/kernels_metallib.h \
		$(METAL_DIR)/libqwen3x-metal.dylib
	$(MAKE) -C scripts clean
	$(MAKE) -C tests clean

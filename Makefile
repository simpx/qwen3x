CXX ?= c++
NVCC ?= nvcc
BUILD ?= build
PROGRAM ?= $(BUILD)/qwen35
CUDA_PROGRAM ?= $(BUILD)/qwen35-cuda

CXXFLAGS ?= -O3 -std=c++17 -fno-exceptions -fno-rtti \
	-Wall -Wextra -Wpedantic -march=native
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

.PHONY: all cuda model-4b serve-4b test cuda-test clean

all: $(PROGRAM)

cuda: $(CUDA_PROGRAM)

model-4b:
	$(MAKE) -C scripts model-4b render

serve-4b: cuda
	test -f "$(BUILD)/qwen35-4b-model.bin" || { echo "run: make model-4b"; exit 1; }
	test -f "$(BUILD)/qwen35-0.8b-render.bin" || { echo "run: make model-4b"; exit 1; }
	$(CUDA_PROGRAM) --model "$(BUILD)/qwen35-4b-model.bin" \
		--render "$(BUILD)/qwen35-0.8b-render.bin" --listen \
		--host 127.0.0.1 --port 8000 --session-slots 1 \
		--session-context 40960 --audit-log "$(BUILD)/qwen35-audit.log"

$(PROGRAM): $(PROGRAM_OBJ)
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $^ $(THREAD_FLAGS) -o $@

$(CUDA_PROGRAM): $(CUDA_OBJ) $(COMMON_OBJ)
	mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) $^ -L$(CUDA_LIB_DIR) -lcublas \
		-Xlinker -rpath -Xlinker $(CUDA_LIB_DIR) -Xcompiler=-pthread -o $@

$(CUDA_OBJ): arch/cuda/engine.cu internal.h model_config.h qwen35.h Makefile
	mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -I. -MMD -MP -c $< -o $@

$(BUILD)/obj/%.o: %.cpp Makefile
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(THIRD_PARTY_FLAGS) -I. $(THREAD_FLAGS) \
		-MMD -MP -c $< -o $@

-include $(PROGRAM_DEP) $(CUDA_DEP)

test: all
	$(MAKE) -C tests test

cuda-test: cuda
	$(MAKE) -C reference compare-cuda PYTHON=python3 VECTORS=build/cpu

clean:
	rm -f "$(PROGRAM)" "$(CUDA_PROGRAM)" $(PROGRAM_OBJ) $(PROGRAM_DEP) \
		$(CUDA_OBJ) $(CUDA_DEP)
	$(MAKE) -C scripts clean
	$(MAKE) -C tests clean

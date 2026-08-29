PROJECT := $(abspath ..)
BUILD ?= $(PROJECT)/build
UV ?= uv
HF ?= $(UV) run --project "$(CURDIR)" --locked hf
PYTHON ?= $(UV) run --project "$(CURDIR)" --locked python
CXX ?= c++

CHECKPOINT_REPO ?= Qwen/Qwen3.5-0.8B
CHECKPOINT ?= $(BUILD)/models/Qwen3.5-0.8B
CHECKPOINT_STAMP ?= $(BUILD)/.qwen3.5-0.8b-checkpoint
MODEL_BIN ?= $(BUILD)/qwen35-0.8b.bin
RENDER_BIN ?= $(BUILD)/qwen35-render.bin
BENCH ?= $(BUILD)/bench

CXXFLAGS ?= -O3 -std=c++17 -fno-exceptions -fno-rtti \
	-Wall -Wextra -Wpedantic -march=native
THIRD_PARTY_FLAGS := -DSPDLOG_COMPILED_LIB -DSPDLOG_NO_EXCEPTIONS \
	-I$(PROJECT)/third_party/spdlog/include
SPDLOG_SRC := $(wildcard $(PROJECT)/third_party/spdlog/src/*.cpp)

.PHONY: all sync checkpoint model render bench clean

all: model render

sync:
	$(UV) sync --project "$(CURDIR)" --locked

checkpoint: $(CHECKPOINT_STAMP)

$(CHECKPOINT_STAMP):
	mkdir -p "$(CHECKPOINT)"
	$(HF) download "$(CHECKPOINT_REPO)" --local-dir "$(CHECKPOINT)"
	rm -rf "$(CHECKPOINT)/.cache"
	touch "$@"

model: $(MODEL_BIN)

$(MODEL_BIN): pack_weights.py $(CHECKPOINT_STAMP)
	mkdir -p "$(BUILD)"
	cd "$(PROJECT)" && $(PYTHON) -m scripts.pack_weights "$(CHECKPOINT)" "$@"

render: $(RENDER_BIN)

$(RENDER_BIN): pack_render.py $(CHECKPOINT_STAMP) \
		$(CHECKPOINT)/tokenizer.json $(CHECKPOINT)/tokenizer_config.json
	mkdir -p "$(BUILD)"
	cd "$(PROJECT)" && $(PYTHON) -m scripts.pack_render "$(CHECKPOINT)" "$@"

$(BENCH): bench.cpp $(PROJECT)/engine.cpp $(PROJECT)/runtime.cpp \
		$(PROJECT)/log.cpp $(PROJECT)/qwen35.h $(PROJECT)/internal.h $(SPDLOG_SRC)
	mkdir -p "$(BUILD)"
	$(CXX) $(CXXFLAGS) $(THIRD_PARTY_FLAGS) -I$(PROJECT) \
		$(PROJECT)/engine.cpp $(PROJECT)/runtime.cpp $(PROJECT)/log.cpp \
		bench.cpp $(SPDLOG_SRC) -pthread -o "$@"

bench: $(BENCH) $(MODEL_BIN)
	$(BENCH) "$(MODEL_BIN)" 8 4

clean:
	rm -f "$(BENCH)" "$(RENDER_BIN)"

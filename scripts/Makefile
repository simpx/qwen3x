PROJECT := $(abspath ..)
BUILD ?= $(PROJECT)/build
UV ?= uv
HF ?= $(UV) run --project "$(CURDIR)" --locked hf
PYTHON ?= $(UV) run --project "$(CURDIR)" --locked python

CHECKPOINT_REPO ?= Qwen/Qwen3.5-0.8B
CHECKPOINT ?= $(BUILD)/models/Qwen3.5-0.8B
CHECKPOINT_STAMP ?= $(BUILD)/.qwen3.5-0.8b-checkpoint
MODEL_BIN ?= $(BUILD)/qwen35-0.8b-model.bin
CHECKPOINT_4B_REPO ?= Qwen/Qwen3.5-4B
CHECKPOINT_4B ?= $(BUILD)/models/Qwen3.5-4B
CHECKPOINT_4B_STAMP ?= $(BUILD)/.qwen3.5-4b-checkpoint
MODEL_4B_BIN ?= $(BUILD)/qwen35-4b-model.bin
RENDER_BIN ?= $(BUILD)/qwen35-0.8b-render.bin
.PHONY: all sync checkpoint checkpoint-4b model model-4b render clean

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

checkpoint-4b: $(CHECKPOINT_4B_STAMP)

$(CHECKPOINT_4B_STAMP):
	mkdir -p "$(CHECKPOINT_4B)"
	$(HF) download "$(CHECKPOINT_4B_REPO)" --local-dir "$(CHECKPOINT_4B)"
	rm -rf "$(CHECKPOINT_4B)/.cache"
	touch "$@"

model-4b: $(MODEL_4B_BIN)

$(MODEL_4B_BIN): pack_weights.py $(CHECKPOINT_4B_STAMP)
	mkdir -p "$(BUILD)"
	cd "$(PROJECT)" && $(PYTHON) -m scripts.pack_weights "$(CHECKPOINT_4B)" "$@"

render: $(RENDER_BIN)

$(RENDER_BIN): pack_render.py $(CHECKPOINT_STAMP) \
		$(CHECKPOINT)/tokenizer.json $(CHECKPOINT)/tokenizer_config.json
	mkdir -p "$(BUILD)"
	cd "$(PROJECT)" && $(PYTHON) -m scripts.pack_render "$(CHECKPOINT)" "$@"

clean:
	rm -f "$(RENDER_BIN)"

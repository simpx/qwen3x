PROJECT := $(abspath ..)
BUILD ?= $(PROJECT)/build
UV ?= uv
HF ?= $(UV) run --project "$(CURDIR)" --locked hf
PYTHON ?= $(UV) run --project "$(CURDIR)" --locked python

CHECKPOINT_REPO ?= Qwen/Qwen3.5-0.8B
CHECKPOINT ?= $(BUILD)/models/Qwen3.5-0.8B
CHECKPOINT_STAMP ?= $(BUILD)/.qwen3.5-0.8b-checkpoint
MODEL_BIN ?= $(BUILD)/qwen35-0.8b-model.bin
RENDER_BIN ?= $(BUILD)/qwen35-0.8b-render.bin
.PHONY: all sync checkpoint model render clean

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

clean:
	rm -f "$(RENDER_BIN)"

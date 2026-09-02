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
CHECKPOINT_9B_REPO ?= Qwen/Qwen3.5-9B
CHECKPOINT_9B_REVISION ?= c202236235762e1c871ad0ccb60c8ee5ba337b9a
CHECKPOINT_9B ?= $(BUILD)/models/Qwen3.5-9B
CHECKPOINT_9B_STAMP ?= $(BUILD)/.qwen3.5-9b-checkpoint
MODEL_9B_BIN ?= $(BUILD)/qwen35-9b-q8_0-model.bin
RENDER_BIN ?= $(BUILD)/qwen35-0.8b-render.bin
.PHONY: all sync checkpoint checkpoint-4b checkpoint-9b model model-4b model-9b render clean

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

checkpoint-9b: $(CHECKPOINT_9B_STAMP)

$(CHECKPOINT_9B_STAMP):
	mkdir -p "$(CHECKPOINT_9B)"
	$(HF) download "$(CHECKPOINT_9B_REPO)" --revision "$(CHECKPOINT_9B_REVISION)" \
		--local-dir "$(CHECKPOINT_9B)"
	rm -rf "$(CHECKPOINT_9B)/.cache"
	touch "$@"

model-9b: $(MODEL_9B_BIN)

$(MODEL_9B_BIN): pack_weights.py $(CHECKPOINT_9B_STAMP)
	mkdir -p "$(BUILD)"
	cd "$(PROJECT)" && $(PYTHON) -m scripts.pack_weights "$(CHECKPOINT_9B)" "$@"

render: $(RENDER_BIN)

$(RENDER_BIN): pack_render.py $(CHECKPOINT_STAMP) \
		$(CHECKPOINT)/tokenizer.json $(CHECKPOINT)/tokenizer_config.json
	mkdir -p "$(BUILD)"
	cd "$(PROJECT)" && $(PYTHON) -m scripts.pack_render "$(CHECKPOINT)" "$@"

clean:
	rm -f "$(RENDER_BIN)"

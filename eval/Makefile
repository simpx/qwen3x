UV ?= uv
EVAL_DIR := $(CURDIR)
PROJECT ?= ..
PROJECT := $(abspath $(PROJECT))
SERVER ?= http://127.0.0.1:8000
TOKENIZER ?= $(abspath ../build/models/Qwen3.5-0.8B)
BIN ?= build/qwen35-4b-model.bin

EVAL_DATASET ?= mmlu_pro
EVAL_MODEL ?= qwen3.5-4b
EVAL_BACKEND ?= qwen35
EVAL_ARTIFACT ?= $(BIN)
EVAL_LIMIT ?=
EVAL_THINKING ?= 0
EVAL_SMOKE ?= 0
EVAL_SEED ?= 42
EVAL_MAX_TOKENS ?= 32768
EVAL_REQUEST_TIMEOUT ?= 7200
EVAL_CONCURRENCY ?= 1
EVAL_OUTPUT ?= $(EVAL_DIR)/results
EVAL_RESUME ?=
AGENT_SCENE ?= all
AGENT_OUTPUT ?=

REFERENCE_SERVER ?= http://127.0.0.1:8002
ENGINE_RUN ?=
REFERENCE_RUN ?=

LIMIT_ARG := $(if $(EVAL_LIMIT),--samples "$(EVAL_LIMIT)",)
RESUME_ARG := $(if $(EVAL_RESUME),--resume "$(EVAL_RESUME)",)
THINKING_ARG := $(if $(filter 1 true yes,$(EVAL_THINKING)),--thinking,)
SMOKE_ARG := $(if $(filter 1 true yes,$(EVAL_SMOKE)),--smoke,)
PYTHON := $(UV) run --project "$(EVAL_DIR)" --locked python

.PHONY: test run smoke smoke-run smoke-4b smoke-9b smoke-report \
	reference compare agent-eval

test:
	cd "$(EVAL_DIR)" && $(PYTHON) -m unittest -v test_eval

run:
	cd "$(PROJECT)" && $(PYTHON) "$(EVAL_DIR)/run.py" \
		--server "$(SERVER)" \
		--model "$(EVAL_MODEL)" \
		--tokenizer "$(TOKENIZER)" \
		--backend "$(EVAL_BACKEND)" \
		--artifact "$(EVAL_ARTIFACT)" \
		--dataset "$(EVAL_DATASET)" \
		$(LIMIT_ARG) \
		$(RESUME_ARG) \
		$(THINKING_ARG) \
		$(SMOKE_ARG) \
		--seed "$(EVAL_SEED)" \
		--max-tokens "$(EVAL_MAX_TOKENS)" \
		--request-timeout "$(EVAL_REQUEST_TIMEOUT)" \
		--concurrency "$(EVAL_CONCURRENCY)" \
		--output "$(EVAL_OUTPUT)"

smoke:
	timeout --signal=TERM --kill-after=15s 30m $(MAKE) smoke-run

smoke-run:
	for dataset in mmlu_pro ceval ifeval; do \
		$(MAKE) run EVAL_DATASET="$$dataset" EVAL_LIMIT=2 EVAL_SMOKE=1 \
			EVAL_THINKING=1 EVAL_SEED=42 EVAL_MAX_TOKENS=32768 || exit $$?; \
	done
	$(MAKE) smoke-report

smoke-4b:
	$(MAKE) smoke EVAL_MODEL=qwen3.5-4b \
		TOKENIZER="$(PROJECT)/build/models/Qwen3.5-4B" \
		EVAL_ARTIFACT="$(PROJECT)/build/qwen35-4b-model.bin" \
		EVAL_OUTPUT="$(EVAL_DIR)/results/smoke/4b"

smoke-9b:
	$(MAKE) smoke EVAL_MODEL=qwen3.5-9b \
		TOKENIZER="$(PROJECT)/build/models/Qwen3.5-4B" \
		EVAL_ARTIFACT="$(PROJECT)/build/qwen35-9b-q8_0-model.bin" \
		EVAL_OUTPUT="$(EVAL_DIR)/results/smoke/9b"

smoke-report:
	cd "$(PROJECT)" && $(PYTHON) "$(EVAL_DIR)/smoke_report.py"

reference:
	$(MAKE) run SERVER="$(REFERENCE_SERVER)" \
		EVAL_MODEL=qwen3.5-0.8b-reference \
		EVAL_BACKEND=transformers \
		EVAL_ARTIFACT="$(TOKENIZER)/model.safetensors-00001-of-00001.safetensors"

compare:
	@test -n "$(ENGINE_RUN)" || { echo 'set ENGINE_RUN'; exit 2; }
	@test -n "$(REFERENCE_RUN)" || { echo 'set REFERENCE_RUN'; exit 2; }
	@cd "$(PROJECT)" && $(PYTHON) "$(EVAL_DIR)/compare.py" \
		--engine "$(ENGINE_RUN)" --reference "$(REFERENCE_RUN)"

agent-eval:
	cd "$(PROJECT)" && python3 scripts/agent_eval.py "$(AGENT_SCENE)" \
		--url "$(SERVER)/v1/chat/completions" \
		--model "$(EVAL_MODEL)" \
		$(if $(AGENT_OUTPUT),--output "$(AGENT_OUTPUT)",)

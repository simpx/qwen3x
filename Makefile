CXX ?= c++
UV ?= uv
PYTHON ?= $(UV) run --locked python
PYTHON_PROD ?= $(UV) run --locked --no-dev python
TOKENIZER ?= models/Qwen3.5-0.8B
BUILD ?= build
PROGRAM ?= $(BUILD)/qwen35
BIN ?= $(BUILD)/qwen35-0.8b.bin
PARSER_TEST ?= $(BUILD)/parser-test
RENDER_BIN ?= $(BUILD)/qwen35-render.bin
RENDER_DRIVER ?= $(BUILD)/render-test
SOCKET ?= /tmp/qwen35.sock
SERVER ?= http://127.0.0.1:8000
SLOTS ?= 2
CONTEXT ?= 40960
MOCK ?= 0
MOCK_ARG := $(if $(filter 1 true yes,$(MOCK)),--mock,)
REFERENCE ?= reference/build/cpu
REFERENCE_DTYPE ?= float32
REFERENCE_CACHE ?= static

CXXFLAGS ?= -O3 -std=c++17 -fno-exceptions -fno-rtti \
	-Wall -Wextra -Wpedantic -march=native

.PHONY: all sync sync-prod weights render-data parser-test render-test unit-test eval-test reference-generate test run chat eval eval-smoke reference-serve eval-reference eval-compare clean

all: $(PROGRAM)

sync:
	$(UV) sync --locked

sync-prod:
	$(UV) sync --locked --no-dev

$(PROGRAM): main.cpp engine.cpp runtime.cpp log.cpp qwen35.h internal.h
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) main.cpp engine.cpp runtime.cpp log.cpp -pthread -o $@

weights: $(BIN)

$(BIN): scripts/pack_weights.py
	mkdir -p $(BUILD)
	$(PYTHON_PROD) -m scripts.pack_weights "$(TOKENIZER)" "$@"

render-data: $(RENDER_BIN)

$(RENDER_BIN): scripts/pack_render.py $(TOKENIZER)/tokenizer.json $(TOKENIZER)/tokenizer_config.json
	mkdir -p $(BUILD)
	$(PYTHON) -m scripts.pack_render "$(TOKENIZER)" "$@"

$(PARSER_TEST): parser.cpp render.h tests/parser_test.cpp third_party/nlohmann/json.hpp
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -I. parser.cpp tests/parser_test.cpp -o $@

parser-test: $(PARSER_TEST)
	$(PARSER_TEST)

$(RENDER_DRIVER): render.cpp parser.cpp render.h tests/render_driver.cpp third_party/nlohmann/json.hpp
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -I. render.cpp parser.cpp tests/render_driver.cpp -o $@

render-test: $(RENDER_BIN) $(RENDER_DRIVER)
	TOKENIZER="$(abspath $(TOKENIZER))" \
	RENDER_BIN="$(abspath $(RENDER_BIN))" \
	RENDER_TEST="$(abspath $(RENDER_DRIVER))" \
	$(PYTHON) -m unittest -v tests.test_render

unit-test:
	$(PYTHON) -m unittest -v tests.test_server tests.test_client

eval-test:
	$(MAKE) -C eval test PROJECT="$(CURDIR)"

reference-generate:
	$(MAKE) -C reference dump \
		MODEL="$(abspath $(TOKENIZER))" \
		OUT="$(abspath $(REFERENCE))" \
		DEVICE=cpu \
		CHAT_TEMPLATE="$(abspath chat_template.jinja)"

test: unit-test parser-test

run: $(PROGRAM) $(BIN)
	$(PROGRAM) -l "$(SOCKET)" -m "$(BIN)" --parallel "$(SLOTS)" --context "$(CONTEXT)" $(MOCK_ARG)

chat:
	curl --silent --show-error --no-buffer "$(SERVER)/v1/chat/completions" \
		-H 'Content-Type: application/json' \
		-d '{"model":"qwen3.5-0.8b","messages":[{"role":"user","content":"你好，请用一句话介绍自己。"}],"temperature":0,"max_completion_tokens":1024,"stream":true,"stream_options":{"include_usage":true}}'

eval: $(BIN)
	$(MAKE) -C eval run PROJECT="$(CURDIR)" SERVER="$(SERVER)" \
		TOKENIZER="$(abspath $(TOKENIZER))" BIN="$(abspath $(BIN))"

eval-smoke:
	$(MAKE) -C eval smoke PROJECT="$(CURDIR)" SERVER="$(SERVER)" \
		TOKENIZER="$(abspath $(TOKENIZER))" BIN="$(abspath $(BIN))"

reference-serve:
	$(MAKE) -C reference serve \
		MODEL="$(abspath $(TOKENIZER))" \
		CHAT_TEMPLATE="$(abspath chat_template.jinja)" \
		DEVICE=cuda DTYPE="$(REFERENCE_DTYPE)" CACHE="$(REFERENCE_CACHE)" \
		MAX_CONTEXT="$(CONTEXT)" PORT=8002

eval-reference:
	$(MAKE) -C eval reference PROJECT="$(CURDIR)" \
		TOKENIZER="$(abspath $(TOKENIZER))"

eval-compare:
	$(MAKE) -C eval compare PROJECT="$(CURDIR)"

clean:
	rm -rf $(BUILD)

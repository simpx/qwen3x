CXX ?= c++
UV ?= uv
SYSTEM_PYTHON ?= python3
PYTHON ?= $(UV) run --project scripts --locked python
PYTHON_PROD ?= $(UV) run --project scripts --locked --no-dev python
TOKENIZER ?= models/Qwen3.5-0.8B
BUILD ?= build
PROGRAM ?= $(BUILD)/qwen35
LIBRARY ?= $(BUILD)/libqwen35.so
BIN ?= $(BUILD)/qwen35-0.8b.bin
PARSER_TEST ?= $(BUILD)/parser-test
RUNTIME_TEST ?= $(BUILD)/runtime-test
BENCH ?= $(BUILD)/bench
RENDER_BIN ?= $(BUILD)/qwen35-render.bin
RENDER_DRIVER ?= $(BUILD)/render-test
SERVER ?= http://127.0.0.1:8000
HOST ?= 127.0.0.1
PORT ?= 8000
SLOTS ?= 2
CONTEXT ?= 40960
MOCK ?= 0
MOCK_ARG := $(if $(filter 1 true yes,$(MOCK)),--mock,)
REFERENCE ?= reference/build/cpu
REFERENCE_DTYPE ?= float32
REFERENCE_CACHE ?= static

CXXFLAGS ?= -O3 -std=c++17 -fno-exceptions -fno-rtti \
	-Wall -Wextra -Wpedantic -march=native
THIRD_PARTY_FLAGS := -DSPDLOG_COMPILED_LIB -DSPDLOG_NO_EXCEPTIONS \
	-Ithird_party/spdlog/include
SPDLOG_SRC := $(wildcard third_party/spdlog/src/*.cpp)
PROGRAM_SRC := main.cpp engine.cpp runtime.cpp log.cpp parser.cpp render.cpp \
	$(SPDLOG_SRC)

.PHONY: all sync sync-prod weights render-data reference-library parser-test runtime-test render-test http-test unit-test eval-test reference-generate test run chat bench eval eval-smoke reference-serve eval-reference eval-compare clean

all: $(PROGRAM)

sync:
	$(UV) sync --project scripts --locked

sync-prod:
	$(UV) sync --project scripts --locked --no-dev

$(PROGRAM): $(PROGRAM_SRC) qwen35.h internal.h render.h \
	third_party/httplib/httplib.h third_party/nlohmann/json.hpp
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(THIRD_PARTY_FLAGS) -I. $(PROGRAM_SRC) -pthread -o $@

$(LIBRARY): engine.cpp runtime.cpp log.cpp qwen35.h internal.h $(SPDLOG_SRC)
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(THIRD_PARTY_FLAGS) -fPIC -shared -I. \
		engine.cpp runtime.cpp log.cpp $(SPDLOG_SRC) -pthread -o $@

reference-library: $(LIBRARY)

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

$(RUNTIME_TEST): engine.cpp runtime.cpp log.cpp qwen35.h internal.h \
		tests/runtime_test.cpp $(SPDLOG_SRC)
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(THIRD_PARTY_FLAGS) -I. \
		engine.cpp runtime.cpp log.cpp tests/runtime_test.cpp $(SPDLOG_SRC) \
		-pthread -o $@

runtime-test: $(RUNTIME_TEST)
	$(RUNTIME_TEST)

$(BENCH): engine.cpp runtime.cpp log.cpp qwen35.h internal.h \
		scripts/bench.cpp $(SPDLOG_SRC)
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(THIRD_PARTY_FLAGS) -I. \
		engine.cpp runtime.cpp log.cpp scripts/bench.cpp $(SPDLOG_SRC) \
		-pthread -o $@

bench: $(BENCH) $(BIN)
	$(BENCH) "$(BIN)" 8 4

$(RENDER_DRIVER): render.cpp parser.cpp render.h tests/render_driver.cpp third_party/nlohmann/json.hpp
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -I. render.cpp parser.cpp tests/render_driver.cpp -o $@

render-test: $(RENDER_BIN) $(RENDER_DRIVER)
	TOKENIZER="$(abspath $(TOKENIZER))" \
	RENDER_BIN="$(abspath $(RENDER_BIN))" \
	RENDER_TEST="$(abspath $(RENDER_DRIVER))" \
	$(PYTHON) -m unittest -v tests.test_render

http-test: $(PROGRAM) $(RENDER_BIN)
	$(SYSTEM_PYTHON) tests/test_http.py \
		--program "$(abspath $(PROGRAM))" --render "$(abspath $(RENDER_BIN))"

unit-test: parser-test runtime-test

eval-test:
	$(MAKE) -C eval test PROJECT="$(CURDIR)"

reference-generate:
	$(MAKE) -C reference dump \
		MODEL="$(abspath $(TOKENIZER))" \
		OUT="$(abspath $(REFERENCE))" \
		DEVICE=cpu \
		CHAT_TEMPLATE="$(abspath reference/chat_template.jinja)"

test: unit-test

run: $(PROGRAM) $(BIN) $(RENDER_BIN)
	$(PROGRAM) -m "$(BIN)" -r "$(RENDER_BIN)" \
		--host "$(HOST)" --port "$(PORT)" \
		--slots "$(SLOTS)" --context "$(CONTEXT)" $(MOCK_ARG)

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
		CHAT_TEMPLATE="$(abspath reference/chat_template.jinja)" \
		DEVICE=cuda DTYPE="$(REFERENCE_DTYPE)" CACHE="$(REFERENCE_CACHE)" \
		MAX_CONTEXT="$(CONTEXT)" PORT=8002

eval-reference:
	$(MAKE) -C eval reference PROJECT="$(CURDIR)" \
		TOKENIZER="$(abspath $(TOKENIZER))"

eval-compare:
	$(MAKE) -C eval compare PROJECT="$(CURDIR)"

clean:
	rm -rf $(BUILD)

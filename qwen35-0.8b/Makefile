CXX ?= c++
UV ?= uv
PYTHON ?= $(UV) run --locked python
PYTHON_PROD ?= $(UV) run --locked --no-dev python
TOKENIZER ?= ../models/Qwen3.5-0.8B
BUILD ?= build
LIBRARY ?= $(BUILD)/libqwen35.so
BIN ?= $(BUILD)/qwen35-0.8b.bin
SERVER ?= http://127.0.0.1:8000
LOG_LEVEL ?= debug
MOCK ?= 0
MOCK_ARG := $(if $(filter 1 true yes,$(MOCK)),--mock,)
BENCH_PREFILL ?= 8
BENCH_DECODE ?= 4
REFERENCE ?= ../reference/build/cpu
REFERENCE_REPORT ?= $(BUILD)/reference-report.json

CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic -march=native

.PHONY: all sync sync-prod weights unit-test mock-test native-test reference-generate reference-test reference test e2e run chat bench clean

all: $(LIBRARY)

sync:
	$(UV) sync --locked

sync-prod:
	$(UV) sync --locked --no-dev

$(LIBRARY): engine.cpp runtime.cpp log.cpp qwen35.h internal.h
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -fPIC -shared engine.cpp runtime.cpp log.cpp -o $@

weights: $(BIN)

$(BIN): pack_weights.py
	mkdir -p $(BUILD)
	$(PYTHON_PROD) pack_weights.py "$(TOKENIZER)" "$@"

unit-test:
	$(PYTHON) -m unittest -v tests.test_server tests.test_client

native-test: $(LIBRARY) $(BIN)
	PYTHONPATH=. $(PYTHON) tests/test_native.py --library "$(LIBRARY)" --bin "$(BIN)"

mock-test: $(LIBRARY) $(BIN)
	PYTHONPATH=. $(PYTHON) tests/test_mock.py --library "$(LIBRARY)" --bin "$(BIN)"

reference-generate:
	$(MAKE) -C ../reference dump \
		MODEL="$(abspath $(TOKENIZER))" \
		OUT="$(abspath $(REFERENCE))" \
		DEVICE=cpu \
		CHAT_TEMPLATE="$(abspath chat_template.jinja)"

reference-test: $(LIBRARY) $(BIN)
	$(MAKE) -C ../reference compare \
		ENGINE_DIR="$(CURDIR)" \
		LIBRARY="$(abspath $(LIBRARY))" \
		BIN="$(abspath $(BIN))" \
		VECTORS="$(abspath $(REFERENCE))" \
		MODEL="$(abspath $(TOKENIZER))" \
		CHAT_TEMPLATE="$(abspath chat_template.jinja)" \
		REPORT="$(abspath $(REFERENCE_REPORT))" \
		CXXFLAGS="$(CXXFLAGS)"

reference: reference-generate reference-test

test: unit-test mock-test native-test

e2e: $(LIBRARY) $(BIN)
	PYTHONPATH=. $(PYTHON) tests/test_e2e.py --tokenizer "$(TOKENIZER)" --library "$(LIBRARY)" --bin "$(BIN)"

run: $(LIBRARY) $(BIN)
	$(PYTHON_PROD) server.py --tokenizer "$(TOKENIZER)" --library "$(LIBRARY)" --bin "$(BIN)" --log-level "$(LOG_LEVEL)" $(MOCK_ARG)

chat:
	curl --silent --show-error --no-buffer "$(SERVER)/v1/chat/completions" \
		-H 'Content-Type: application/json' \
		-d '{"model":"qwen3.5-0.8b","messages":[{"role":"user","content":"你好，请用一句话介绍自己。"}],"temperature":0,"max_completion_tokens":1024,"stream":true,"stream_options":{"include_usage":true}}'

bench: $(LIBRARY) $(BIN)
	PYTHONPATH=. $(PYTHON_PROD) bench.py \
		--library "$(LIBRARY)" \
		--bin "$(BIN)" \
		--prefill-tokens "$(BENCH_PREFILL)" \
		--decode-tokens "$(BENCH_DECODE)"

clean:
	rm -rf $(BUILD)

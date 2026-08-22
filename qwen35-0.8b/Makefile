CXX ?= c++
UV ?= uv
PYTHON ?= $(UV) run --locked python
PYTHON_PROD ?= $(UV) run --locked --no-dev python
MODEL ?= ../models/Qwen3.5-0.8B
BUILD ?= build
LIBRARY ?= $(BUILD)/libqwen35.so
WEIGHTS ?= $(BUILD)/qwen35-0.8b.bin
SERVER ?= http://127.0.0.1:8000

CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic

.PHONY: all sync sync-prod weights unit-test native-test test e2e run chat clean

all: $(LIBRARY)

sync:
	$(UV) sync --locked

sync-prod:
	$(UV) sync --locked --no-dev

$(LIBRARY): engine.cpp runtime.cpp log.cpp qwen35.h internal.h
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -fPIC -shared engine.cpp runtime.cpp log.cpp -o $@

weights: $(WEIGHTS)

$(WEIGHTS): pack_weights.py
	mkdir -p $(BUILD)
	$(PYTHON_PROD) pack_weights.py "$(MODEL)" "$@"

unit-test:
	$(PYTHON) -m unittest discover -v -s tests -p 'test_server.py'

native-test: $(LIBRARY) $(WEIGHTS)
	PYTHONPATH=. $(PYTHON) tests/test_native.py --library "$(LIBRARY)" --weights "$(WEIGHTS)"

test: unit-test native-test

e2e: $(LIBRARY) $(WEIGHTS)
	PYTHONPATH=. $(PYTHON) tests/test_e2e.py --model "$(MODEL)" --library "$(LIBRARY)" --weights "$(WEIGHTS)"

run: $(LIBRARY) $(WEIGHTS)
	$(PYTHON_PROD) server.py --model "$(MODEL)" --library "$(LIBRARY)" --weights "$(WEIGHTS)"

chat:
	curl --silent --show-error --no-buffer "$(SERVER)/v1/chat/completions" \
		-H 'Content-Type: application/json' \
		-d '{"model":"qwen3.5-0.8b","session_id":"make-chat","messages":[{"role":"user","content":"你好，请用一句话介绍自己。"}],"temperature":0,"max_completion_tokens":1024,"stream":true}'

clean:
	rm -rf $(BUILD)

CXX ?= c++
BUILD ?= build
PROGRAM ?= $(BUILD)/qwen35
MODEL_BIN ?= $(BUILD)/qwen35-0.8b.bin
RENDER_BIN ?= $(BUILD)/qwen35-render.bin
SERVER ?= http://127.0.0.1:8000
HOST ?= 127.0.0.1
PORT ?= 8000
SLOTS ?= 2
CONTEXT ?= 40960
MOCK ?= 0
MOCK_ARG := $(if $(filter 1 true yes,$(MOCK)),--mock,)

CXXFLAGS ?= -O3 -std=c++17 -fno-exceptions -fno-rtti \
	-Wall -Wextra -Wpedantic -march=native
THIRD_PARTY_FLAGS := -DSPDLOG_COMPILED_LIB -DSPDLOG_NO_EXCEPTIONS \
	-Ithird_party/spdlog/include
SPDLOG_SRC := $(wildcard third_party/spdlog/src/*.cpp)
PROGRAM_SRC := main.cpp engine.cpp runtime.cpp log.cpp parser.cpp render.cpp \
	$(SPDLOG_SRC)

.PHONY: all test run chat clean

all: $(PROGRAM)

$(PROGRAM): $(PROGRAM_SRC) qwen35.h internal.h render.h \
	third_party/httplib/httplib.h third_party/nlohmann/json.hpp
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(THIRD_PARTY_FLAGS) -I. $(PROGRAM_SRC) -pthread -o $@

test: all
	$(MAKE) -C tests test

run: $(PROGRAM)
	$(PROGRAM) -m "$(MODEL_BIN)" -r "$(RENDER_BIN)" \
		--host "$(HOST)" --port "$(PORT)" \
		--slots "$(SLOTS)" --context "$(CONTEXT)" $(MOCK_ARG)

chat:
	curl --silent --show-error --no-buffer "$(SERVER)/v1/chat/completions" \
		-H 'Content-Type: application/json' \
		-d '{"model":"qwen3.5-0.8b","messages":[{"role":"user","content":"你好，请用一句话介绍自己。"}],"temperature":0,"max_completion_tokens":1024,"stream":true,"stream_options":{"include_usage":true}}'

clean:
	rm -f "$(PROGRAM)"
	$(MAKE) -C scripts clean
	$(MAKE) -C tests clean

PROJECT := $(abspath ..)
BUILD ?= $(PROJECT)/build
CXX ?= c++
UV ?= uv
PYTHON ?= $(UV) run --project "$(PROJECT)/scripts" --locked python
SYSTEM_PYTHON ?= python3

PROGRAM ?= $(BUILD)/qwen35
CHECKPOINT ?= $(BUILD)/models/Qwen3.5-0.8B
RENDER_BIN ?= $(BUILD)/qwen35-0.8b-render.bin
PARSER_TEST ?= $(BUILD)/parser-test
RUNTIME_TEST ?= $(BUILD)/runtime-test
RENDER_DRIVER ?= $(BUILD)/render-test

CXXFLAGS ?= -O3 -std=c++17 -fno-exceptions -fno-rtti \
	-Wall -Wextra -Wpedantic -march=native
THIRD_PARTY_FLAGS := -DSPDLOG_COMPILED_LIB -DSPDLOG_NO_EXCEPTIONS \
	-I$(PROJECT)/third_party/spdlog/include
SPDLOG_SRC := $(wildcard $(PROJECT)/third_party/spdlog/src/*.cpp)

.PHONY: test parser-test runtime-test cli-test chat-test agent-test render-test http-test clean

test: parser-test runtime-test cli-test chat-test agent-test

$(PARSER_TEST): parser_test.cpp $(PROJECT)/parser.cpp $(PROJECT)/render.cpp \
		$(PROJECT)/render.h \
		$(PROJECT)/third_party/nlohmann/json.hpp
	mkdir -p "$(BUILD)"
	$(CXX) $(CXXFLAGS) -I$(PROJECT) \
		$(PROJECT)/parser.cpp $(PROJECT)/render.cpp parser_test.cpp -o "$@"

parser-test: $(PARSER_TEST)
	$(PARSER_TEST)

$(RUNTIME_TEST): runtime_test.cpp $(PROJECT)/engine.cpp $(PROJECT)/runtime.cpp \
		$(PROJECT)/log.cpp $(PROJECT)/qwen35.h $(PROJECT)/internal.h $(SPDLOG_SRC)
	mkdir -p "$(BUILD)"
	$(CXX) $(CXXFLAGS) $(THIRD_PARTY_FLAGS) -I$(PROJECT) \
		$(PROJECT)/engine.cpp $(PROJECT)/runtime.cpp $(PROJECT)/log.cpp \
		runtime_test.cpp $(SPDLOG_SRC) -pthread -o "$@"

runtime-test: $(RUNTIME_TEST)
	$(RUNTIME_TEST)

cli-test: $(PROGRAM)
	@output=`$(PROGRAM) --mock --bench 8 4` && \
		echo "$$output" | grep -q '^prefill  8 tokens' && \
		echo "$$output" | grep -q '^decode   4 tokens' && \
		echo "cli-test: ok"

chat-test:
	cd "$(PROJECT)" && $(SYSTEM_PYTHON) -m unittest discover \
		-s tests -p test_chat.py -v

agent-test:
	cd "$(PROJECT)" && $(SYSTEM_PYTHON) -m unittest discover \
		-s tests -p test_agent.py -v

$(RENDER_DRIVER): render_driver.cpp $(PROJECT)/render.cpp $(PROJECT)/parser.cpp \
		$(PROJECT)/render.h $(PROJECT)/third_party/nlohmann/json.hpp
	mkdir -p "$(BUILD)"
	$(CXX) $(CXXFLAGS) -I$(PROJECT) $(PROJECT)/render.cpp \
		$(PROJECT)/parser.cpp render_driver.cpp -o "$@"

$(RENDER_BIN):
	$(MAKE) -C "$(PROJECT)/scripts" render

render-test: $(RENDER_BIN) $(RENDER_DRIVER)
	cd "$(PROJECT)" && \
		TOKENIZER="$(CHECKPOINT)" RENDER_BIN="$(RENDER_BIN)" \
		RENDER_TEST="$(RENDER_DRIVER)" \
		$(PYTHON) -m unittest -v tests.test_render

$(PROGRAM):
	$(MAKE) -C "$(PROJECT)"

http-test: $(PROGRAM) $(RENDER_BIN)
	cd "$(PROJECT)" && $(SYSTEM_PYTHON) tests/test_http.py \
		--program "$(PROGRAM)"

clean:
	rm -f "$(PARSER_TEST)" "$(RUNTIME_TEST)" "$(RENDER_DRIVER)"

CXX ?= c++
BUILD ?= build
PROGRAM ?= $(BUILD)/qwen35

CXXFLAGS ?= -O3 -std=c++17 -fno-exceptions -fno-rtti \
	-Wall -Wextra -Wpedantic -march=native
THIRD_PARTY_FLAGS := -DSPDLOG_COMPILED_LIB -DSPDLOG_NO_EXCEPTIONS \
	-Ithird_party/spdlog/include
SPDLOG_SRC := $(wildcard third_party/spdlog/src/*.cpp)
PROGRAM_SRC := main.cpp engine.cpp runtime.cpp log.cpp parser.cpp render.cpp \
	$(SPDLOG_SRC)

.PHONY: all test clean

all: $(PROGRAM)

$(PROGRAM): $(PROGRAM_SRC) qwen35.h internal.h render.h \
	third_party/httplib/httplib.h third_party/nlohmann/json.hpp
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(THIRD_PARTY_FLAGS) -I. $(PROGRAM_SRC) -pthread -o $@

test: all
	$(MAKE) -C tests test

clean:
	rm -f "$(PROGRAM)"
	$(MAKE) -C scripts clean
	$(MAKE) -C tests clean

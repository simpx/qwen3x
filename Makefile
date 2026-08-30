CXX ?= c++
BUILD ?= build
PROGRAM ?= $(BUILD)/qwen35

CXXFLAGS ?= -O3 -std=c++17 -fno-exceptions -fno-rtti \
	-Wall -Wextra -Wpedantic -march=native
THIRD_PARTY_FLAGS := -DSPDLOG_COMPILED_LIB -DSPDLOG_NO_EXCEPTIONS \
	-Ithird_party/spdlog/include
THREAD_FLAGS := -pthread
SPDLOG_SRC := $(wildcard third_party/spdlog/src/*.cpp)
PROGRAM_SRC := main.cpp engine.cpp runtime.cpp log.cpp parser.cpp render.cpp \
	$(SPDLOG_SRC)
PROGRAM_OBJ := $(patsubst %.cpp,$(BUILD)/obj/%.o,$(PROGRAM_SRC))
PROGRAM_DEP := $(PROGRAM_OBJ:.o=.d)

.DELETE_ON_ERROR:

.PHONY: all test clean

all: $(PROGRAM)

$(PROGRAM): $(PROGRAM_OBJ)
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $^ $(THREAD_FLAGS) -o $@

$(BUILD)/obj/%.o: %.cpp Makefile
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(THIRD_PARTY_FLAGS) -I. $(THREAD_FLAGS) \
		-MMD -MP -c $< -o $@

-include $(PROGRAM_DEP)

test: all
	$(MAKE) -C tests test

clean:
	rm -f "$(PROGRAM)" $(PROGRAM_OBJ) $(PROGRAM_DEP)
	$(MAKE) -C scripts clean
	$(MAKE) -C tests clean

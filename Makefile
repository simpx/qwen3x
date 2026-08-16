# 课程只有两类命令：逐课运行，或运行最终的真实 0.8B capstone。
CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic

LESSON_SOURCES := $(sort $(wildcard lessons/*.cpp))
LESSON_BINS := $(LESSON_SOURCES:.cpp=)

.PHONY: all test lesson-test course-test course-oracle-test clean

all: test

lessons/%: lessons/%.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

lesson-test: $(LESSON_BINS)
	@for lesson in $(LESSON_BINS); do ./$$lesson; done

qwen38: capstone/qwen38.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

course-test: qwen38
	./qwen38 --self-test

test: lesson-test course-test

course-oracle-test: qwen38
	@test -n "$(MODEL)" || (echo "usage: make course-oracle-test MODEL=/path/to/Qwen3.5-0.8B" >&2; exit 2)
	scripts/test_course_08b.sh "$(MODEL)"

clean:
	rm -f qwen38 $(LESSON_BINS)

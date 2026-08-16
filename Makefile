# 课程只有两类命令：逐课运行，或运行最终的真实 0.8B capstone。
#
# `make test` 不会下载模型：它编译/运行每个玩具 lesson 和 capstone 的微型自检。
# 真正的官方权重回归需显式写 MODEL=...，以免 1.6 GiB checkpoint 成为隐式依赖。
CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic

LESSON_SOURCES := $(sort $(wildcard lessons/*.cpp))  # 00..08；新增课会自动进入测试。
LESSON_BINS := $(LESSON_SOURCES:.cpp=)               # 每课是独立的单文件可执行程序。

.PHONY: all test lesson-test course-test course-oracle-test clean

all: test  # 默认命令也是课程的快速 smoke test。

lessons/%: lessons/%.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# 顺序运行，输出正好是博客中可引用的每课中间数值。
lesson-test: $(LESSON_BINS)
	@for lesson in $(LESSON_BINS); do ./$$lesson; done

qwen38: capstone/qwen38.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# 不读真实 checkpoint；只验证 BF16 conversion、RMSNorm 和 argmax 的不变量。
course-test: qwen38
	./qwen38 --self-test

test: lesson-test course-test

# 此目标会临时转换 MODEL，不会把 1.4 GiB bin 留在仓库目录。
course-oracle-test: qwen38
	@test -n "$(MODEL)" || (echo "usage: make course-oracle-test MODEL=/path/to/Qwen3.5-0.8B" >&2; exit 2)
	scripts/test_course_08b.sh "$(MODEL)"

clean:
	rm -f qwen38 $(LESSON_BINS)

# 真实 Qwen3.5-0.8B 的独立小目录：`make` 只编译 qwen35。

.PHONY: all test oracle-test clean

all: qwen35

qwen35: qwen35.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic qwen35.cpp -o qwen35

test: qwen35
	./qwen35 --self-test

oracle-test: qwen35
	@test -n "$(MODEL)" || (echo "usage: make oracle-test MODEL=/path/to/Qwen3.5-0.8B" >&2; exit 2)
	./test.sh "$(MODEL)"

clean:
	rm -f qwen35

# 真实 Qwen3.5-0.8B 的独立小目录：`make` 只编译 qwen3x。

.PHONY: all test oracle-test clean

all: qwen3x

qwen3x: qwen3x.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic qwen3x.cpp -o qwen3x

test: qwen3x
	./qwen3x --self-test

oracle-test: qwen3x
	@test -n "$(MODEL)" || (echo "usage: make oracle-test MODEL=/path/to/Qwen3.5-0.8B" >&2; exit 2)
	./test.sh "$(MODEL)"

clean:
	rm -f qwen3x

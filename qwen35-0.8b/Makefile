# 真实 Qwen3.5-0.8B 的独立小目录：`make` 只编译这一个固定模型程序。

.PHONY: all test oracle-test clean

all: qwen38

qwen38: qwen38.cpp
	c++ -O3 -std=c++17 -Wall -Wextra -Wpedantic qwen38.cpp -o qwen38

test: qwen38
	./qwen38 --self-test

oracle-test: qwen38
	@test -n "$(MODEL)" || (echo "usage: make oracle-test MODEL=/path/to/Qwen3.5-0.8B" >&2; exit 2)
	./test.sh "$(MODEL)"

clean:
	rm -f qwen38

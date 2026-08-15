CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
NVCC ?= nvcc

.PHONY: test tiny-test dev-test oracle-test tokenizer-test tiny-cuda-test cuda-dev-test clean

qwen38: qwen38.cpp
	$(CXX) $(CXXFLAGS) qwen38.cpp -o $@

test: qwen38
	./qwen38 --self-test

qwen38_tiny: qwen38.cpp
	$(CXX) $(CXXFLAGS) -DQWEN38_TINY_MODEL=1 qwen38.cpp -o $@

tiny-test: qwen38_tiny
	./qwen38_tiny --self-test

# A real, fixed Qwen3.5-0.8B text build for local development.  This
# dependency-free Makefile target accepts token IDs; use CMake for tokenizer
# support and --generate-text.
qwen38_08b: qwen38.cpp
	$(CXX) $(CXXFLAGS) -DQWEN38_DEV_08B_MODEL=1 qwen38.cpp -o $@

dev-test: qwen38_08b
	./qwen38_08b --self-test

# Local real-weight regression. MODEL is intentionally required so CI and a
# fresh checkout never download a checkpoint implicitly.
oracle-test:
	@test -n "$(MODEL)" || (echo "usage: make oracle-test MODEL=/path/to/Qwen3.5-0.8B" >&2; exit 2)
	scripts/test_08b_oracle.sh "$(MODEL)"

tokenizer-test:
	@test -n "$(MODEL)" || (echo "usage: make tokenizer-test MODEL=/path/to/Qwen3.5-0.8B" >&2; exit 2)
	scripts/test_08b_tokenizer.sh "$(MODEL)"

qwen38_tiny_cuda: kernels/qwen38_tiny_cuda.cu qwen38.cpp
	$(NVCC) -O3 -std=c++17 kernels/qwen38_tiny_cuda.cu -o $@

tiny-cuda-test: qwen38_tiny_cuda
	@fixture=$$(mktemp -d); ./qwen38_tiny_cuda --make-fake "$$fixture"; ./qwen38_tiny_cuda --cuda-compare "$$fixture" 1,2,3,4,5,6,7,8

qwen38_08b_cuda: kernels/qwen38_08b_cuda.cu kernels/qwen38_tiny_cuda.cu qwen38.cpp
	$(NVCC) -O3 -std=c++17 kernels/qwen38_08b_cuda.cu -o $@

cuda-dev-test: qwen38_08b_cuda
	@test -n "$(MODEL)" || (echo "usage: make cuda-dev-test MODEL=/path/to/Qwen3.5-0.8B" >&2; exit 2)
	./qwen38_08b_cuda --cuda-compare "$(MODEL)" 248044,198,198

clean:
	rm -f qwen38 qwen38_tiny qwen38_tiny_cuda qwen38_08b qwen38_08b_cuda

// Fixed Qwen3.5-0.8B CUDA correctness build.
//
// Keep the implementation in qwen38_tiny_cuda.cu: it is intentionally a
// model-specific, direct CUDA forward, parameterized only at compile time by
// qwen38.cpp's fixed Config variants. This wrapper is not a backend framework.

#define QWEN38_CUDA_MODEL_DEFINE 1
#define QWEN38_DEV_08B_MODEL 1
#include "qwen38_tiny_cuda.cu"

#pragma once

#include "model_config.h"

namespace q3x_cpu {

// CPU extension entry points. No Model/State ownership or weight allocation.
// Synchronous: true means output is complete; false leaves output untouched and
// runs engine.cpp's BF16 baseline (or its FP32 dot). Unknown formats are rejected.
// CPU_OPT=0 skips architecture acceleration but retains scalar quantization.
bool try_embed(const void* weights, q3x_model::MatrixType type, int cols, int token, float* out);
bool try_mv(const void* weights, q3x_model::MatrixType type, int rows, int cols,
            const float* x, float* y);
bool try_dot(const float* a, const float* b, int n, float* out);

// Quantized payload size; BF16, unknown formats and invalid shapes return false
// without modifying bytes. The engine handles BF16 size, alignment and file bounds.
bool quantized_bytes(q3x_model::MatrixType type, int rows, int cols, size_t* bytes);

}  // namespace q3x_cpu

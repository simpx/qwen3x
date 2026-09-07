#pragma once

#include "model_config.h"

namespace q3x_cpu {

// Synchronous: true means output is complete; false leaves output untouched.
// No Model/State ownership here. Unsupported operations use engine.cpp's loops.
bool try_mv(const void* weights, q3x_model::MatrixType type, int rows, int cols,
            const float* x, float* y);
bool try_dot(const float* a, const float* b, int n, float* out);

}  // namespace q3x_cpu

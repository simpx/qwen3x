// CPU acceleration selection. All architecture/OS conditions stay here.
#include "cpu.h"

#ifndef Q3X_CPU_OPT
#define Q3X_CPU_OPT 1
#endif

#if Q3X_CPU_OPT
#include <cstring>
#include "q8.h"
#if defined(__aarch64__) && defined(__ARM_NEON)
#include "arm/kernels.h"
#endif
#if defined(__AVX512F__)
#include "x86/kernels.h"
#endif
#if defined(__APPLE__)
#include "apple/parallel.h"
#endif

namespace q3x_cpu {

// A small local row loop lets GCD also accelerate CPUs without SIMD kernels.
#if defined(__APPLE__) || (defined(__AVX512F__) && defined(__AVX512BW__))
static float dot_bf16(const uint16_t* w, const float* x, int n) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    return x86::dot_bf16(w, x, n);
#else
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        const uint32_t bits = static_cast<uint32_t>(w[i]) << 16;
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        sum += value * x[i];
    }
    return sum;
#endif
}
#endif

#if defined(__APPLE__) || (defined(__aarch64__) && defined(__ARM_NEON))
static float dot_q8(const q3x_q8::Block* w, const float* x, int n) {
#if defined(__aarch64__) && defined(__ARM_NEON)
    return arm::dot_q8(w, x, n);
#else
    // Intel Macs still use the GCD row path, with native half conversion.
    float sum = 0.0f;
    for (int block = 0; block < n / q3x_q8::BLOCK_SIZE; ++block) {
        _Float16 scale;
        std::memcpy(&scale, &w[block].scale, sizeof(scale));
        float inner = 0.0f;
        for (int i = 0; i < q3x_q8::BLOCK_SIZE; ++i)
            inner += w[block].values[i] * x[block * q3x_q8::BLOCK_SIZE + i];
        sum += static_cast<float>(scale) * inner;
    }
    return sum;
#endif
}
#endif

template <typename Row>
static void matrix_rows(int rows, const Row& row) {
#if defined(__APPLE__)
    if (rows >= 1024) { apple::parallel_rows(rows, row); return; }
#endif
    for (int i = 0; i < rows; ++i) row(i);
}

bool try_mv(const void* weights, q3x_model::MatrixType type, int rows, int cols,
            const float* x, float* y) {
#if defined(__APPLE__) || (defined(__AVX512F__) && defined(__AVX512BW__))
    if (type == q3x_model::MATRIX_BF16) {
#if !defined(__AVX512F__) || !defined(__AVX512BW__)
        if (rows < 1024) return false;
#endif
        const auto* w = static_cast<const uint16_t*>(weights);
        matrix_rows(rows, [&](int row) {
            y[row] = dot_bf16(w + static_cast<size_t>(row) * cols, x, cols);
        });
        return true;
    }
#endif
#if defined(__APPLE__) || (defined(__aarch64__) && defined(__ARM_NEON))
    if (type == q3x_model::MATRIX_Q8_0 && cols % q3x_q8::BLOCK_SIZE == 0) {
        const auto* w = static_cast<const q3x_q8::Block*>(weights);
        matrix_rows(rows, [&](int row) {
            y[row] = dot_q8(w + static_cast<size_t>(row) * (cols / q3x_q8::BLOCK_SIZE), x, cols);
        });
        return true;
    }
#endif
    (void)weights; (void)type; (void)rows; (void)cols; (void)x; (void)y;
    return false;
}

bool try_dot(const float* a, const float* b, int n, float* out) {
#if defined(__AVX512F__)
    *out = x86::dot_f32(a, b, n);
    return true;
#else
    (void)a; (void)b; (void)n; (void)out;
    return false;
#endif
}

}  // namespace q3x_cpu

#else
namespace q3x_cpu {
bool try_mv(const void*, q3x_model::MatrixType, int, int, const float*, float*) {
    return false;
}
bool try_dot(const float*, const float*, int, float*) { return false; }
}  // namespace q3x_cpu
#endif

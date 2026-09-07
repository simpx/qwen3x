// CPU extensions: architecture acceleration, then scalar quantized operations.
#include "cpu.h"
#include <cstring>
#include <limits>

#include "internal.h"
#include "q4.h"
#include "q8.h"

#ifndef Q3X_CPU_OPT
#define Q3X_CPU_OPT 1
#endif

#if Q3X_CPU_OPT
#if defined(__aarch64__) && defined(__ARM_NEON)
#include "arm/kernels.h"
#endif
#if defined(__AVX512F__)
#include "x86/kernels.h"
#endif
#if defined(__APPLE__)
#include "apple/parallel.h"
#endif

#endif  // Q3X_CPU_OPT

namespace q3x_cpu {
namespace {

// Scalar quantization is available in every CPU build.
float f16(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 113;
            while (!(mantissa & 0x0400u)) { mantissa <<= 1; --exponent; }
            bits = sign | (exponent << 23) | ((mantissa & 0x03ffu) << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

// Q8_0: one FP16 scale and 32 signed 8-bit values per block.
void embed_q8_0(const q3x_q8::Block* weights, int cols, int token, float* out) {
    const int blocks = cols / q3x_q8::BLOCK_SIZE;
    const q3x_q8::Block* row = weights + static_cast<size_t>(token) * blocks;
    for (int block = 0; block < blocks; ++block) {
        const float scale = f16(row[block].scale);
        for (int i = 0; i < q3x_q8::BLOCK_SIZE; ++i)
            out[block * q3x_q8::BLOCK_SIZE + i] = scale * row[block].values[i];
    }
}
float dot_q8_0(const q3x_q8::Block* blocks, const float* x, int n) {
    float sum = 0.0f;
    for (int block = 0; block < n / q3x_q8::BLOCK_SIZE; ++block) {
        float inner = 0.0f;
        for (int i = 0; i < q3x_q8::BLOCK_SIZE; ++i)
            inner += blocks[block].values[i] * x[block * q3x_q8::BLOCK_SIZE + i];
        sum += f16(blocks[block].scale) * inner;
    }
    return sum;
}
void mv_q8_0(const q3x_q8::Block* weights, int rows, int cols, const float* x, float* y) {
    const int blocks = cols / q3x_q8::BLOCK_SIZE;
    for (int row = 0; row < rows; ++row)
        y[row] = dot_q8_0(weights + static_cast<size_t>(row) * blocks, x, cols);
}

// Q4_0: one FP16 scale and 32 packed 4-bit values per block; see q4.h.
void embed_q4_0(const q3x_q4::Block* weights, int cols, int token, float* out) {
    const int blocks = cols / q3x_q4::BLOCK_SIZE;
    const q3x_q4::Block* row = weights + static_cast<size_t>(token) * blocks;
    for (int block = 0; block < blocks; ++block) {
        const float scale = f16(row[block].scale);
        for (int i = 0; i < q3x_q4::BLOCK_SIZE; ++i)
            out[block * q3x_q4::BLOCK_SIZE + i] =
                scale * q3x_q4::value(row[block], i);
    }
}
float dot_q4_0(const q3x_q4::Block* blocks, const float* x, int n) {
    float sum = 0.0f;
    for (int block = 0; block < n / q3x_q4::BLOCK_SIZE; ++block) {
        float inner = 0.0f;
        for (int i = 0; i < q3x_q4::BLOCK_SIZE; ++i)
            inner += q3x_q4::value(blocks[block], i) *
                     x[block * q3x_q4::BLOCK_SIZE + i];
        sum += f16(blocks[block].scale) * inner;
    }
    return sum;
}
void mv_q4_0(const q3x_q4::Block* weights, int rows, int cols, const float* x, float* y) {
    const int blocks = cols / q3x_q4::BLOCK_SIZE;
    for (int row = 0; row < rows; ++row)
        y[row] = dot_q4_0(weights + static_cast<size_t>(row) * blocks, x, cols);
}

#if Q3X_CPU_OPT

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

#endif  // Q3X_CPU_OPT

}  // namespace

// Public extension entry points; false leaves the engine baseline to run.
bool quantized_bytes(q3x_model::MatrixType type, int rows, int cols, size_t* bytes) {
    int block_size;
    size_t block_bytes;
    switch (type) {
    case q3x_model::MATRIX_Q8_0:
        block_size = q3x_q8::BLOCK_SIZE; block_bytes = sizeof(q3x_q8::Block); break;
    case q3x_model::MATRIX_Q4_0:
        block_size = q3x_q4::BLOCK_SIZE; block_bytes = sizeof(q3x_q4::Block); break;
    default: return false;
    }
    if (rows <= 0 || cols <= 0 || cols % block_size != 0) return false;
    const size_t blocks = static_cast<size_t>(cols) / block_size;
    const size_t maximum = std::numeric_limits<size_t>::max();
    if (blocks > maximum / block_bytes) return false;
    const size_t row_bytes = blocks * block_bytes;
    if (static_cast<size_t>(rows) > maximum / row_bytes) return false;
    *bytes = static_cast<size_t>(rows) * row_bytes;
    return true;
}

bool try_embed(const void* weights, q3x_model::MatrixType type, int cols, int token, float* out) {
    switch (type) {
    case q3x_model::MATRIX_Q8_0:
        embed_q8_0(static_cast<const q3x_q8::Block*>(weights), cols, token, out); return true;
    case q3x_model::MATRIX_Q4_0:
        embed_q4_0(static_cast<const q3x_q4::Block*>(weights), cols, token, out); return true;
    case q3x_model::MATRIX_BF16: return false;
    }
    Q3X_ASSERT(false, "quantized embedding type=%u", static_cast<unsigned>(type));
}

bool try_mv(const void* weights, q3x_model::MatrixType type, int rows, int cols,
            const float* x, float* y) {
#if Q3X_CPU_OPT
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
#endif  // Q3X_CPU_OPT
    // Quantization remains available when architecture acceleration is disabled.
    switch (type) {
    case q3x_model::MATRIX_Q8_0:
        mv_q8_0(static_cast<const q3x_q8::Block*>(weights), rows, cols, x, y); return true;
    case q3x_model::MATRIX_Q4_0:
        mv_q4_0(static_cast<const q3x_q4::Block*>(weights), rows, cols, x, y); return true;
    case q3x_model::MATRIX_BF16: return false;
    }
    Q3X_ASSERT(false, "quantized mv type=%u", static_cast<unsigned>(type));
}

bool try_dot(const float* a, const float* b, int n, float* out) {
#if Q3X_CPU_OPT && defined(__AVX512F__)
    *out = x86::dot_f32(a, b, n);
    return true;
#else
    (void)a; (void)b; (void)n; (void)out;
    return false;
#endif
}

}  // namespace q3x_cpu

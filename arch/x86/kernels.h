#pragma once

#include <immintrin.h>
#include <cstdint>
#include <cstring>

namespace q3x_cpu::x86 {

// Restored from 445f3a1: FP32 accumulation, including for BF16 weights.
inline float dot_f32(const float* a, const float* b, int n) {
    __m512 sums = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16)
        sums = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), sums);
    float sum = _mm512_reduce_add_ps(sums);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

#if defined(__AVX512BW__)
inline float dot_bf16(const uint16_t* w, const float* x, int n) {
    __m512 sums = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256i packed = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
        const __m512i bits = _mm512_slli_epi32(_mm512_cvtepu16_epi32(packed), 16);
        sums = _mm512_fmadd_ps(_mm512_castsi512_ps(bits), _mm512_loadu_ps(x + i), sums);
    }
    float sum = _mm512_reduce_add_ps(sums);
    for (; i < n; ++i) {
        const uint32_t bits = static_cast<uint32_t>(w[i]) << 16;
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        sum += value * x[i];
    }
    return sum;
}
#endif

}  // namespace q3x_cpu::x86

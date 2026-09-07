#pragma once

#include <arm_neon.h>
#include <cstring>
#include "q8.h"

namespace q3x_cpu::arm {

inline float half_to_float(uint16_t value) {
    __fp16 half;
    std::memcpy(&half, &value, sizeof(half));
    return static_cast<float>(half);
}

// Packed Q8 weights times FP32 input; keep the original NEON reduction order.
inline float dot_q8(const q3x_q8::Block* blocks, const float* x, int n) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    for (int block = 0; block < n / q3x_q8::BLOCK_SIZE; ++block) {
        const int8x16_t q0 = vld1q_s8(blocks[block].values);
        const int8x16_t q1 = vld1q_s8(blocks[block].values + 16);
        const int16x8_t q00 = vmovl_s8(vget_low_s8(q0));
        const int16x8_t q01 = vmovl_s8(vget_high_s8(q0));
        const int16x8_t q10 = vmovl_s8(vget_low_s8(q1));
        const int16x8_t q11 = vmovl_s8(vget_high_s8(q1));
        const float* input = x + block * q3x_q8::BLOCK_SIZE;
        float32x4_t inner = vmulq_f32(
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(q00))), vld1q_f32(input));
        inner = vfmaq_f32(inner,
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(q00))), vld1q_f32(input + 4));
        inner = vfmaq_f32(inner,
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(q01))), vld1q_f32(input + 8));
        inner = vfmaq_f32(inner,
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(q01))), vld1q_f32(input + 12));
        inner = vfmaq_f32(inner,
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(q10))), vld1q_f32(input + 16));
        inner = vfmaq_f32(inner,
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(q10))), vld1q_f32(input + 20));
        inner = vfmaq_f32(inner,
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(q11))), vld1q_f32(input + 24));
        inner = vfmaq_f32(inner,
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(q11))), vld1q_f32(input + 28));
        sum = vfmaq_n_f32(sum, inner, half_to_float(blocks[block].scale));
    }
    return vaddvq_f32(sum);
}

}  // namespace q3x_cpu::arm

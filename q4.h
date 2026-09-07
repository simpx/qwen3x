#ifndef QWEN3X_Q4_H
#define QWEN3X_Q4_H

#include <cstdint>

namespace q3x_q4 {

constexpr int BLOCK_SIZE = 32;

// ggml Q4_0: one FP16 scale and 32 four-bit values. The low nibbles hold
// values 0..15 and the high nibbles hold values 16..31.
struct Block {
    uint16_t scale;
    uint8_t values[BLOCK_SIZE / 2];
};

static_assert(sizeof(Block) == 18, "Q4_0 block layout mismatch");

inline int value(const Block& block, int index) {
    const uint8_t packed = block.values[index % (BLOCK_SIZE / 2)];
    const int quant = index < BLOCK_SIZE / 2 ? packed & 0x0f : packed >> 4;
    return quant - 8;
}

}  // namespace q3x_q4

#endif

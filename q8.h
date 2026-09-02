#ifndef QWEN3X_Q8_H
#define QWEN3X_Q8_H

#include <cstdint>

namespace q35_q8 {

constexpr int BLOCK_SIZE = 32;

struct Block {
    uint16_t scale;
    int8_t values[BLOCK_SIZE];
};

static_assert(sizeof(Block) == 34, "Q8_0 block layout mismatch");

}  // namespace q35_q8

#endif

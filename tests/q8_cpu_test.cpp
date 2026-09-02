#include <cassert>
#include <cmath>

#include "../engine.cpp"

int main() {
    q35_q8::Block blocks[4]{};
    const uint16_t scales[4] = {0x3c00, 0x3800, 0x4000, 0x3400};
    for (int block = 0; block < 4; ++block) {
        blocks[block].scale = scales[block];
        for (int index = 0; index < q35_q8::BLOCK_SIZE; ++index)
            blocks[block].values[index] = static_cast<int8_t>((index % 9) - 4 + block);
    }
    float input[64];
    for (int index = 0; index < 64; ++index)
        input[index] = (index % 7 - 3) * 0.125f;

    const float decoded_scales[4] = {1.0f, 0.5f, 2.0f, 0.25f};
    float expected[2]{};
    for (int row = 0; row < 2; ++row) {
        for (int block = 0; block < 2; ++block) {
            float inner = 0.0f;
            for (int index = 0; index < q35_q8::BLOCK_SIZE; ++index) {
                inner += blocks[row * 2 + block].values[index] *
                         input[block * q35_q8::BLOCK_SIZE + index];
            }
            expected[row] += decoded_scales[row * 2 + block] * inner;
        }
    }

    q35_backend::Linear matrix {
        blocks, 2, 64, q35_model::MATRIX_Q8_0,
    };
    float output[2]{};
    q35_backend::mv(matrix, input, output);
    assert(std::abs(output[0] - expected[0]) < 1e-6f);
    assert(std::abs(output[1] - expected[1]) < 1e-6f);

    float embedding[64]{};
    q35_backend::embed(matrix, 1, embedding);
    for (int block = 0; block < 2; ++block) {
        for (int index = 0; index < q35_q8::BLOCK_SIZE; ++index) {
            const float value = decoded_scales[2 + block] *
                                blocks[2 + block].values[index];
            assert(embedding[block * q35_q8::BLOCK_SIZE + index] == value);
        }
    }
    std::puts("q8-cpu-test: ok");
}

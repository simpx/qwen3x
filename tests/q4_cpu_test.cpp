#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>

#include "../engine.cpp"

namespace {

constexpr size_t QWEN36_FILE_SIZE = 19528534784ULL;

size_t q4_35b_file_size() {
    const q3x_model::ModelConfig& c = q3x_model::QWEN36_35B_A3B;
    size_t cursor = q3x_model::HEADER_SIZE;
    auto tensor = [&](size_t bytes) {
        cursor += (64 - cursor % 64) % 64;
        cursor += bytes;
    };
    auto q4 = [&](size_t values) {
        assert(values % q3x_q4::BLOCK_SIZE == 0);
        tensor(values / q3x_q4::BLOCK_SIZE * sizeof(q3x_q4::Block));
    };
    auto bf16 = [&](size_t values) { tensor(values * sizeof(q3x_backend::BF16)); };
    const int AS = c.AH * c.AD, KVW = c.KVH * c.AD;
    const int DO = c.VH * c.VD, DQKV = 2 * c.KH * c.KD + DO;
    q4(static_cast<size_t>(c.V) * c.H);
    q4(static_cast<size_t>(c.V) * c.H);
    bf16(c.H);
    for (int layer = 0; layer < c.N; ++layer) {
        bf16(c.H);
        if (layer % c.AI != c.AI - 1) {
            q4(static_cast<size_t>(DQKV) * c.H);
            q4(static_cast<size_t>(DO) * c.H);
            q4(static_cast<size_t>(c.VH) * c.H);
            q4(static_cast<size_t>(c.VH) * c.H);
            bf16(static_cast<size_t>(DQKV) * c.CK);
            tensor(static_cast<size_t>(c.VH) * sizeof(float));
            bf16(c.VH);
            tensor(static_cast<size_t>(c.VD) * sizeof(float));
            q4(static_cast<size_t>(c.H) * DO);
        } else {
            q4(static_cast<size_t>(2 * AS) * c.H);
            q4(static_cast<size_t>(KVW) * c.H);
            q4(static_cast<size_t>(KVW) * c.H);
            bf16(c.AD);
            bf16(c.AD);
            q4(static_cast<size_t>(c.H) * AS);
        }
        bf16(c.H);
        bf16(static_cast<size_t>(c.experts) * c.H);
        q4(static_cast<size_t>(c.experts) * 2 * c.I * c.H);
        q4(static_cast<size_t>(c.experts) * c.H * c.I);
        q4(static_cast<size_t>(c.shared_I) * c.H);
        q4(static_cast<size_t>(c.shared_I) * c.H);
        q4(static_cast<size_t>(c.H) * c.shared_I);
        bf16(c.H);
    }
    return cursor;
}

std::array<uint8_t, q3x_model::HEADER_SIZE> q4_35b_header() {
    std::array<uint8_t, q3x_model::HEADER_SIZE> header {};
    std::memcpy(header.data(), "Q3XMODL\0", 8);
    const auto& c = q3x_model::QWEN36_35B_A3B;
    const uint32_t fields[q3x_model::CONFIG_FIELD_COUNT] = {
        c.id, static_cast<uint32_t>(c.V), static_cast<uint32_t>(c.H),
        static_cast<uint32_t>(c.I), static_cast<uint32_t>(c.N),
        static_cast<uint32_t>(c.AI), static_cast<uint32_t>(c.AH),
        static_cast<uint32_t>(c.KVH), static_cast<uint32_t>(c.AD),
        static_cast<uint32_t>(c.RD), static_cast<uint32_t>(c.KH),
        static_cast<uint32_t>(c.VH), static_cast<uint32_t>(c.KD),
        static_cast<uint32_t>(c.VD), static_cast<uint32_t>(c.CK),
        q3x_model::MAX_CONTEXT,
    };
    std::memcpy(header.data() + q3x_model::HEADER_PREFIX_SIZE,
                fields, sizeof(fields));
    return header;
}

void loader_test() {
    char path[] = "/tmp/qwen36-q4-loader-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    const auto header = q4_35b_header();
    assert(q4_35b_file_size() == QWEN36_FILE_SIZE);
    auto restore = [&]() {
        assert(ftruncate(fd, static_cast<off_t>(QWEN36_FILE_SIZE)) == 0);
        assert(pwrite(fd, header.data(), header.size(), 0) ==
               static_cast<ssize_t>(header.size()));
    };
    restore();
    {
        q3x_backend::Model model;
        const char* error = nullptr;
        assert(model.load(path, &error));
        assert(error == nullptr);
        assert(model.config->id == 36035);
        assert(model.layer[39].moe.gate_up.experts == 256);
    }
    auto reject_header = [&](off_t offset, uint32_t value, const char* expected) {
        restore();
        assert(pwrite(fd, &value, sizeof(value), offset) ==
               static_cast<ssize_t>(sizeof(value)));
        q3x_backend::Model model;
        const char* error = nullptr;
        assert(!model.load(path, &error));
        assert(error && std::strstr(error, expected));
    };
    reject_header(0, 0, "magic");
    reject_header(q3x_model::HEADER_PREFIX_SIZE, 12345, "model ID");
    reject_header(q3x_model::HEADER_PREFIX_SIZE + sizeof(uint32_t), 1,
                  "header mismatch");
    restore();
    assert(ftruncate(fd, static_cast<off_t>(QWEN36_FILE_SIZE - 1)) == 0);
    {
        q3x_backend::Model model;
        const char* error = nullptr;
        assert(!model.load(path, &error));
        assert(error && std::strstr(error, "truncated"));
    }
    restore();
    assert(ftruncate(fd, static_cast<off_t>(QWEN36_FILE_SIZE + 1)) == 0);
    {
        q3x_backend::Model model;
        const char* error = nullptr;
        assert(!model.load(path, &error));
        assert(error && std::strstr(error, "size does not match schema"));
    }
    close(fd);
    assert(unlink(path) == 0);
}

void q4_math_test() {
    q3x_q4::Block blocks[4] {};
    const uint16_t scales[4] = {0x3c00, 0x3800, 0x4000, 0x3400};
    for (int block = 0; block < 4; ++block) {
        blocks[block].scale = scales[block];
        for (int index = 0; index < 16; ++index) {
            const int low = (index + block) % 16;
            const int high = (15 - index + block) % 16;
            blocks[block].values[index] = static_cast<uint8_t>(low | (high << 4));
        }
    }
    float input[64];
    for (int index = 0; index < 64; ++index)
        input[index] = (index % 7 - 3) * 0.125f;
    const float decoded_scales[4] = {1.0f, 0.5f, 2.0f, 0.25f};
    float expected[2] {};
    for (int row = 0; row < 2; ++row) {
        for (int block = 0; block < 2; ++block) {
            for (int index = 0; index < 16; ++index) {
                const uint8_t packed = blocks[row * 2 + block].values[index];
                expected[row] += decoded_scales[row * 2 + block] *
                    ((packed & 15) - 8) * input[block * 32 + index];
                expected[row] += decoded_scales[row * 2 + block] *
                    ((packed >> 4) - 8) * input[block * 32 + index + 16];
            }
        }
    }
    q3x_backend::Linear matrix {blocks, 2, 64, q3x_model::MATRIX_Q4_0};
    float output[2] {};
    q3x_backend::mv(matrix, input, output);
    assert(std::abs(output[0] - expected[0]) < 1e-6f);
    assert(std::abs(output[1] - expected[1]) < 1e-6f);
    float embedding[64] {};
    q3x_backend::embed(matrix, 1, embedding);
    for (int index = 0; index < 16; ++index) {
        const uint8_t low = blocks[2].values[index] & 15;
        const uint8_t high = blocks[2].values[index] >> 4;
        assert(embedding[index] == 2.0f * (low - 8));
        assert(embedding[index + 16] == 2.0f * (high - 8));
    }
}

void router_test() {
    const float logits[6] = {-2.0f, 3.0f, 1.0f, 4.0f, -1.0f, 2.0f};
    float probabilities[6] {};
    int ids[3] {};
    q3x_backend::router_top_k(logits, 6, 3, ids, probabilities);
    assert(ids[0] == 3 && ids[1] == 1 && ids[2] == 5);
    const float sum = probabilities[ids[0]] + probabilities[ids[1]] +
                      probabilities[ids[2]];
    assert(std::abs(sum - 1.0f) < 1e-6f);

    float wide_logits[256], wide_probabilities[256] {};
    int wide_ids[8] {};
    for (int expert = 0; expert < 256; ++expert)
        wide_logits[expert] = expert * 0.03125f;
    q3x_backend::router_top_k(wide_logits, 256, 8,
                              wide_ids, wide_probabilities);
    float wide_sum = 0.0f;
    for (int slot = 0; slot < 8; ++slot) {
        assert(wide_ids[slot] == 255 - slot);
        wide_sum += wide_probabilities[wide_ids[slot]];
    }
    assert(std::abs(wide_sum - 1.0f) < 1e-6f);
}

q3x_backend::BF16 bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<q3x_backend::BF16>(bits >> 16);
}

void fill_constant_q4(q3x_q4::Block* blocks, size_t count, uint16_t scale) {
    for (size_t index = 0; index < count; ++index) {
        blocks[index].scale = scale;
        std::memset(blocks[index].values, 0x99, sizeof(blocks[index].values));
    }
}

void moe_test() {
    const q3x_model::ModelConfig config {
        1, "synthetic-MoE", 64, 32, 32, 1,
        1, 1, 1, 32, 16, 1, 1, 32, 32, 4,
        4, 2, 32, q3x_model::MATRIX_Q4_0, false,
    };
    constexpr int EXPERTS = 4, H = 32, I = 32;
    q3x_backend::BF16 router[EXPERTS * H] {};
    const float router_values[EXPERTS] = {-0.125f, 0.0625f, 0.125f, -0.0625f};
    for (int expert = 0; expert < EXPERTS; ++expert)
        for (int column = 0; column < H; ++column)
            router[expert * H + column] = bf16(router_values[expert]);

    q3x_q4::Block gate_up[EXPERTS * 2 * I] {};
    q3x_q4::Block down[EXPERTS * H] {};
    const uint16_t scales[EXPERTS] = {0x3400, 0x3800, 0x3c00, 0x4000};
    for (int expert = 0; expert < EXPERTS; ++expert) {
        fill_constant_q4(gate_up + expert * 2 * I, 2 * I, scales[expert]);
        fill_constant_q4(down + expert * H, H, scales[expert]);
    }
    q3x_q4::Block shared_gate[I] {}, shared_up[I] {}, shared_down[H] {};
    fill_constant_q4(shared_gate, I, 0x3400);
    fill_constant_q4(shared_up, I, 0x3400);
    fill_constant_q4(shared_down, H, 0x3400);
    q3x_backend::BF16 shared_scale[H];
    std::fill(shared_scale, shared_scale + H, bf16(0.03125f));
    q3x_backend::MoeWeights weights {
        {router, EXPERTS, H, q3x_model::MATRIX_BF16},
        {gate_up, EXPERTS, 2 * I, H, q3x_model::MATRIX_Q4_0},
        {down, EXPERTS, H, I, q3x_model::MATRIX_Q4_0},
        {shared_gate, I, H, q3x_model::MATRIX_Q4_0},
        {shared_up, I, H, q3x_model::MATRIX_Q4_0},
        {shared_down, H, I, q3x_model::MATRIX_Q4_0},
        {shared_scale, 1, H, q3x_model::MATRIX_BF16},
    };
    float input[H];
    std::fill(input, input + H, 1.0f);
    q3x_backend::Work work(config);
    float output[H] {};
    q3x_backend::moe(weights, input, work, output, config);

    const float logits[EXPERTS] = {-4.0f, 2.0f, 4.0f, -2.0f};
    const float p1 = std::exp(logits[1] - logits[2]);
    const float expert1 = 0.5f * I *
        q3x_backend::silu(0.5f * H) * (0.5f * H);
    const float expert2 = 1.0f * I *
        q3x_backend::silu(1.0f * H) * (1.0f * H);
    const float routed = (p1 * expert1 + expert2) / (p1 + 1.0f);
    const float shared_hidden = q3x_backend::silu(0.25f * H) * (0.25f * H);
    const float shared = q3x_backend::sigmoid(1.0f) * 0.25f * I * shared_hidden;
    const float expected = routed + shared;
    for (float value : output)
        assert(std::abs(value - expected) < 1e-2f);

    float residual[H];
    std::fill(residual, residual + H, 3.0f);
    q3x_backend::residual_add(residual, output, H);
    for (float value : residual)
        assert(std::abs(value - (expected + 3.0f)) < 1e-2f);
}

}  // namespace

int main() {
    q4_math_test();
    router_test();
    moe_test();
    loader_test();
    std::puts("q4-cpu-test: ok");
}

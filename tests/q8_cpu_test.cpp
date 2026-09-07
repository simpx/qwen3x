#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include "../engine.cpp"

namespace {

// Different rows, non-binary-exact Q8 scales, SIMD tails and the GCD threshold.
// Double accumulation is an independent numerical oracle for both build modes.
void cpu_kernel_test() {
    uint32_t random = 42;
    auto sample = [&]() {
        random = random * 1664525u + 1013904223u;
        return (static_cast<int>(random >> 16) - 32768) / 32768.0f;
    };
    auto check = [](float actual, double expected, double magnitude) {
        assert(std::isfinite(actual));
        assert(std::abs(actual - expected) <= 2e-6 * (1.0 + magnitude));
    };
    for (int cols : {1, 15, 16, 17, 31, 32, 63, 96, 257}) {
        std::vector<float> x(cols), a(cols);
        double expected_dot = 0.0, magnitude_dot = 0.0;
        for (int i = 0; i < cols; ++i) {
            x[i] = sample(); a[i] = sample();
            const double product = static_cast<double>(x[i]) * a[i];
            expected_dot += product; magnitude_dot += std::abs(product);
        }
        check(q3x_backend::dot(a.data(), x.data(), cols), expected_dot, magnitude_dot);
        float fast_dot = 123.0f;
        if (q3x_cpu::try_dot(a.data(), x.data(), cols, &fast_dot))
            check(fast_dot, expected_dot, magnitude_dot);
        else
            assert(fast_dot == 123.0f);

        for (int rows : {3, 1023, 1024, 1031}) {
            std::vector<uint16_t> weights(rows * cols);
            for (auto& w : weights) {
                const float value = sample();
                uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
                w = static_cast<uint16_t>(bits >> 16);
            }
            std::vector<float> output(rows), fast(rows, 123.0f);
            q3x_backend::Linear matrix {weights.data(), rows, cols, q3x_model::MATRIX_BF16};
            q3x_backend::mv(matrix, x.data(), output.data());
            const bool accelerated = q3x_cpu::try_mv(weights.data(), matrix.type, rows, cols,
                                                     x.data(), fast.data());
            for (int row = 0; row < rows; ++row) {
                double expected = 0.0, magnitude = 0.0;
                for (int i = 0; i < cols; ++i) {
                    const double product = static_cast<double>(q3x_backend::f32(weights[row * cols + i])) * x[i];
                    expected += product; magnitude += std::abs(product);
                }
                check(output[row], expected, magnitude);
                if (accelerated) check(fast[row], expected, magnitude);
                else assert(fast[row] == 123.0f);
            }
        }
    }
    for (int rows : {3, 1023, 1024, 1031}) {
        constexpr int cols = 96, blocks = cols / q3x_q8::BLOCK_SIZE;
        const uint16_t scales[] = {0x3555, 0x2e66, 0x0400, 0x0001};
        const double decoded[] = {0.333251953125, 0.0999755859375,
                                  0.00006103515625, 0.000000059604644775390625};
        std::vector<q3x_q8::Block> weights(rows * blocks);
        float x[cols];
        for (float& v : x) v = sample();
        for (int block = 0; block < rows * blocks; ++block) {
            weights[block].scale = scales[block % 4];
            for (auto& v : weights[block].values) v = static_cast<int8_t>(sample() * 127);
        }
        std::vector<float> output(rows);
        q3x_backend::Linear matrix {weights.data(), rows, cols, q3x_model::MATRIX_Q8_0};
        q3x_backend::mv(matrix, x, output.data());
        for (int row = 0; row < rows; ++row) {
            double expected = 0.0, magnitude = 0.0;
            for (int block = 0; block < blocks; ++block)
                for (int i = 0; i < q3x_q8::BLOCK_SIZE; ++i) {
                    const int index = row * blocks + block;
                    const double product = decoded[index % 4] * weights[index].values[i] *
                                           x[block * q3x_q8::BLOCK_SIZE + i];
                    expected += product; magnitude += std::abs(product);
                }
            check(output[row], expected, magnitude);
        }
    }
    float untouched = 123.0f;
    assert(!q3x_cpu::try_mv(nullptr, q3x_model::MATRIX_Q4_0, 1, 32, nullptr, &untouched));
    assert(untouched == 123.0f);
}

size_t q8_9b_file_size() {
    const q3x_model::ModelConfig& c = q3x_model::QWEN35_9B;
    size_t cursor = q3x_model::HEADER_SIZE;
    auto tensor = [&](size_t bytes) {
        cursor += (64 - cursor % 64) % 64;
        cursor += bytes;
    };
    auto linear = [&](int rows, int columns) {
        tensor(static_cast<size_t>(rows) * columns /
               q3x_q8::BLOCK_SIZE * sizeof(q3x_q8::Block));
    };
    const int AS = c.AH * c.AD;
    const int KVW = c.KVH * c.AD;
    const int DO = c.VH * c.VD;
    const int DQKV = 2 * c.KH * c.KD + DO;
    linear(c.V, c.H);
    linear(c.V, c.H);
    tensor(static_cast<size_t>(c.H) * sizeof(q3x_backend::BF16));
    for (int layer = 0; layer < c.N; ++layer) {
        tensor(static_cast<size_t>(c.H) * sizeof(q3x_backend::BF16));
        if (layer % c.AI != c.AI - 1) {
            linear(DQKV, c.H);
            linear(DO, c.H);
            linear(c.VH, c.H);
            linear(c.VH, c.H);
            tensor(static_cast<size_t>(DQKV) * c.CK *
                   sizeof(q3x_backend::BF16));
            tensor(static_cast<size_t>(c.VH) * sizeof(float));
            tensor(static_cast<size_t>(c.VH) * sizeof(q3x_backend::BF16));
            tensor(static_cast<size_t>(c.VD) * sizeof(float));
            linear(c.H, DO);
        } else {
            linear(2 * AS, c.H);
            linear(KVW, c.H);
            linear(KVW, c.H);
            tensor(static_cast<size_t>(c.AD) * sizeof(q3x_backend::BF16));
            tensor(static_cast<size_t>(c.AD) * sizeof(q3x_backend::BF16));
            linear(c.H, AS);
        }
        tensor(static_cast<size_t>(c.H) * sizeof(q3x_backend::BF16));
        linear(c.I, c.H);
        linear(c.I, c.H);
        linear(c.H, c.I);
    }
    return cursor;
}

std::array<uint8_t, q3x_model::HEADER_SIZE> q8_9b_header() {
    std::array<uint8_t, q3x_model::HEADER_SIZE> header {};
    std::memcpy(header.data(), "Q3XMODL\0", 8);
    const q3x_model::ModelConfig& c = q3x_model::QWEN35_9B;
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

void write_exact(int fd, const void* data, size_t size, off_t offset) {
    assert(pwrite(fd, data, size, offset) == static_cast<ssize_t>(size));
}

void check_loader(const char* path, bool expected, const char* expected_error) {
    q3x_backend::Model model;
    const char* error = nullptr;
    assert(model.load(path, &error) == expected);
    if (expected) {
        assert(error == nullptr);
    } else {
        assert(error != nullptr);
        assert(std::strstr(error, expected_error) != nullptr);
    }
}

void loader_test() {
    char path[] = "/tmp/qwen3x-q8-loader-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    const auto header = q8_9b_header();
    const size_t expected_size = q8_9b_file_size();
    assert(expected_size == 9514418816ULL);
    auto restore = [&]() {
        assert(ftruncate(fd, static_cast<off_t>(expected_size)) == 0);
        write_exact(fd, header.data(), header.size(), 0);
    };

    restore();
    check_loader(path, true, nullptr);

    assert(ftruncate(fd, static_cast<off_t>(expected_size - 1)) == 0);
    check_loader(path, false, "truncated");
    restore();
    assert(ftruncate(fd, static_cast<off_t>(expected_size + 1)) == 0);
    check_loader(path, false, "size does not match schema");

    restore();
    const uint8_t wrong_magic = 'X';
    write_exact(fd, &wrong_magic, sizeof(wrong_magic), 0);
    check_loader(path, false, "wrong model.bin magic");

    restore();
    const uint32_t unknown_id = 9001;
    write_exact(fd, &unknown_id, sizeof(unknown_id),
                q3x_model::HEADER_PREFIX_SIZE +
                q3x_model::MODEL_ID * sizeof(uint32_t));
    check_loader(path, false, "unsupported Qwen model ID");

    restore();
    const uint32_t wrong_hidden = 4095;
    write_exact(fd, &wrong_hidden, sizeof(wrong_hidden),
                q3x_model::HEADER_PREFIX_SIZE +
                q3x_model::HIDDEN_SIZE * sizeof(uint32_t));
    check_loader(path, false, "header mismatch");

    close(fd);
    assert(unlink(path) == 0);
}

}  // namespace

int main() {
    cpu_kernel_test();
    q3x_q8::Block blocks[4]{};
    const uint16_t scales[4] = {0x3c00, 0x3800, 0x4000, 0x3400};
    for (int block = 0; block < 4; ++block) {
        blocks[block].scale = scales[block];
        for (int index = 0; index < q3x_q8::BLOCK_SIZE; ++index)
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
            for (int index = 0; index < q3x_q8::BLOCK_SIZE; ++index) {
                inner += blocks[row * 2 + block].values[index] *
                         input[block * q3x_q8::BLOCK_SIZE + index];
            }
            expected[row] += decoded_scales[row * 2 + block] * inner;
        }
    }

    q3x_backend::Linear matrix {
        blocks, 2, 64, q3x_model::MATRIX_Q8_0,
    };
    float output[2]{};
    q3x_backend::mv(matrix, input, output);
    assert(std::abs(output[0] - expected[0]) < 1e-6f);
    assert(std::abs(output[1] - expected[1]) < 1e-6f);

    float embedding[64]{};
    q3x_backend::embed(matrix, 1, embedding);
    for (int block = 0; block < 2; ++block) {
        for (int index = 0; index < q3x_q8::BLOCK_SIZE; ++index) {
            const float value = decoded_scales[2 + block] *
                                blocks[2 + block].values[index];
            assert(embedding[block * q3x_q8::BLOCK_SIZE + index] == value);
        }
    }

    // Exercise the macOS row-parallel path, whose threshold deliberately
    // stays above the tiny fixed-vector check above.
    constexpr int rows = 1024;
    std::vector<q3x_q8::Block> parallel_weights(rows * 2);
    for (int row = 0; row < rows; ++row) {
        parallel_weights[row * 2] = blocks[0];
        parallel_weights[row * 2 + 1] = blocks[1];
    }
    std::vector<float> parallel_output(rows);
    q3x_backend::Linear parallel_matrix {
        parallel_weights.data(), rows, 64, q3x_model::MATRIX_Q8_0,
    };
    q3x_backend::mv(parallel_matrix, input, parallel_output.data());
    for (float value : parallel_output)
        assert(std::abs(value - expected[0]) < 1e-6f);

    q3x_q4::Block q4_blocks[2]{};
    for (int block = 0; block < 2; ++block) {
        q4_blocks[block].scale = scales[block];
        for (int index = 0; index < 16; ++index) {
            const uint8_t low = static_cast<uint8_t>((index + block) % 16);
            const uint8_t high = static_cast<uint8_t>((index + block + 3) % 16);
            q4_blocks[block].values[index] = low | (high << 4);
        }
    }
    float q4_expected = 0.0f;
    for (int block = 0; block < 2; ++block)
        for (int index = 0; index < q3x_q4::BLOCK_SIZE; ++index)
            q4_expected += decoded_scales[block] *
                           q3x_q4::value(q4_blocks[block], index) *
                           input[block * q3x_q4::BLOCK_SIZE + index];
    q3x_backend::Linear q4_matrix {
        q4_blocks, 1, 64, q3x_model::MATRIX_Q4_0,
    };
    float q4_output = 0.0f;
    q3x_backend::mv(q4_matrix, input, &q4_output);
    assert(std::abs(q4_output - q4_expected) < 1e-6f);

    float q4_embedding[64]{};
    q3x_backend::embed(q4_matrix, 0, q4_embedding);
    for (int index = 0; index < 64; ++index)
        assert(q4_embedding[index] == decoded_scales[index / 32] *
                                      q3x_q4::value(q4_blocks[index / 32], index % 32));

    loader_test();
    std::puts("q8-cpu-test: ok");
}

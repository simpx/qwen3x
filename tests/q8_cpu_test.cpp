#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>

#include "../engine.cpp"

namespace {

size_t q8_9b_file_size() {
    const q35_model::ModelConfig& c = q35_model::QWEN35_9B;
    size_t cursor = q35_model::HEADER_SIZE;
    auto tensor = [&](size_t bytes) {
        cursor += (64 - cursor % 64) % 64;
        cursor += bytes;
    };
    auto linear = [&](int rows, int columns) {
        tensor(static_cast<size_t>(rows) * columns /
               q35_q8::BLOCK_SIZE * sizeof(q35_q8::Block));
    };
    const int AS = c.AH * c.AD;
    const int KVW = c.KVH * c.AD;
    const int DO = c.VH * c.VD;
    const int DQKV = 2 * c.KH * c.KD + DO;
    linear(c.V, c.H);
    linear(c.V, c.H);
    tensor(static_cast<size_t>(c.H) * sizeof(q35_backend::BF16));
    for (int layer = 0; layer < c.N; ++layer) {
        tensor(static_cast<size_t>(c.H) * sizeof(q35_backend::BF16));
        if (layer % c.AI != c.AI - 1) {
            linear(DQKV, c.H);
            linear(DO, c.H);
            linear(c.VH, c.H);
            linear(c.VH, c.H);
            tensor(static_cast<size_t>(DQKV) * c.CK *
                   sizeof(q35_backend::BF16));
            tensor(static_cast<size_t>(c.VH) * sizeof(float));
            tensor(static_cast<size_t>(c.VH) * sizeof(q35_backend::BF16));
            tensor(static_cast<size_t>(c.VD) * sizeof(float));
            linear(c.H, DO);
        } else {
            linear(2 * AS, c.H);
            linear(KVW, c.H);
            linear(KVW, c.H);
            tensor(static_cast<size_t>(c.AD) * sizeof(q35_backend::BF16));
            tensor(static_cast<size_t>(c.AD) * sizeof(q35_backend::BF16));
            linear(c.H, AS);
        }
        tensor(static_cast<size_t>(c.H) * sizeof(q35_backend::BF16));
        linear(c.I, c.H);
        linear(c.I, c.H);
        linear(c.H, c.I);
    }
    return cursor;
}

std::array<uint8_t, q35_model::HEADER_SIZE> q8_9b_header() {
    std::array<uint8_t, q35_model::HEADER_SIZE> header {};
    std::memcpy(header.data(), "Q35MODL\0", 8);
    const uint32_t version = q35_model::FORMAT_VERSION;
    std::memcpy(header.data() + 8, &version, sizeof(version));
    const q35_model::ModelConfig& c = q35_model::QWEN35_9B;
    const uint32_t fields[q35_model::CONFIG_FIELD_COUNT] = {
        c.id, static_cast<uint32_t>(c.V), static_cast<uint32_t>(c.H),
        static_cast<uint32_t>(c.I), static_cast<uint32_t>(c.N),
        static_cast<uint32_t>(c.AI), static_cast<uint32_t>(c.AH),
        static_cast<uint32_t>(c.KVH), static_cast<uint32_t>(c.AD),
        static_cast<uint32_t>(c.RD), static_cast<uint32_t>(c.KH),
        static_cast<uint32_t>(c.VH), static_cast<uint32_t>(c.KD),
        static_cast<uint32_t>(c.VD), static_cast<uint32_t>(c.CK),
        q35_model::MAX_CONTEXT,
    };
    std::memcpy(header.data() + q35_model::HEADER_PREFIX_SIZE,
                fields, sizeof(fields));
    return header;
}

void write_exact(int fd, const void* data, size_t size, off_t offset) {
    assert(pwrite(fd, data, size, offset) == static_cast<ssize_t>(size));
}

void check_loader(const char* path, bool expected, const char* expected_error) {
    q35_backend::Model model;
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
    char path[] = "/tmp/qwen35-q8-loader-XXXXXX";
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
    const uint32_t nonzero = 1;
    write_exact(fd, &nonzero, sizeof(nonzero), 12);
    check_loader(path, false, "unsupported model.bin version");

    restore();
    const uint32_t unknown_id = 9001;
    write_exact(fd, &unknown_id, sizeof(unknown_id),
                q35_model::HEADER_PREFIX_SIZE +
                q35_model::MODEL_ID * sizeof(uint32_t));
    check_loader(path, false, "unsupported Qwen3.5 model ID");

    restore();
    const uint32_t wrong_hidden = 4095;
    write_exact(fd, &wrong_hidden, sizeof(wrong_hidden),
                q35_model::HEADER_PREFIX_SIZE +
                q35_model::HIDDEN_SIZE * sizeof(uint32_t));
    check_loader(path, false, "header mismatch");

    close(fd);
    assert(unlink(path) == 0);
}

}  // namespace

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
    loader_test();
    std::puts("q8-cpu-test: ok");
}

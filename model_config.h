#ifndef QWEN3X_MODEL_CONFIG_H
#define QWEN3X_MODEL_CONFIG_H

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace q3x_model {

constexpr size_t HEADER_PREFIX_SIZE = 8;
constexpr size_t CONFIG_FIELD_COUNT = 16;
constexpr size_t HEADER_SIZE = HEADER_PREFIX_SIZE + CONFIG_FIELD_COUNT * 4;
constexpr int MAX_CONTEXT = 262144;
constexpr int MAX_TOP_K = 8;

enum MatrixType : uint32_t {
    MATRIX_BF16,
    MATRIX_Q8_0,
    MATRIX_Q4_0,
};

enum ConfigField : size_t {
    MODEL_ID,
    VOCAB_SIZE,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    LAYER_COUNT,
    ATTENTION_INTERVAL,
    ATTENTION_HEADS,
    KEY_VALUE_HEADS,
    ATTENTION_HEAD_DIM,
    ROTARY_DIM,
    DELTA_KEY_HEADS,
    DELTA_VALUE_HEADS,
    DELTA_KEY_DIM,
    DELTA_VALUE_DIM,
    CONV_KERNEL,
    CONTEXT_SIZE,
};

inline uint32_t header_field(const uint8_t* data, ConfigField field) {
    uint32_t value = 0;
    std::memcpy(&value, data + HEADER_PREFIX_SIZE +
                static_cast<size_t>(field) * sizeof(value), sizeof(value));
    return value;
}

struct ModelConfig {
    uint32_t id;
    const char* name;
    int V, H, I, N;
    int AI, AH, KVH, AD, RD;
    int KH, VH, KD, VD, CK;
    int experts, top_k, shared_I;
    MatrixType matrix_type;
    bool tied_embeddings;
};

constexpr ModelConfig QWEN35_08B = {
    800, "Qwen3.5-0.8B",              // id, name
    248320, 1024, 3584, 24,            // V, H, I, N
    4, 8, 2, 256, 64,                  // AI, AH, KVH, AD, RD
    16, 16, 128, 128, 4,               // KH, VH, KD, VD, CK
    0, 0, 0,                            // experts, top_k, shared_I
    MATRIX_BF16, true,                  // matrix_type, tied_embeddings
};

constexpr ModelConfig QWEN35_4B = {
    4000, "Qwen3.5-4B",                // id, name
    248320, 2560, 9216, 32,            // V, H, I, N
    4, 16, 4, 256, 64,                 // AI, AH, KVH, AD, RD
    16, 32, 128, 128, 4,               // KH, VH, KD, VD, CK
    0, 0, 0,                            // experts, top_k, shared_I
    MATRIX_BF16, true,                  // matrix_type, tied_embeddings
};

constexpr ModelConfig QWEN35_9B = {
    9000, "Qwen3.5-9B",                // id, name
    248320, 4096, 12288, 32,           // V, H, I, N
    4, 16, 4, 256, 64,                 // AI, AH, KVH, AD, RD
    16, 32, 128, 128, 4,               // KH, VH, KD, VD, CK
    0, 0, 0,                            // experts, top_k, shared_I
    MATRIX_Q8_0, false,                 // matrix_type, tied_embeddings
};

constexpr ModelConfig QWEN36_35B_A3B = {
    36035, "Qwen3.6-35B-A3B",          // id, name
    248320, 2048, 512, 40,             // V, H, I, N
    4, 16, 2, 256, 64,                 // AI, AH, KVH, AD, RD
    16, 32, 128, 128, 4,               // KH, VH, KD, VD, CK
    256, 8, 512,                        // experts, top_k, shared_I
    MATRIX_Q4_0, false,                 // matrix_type, tied_embeddings
};

constexpr ModelConfig QWEN38_27B = {
    38027, "Qwen3.8-27B",              // id, name
    248320, 5120, 17408, 64,           // V, H, I, N
    4, 24, 4, 256, 64,                 // AI, AH, KVH, AD, RD
    16, 48, 128, 128, 4,               // KH, VH, KD, VD, CK
    0, 0, 0,                            // experts, top_k, shared_I
    MATRIX_Q4_0, false,                 // matrix_type, tied_embeddings
};

inline const ModelConfig* config_for_id(uint32_t id) {
    switch (id) {
    case QWEN35_08B.id: return &QWEN35_08B;
    case QWEN35_4B.id: return &QWEN35_4B;
    case QWEN35_9B.id: return &QWEN35_9B;
    case QWEN36_35B_A3B.id: return &QWEN36_35B_A3B;
    case QWEN38_27B.id: return &QWEN38_27B;
    default: return nullptr;
    }
}

inline bool header_matches(const uint8_t* data, size_t size,
                           const ModelConfig& config) {
    if (!data || size < HEADER_SIZE) return false;
    const uint32_t expected[CONFIG_FIELD_COUNT] = {
        config.id,
        static_cast<uint32_t>(config.V),
        static_cast<uint32_t>(config.H),
        static_cast<uint32_t>(config.I),
        static_cast<uint32_t>(config.N),
        static_cast<uint32_t>(config.AI),
        static_cast<uint32_t>(config.AH),
        static_cast<uint32_t>(config.KVH),
        static_cast<uint32_t>(config.AD),
        static_cast<uint32_t>(config.RD),
        static_cast<uint32_t>(config.KH),
        static_cast<uint32_t>(config.VH),
        static_cast<uint32_t>(config.KD),
        static_cast<uint32_t>(config.VD),
        static_cast<uint32_t>(config.CK),
        MAX_CONTEXT,
    };
    for (size_t index = 0; index < CONFIG_FIELD_COUNT; ++index) {
        if (header_field(data, static_cast<ConfigField>(index)) !=
            expected[index]) return false;
    }
    return true;
}

}  // namespace q3x_model

#endif

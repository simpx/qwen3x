// qwen38.cpp -- a small, readable Qwen3.8 CPU reference.
//
// The reference starts with the numerics and state machines. It has no tensor
// framework and no runtime dependency. Accelerated kernels are added beside
// these functions rather than replacing the readable CPU oracle.

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef QWEN38_WITH_TOKENIZER
#include "qwen38_tokenizer.h"
#endif

namespace qwen38 {

// These values are the text_config of Qwen/Qwen3.8-27B.  The vision encoder
// and MTP head are intentionally outside the text-only execution path.
struct Config {
#ifdef QWEN38_TINY_MODEL
    // The fake checkpoint keeps Qwen3.8's 3 DeltaNet : 1 attention pattern
    // while reducing every width. It exists solely for end-to-end CPU/CUDA
    // verification on developer machines that cannot hold 27B BF16 weights.
    static constexpr int kVocabSize = 64;
    static constexpr int kHiddenSize = 32;
    static constexpr int kIntermediateSize = 64;
    static constexpr int kLayers = 4;
    static constexpr int kAttentionHeads = 4;
    static constexpr int kKvHeads = 1;
    static constexpr int kAttentionHeadDim = 8;
    static constexpr int kRotaryDim = 4;
    static constexpr int kDeltaKeyHeads = 2;
    static constexpr int kDeltaValueHeads = 4;
    static constexpr int kDeltaKeyDim = 4;
    static constexpr int kDeltaValueDim = 8;
    static constexpr const char* kDeltaAuxiliaryDtype = "BF16";
    static constexpr bool kTiedWordEmbeddings = false;
    static constexpr const char* kModelName = "tiny Qwen3.8-shaped fake";
#elif defined(QWEN38_DEV_08B_MODEL)
    // Daily development checkpoint: Qwen/Qwen3.5-0.8B.  This is deliberately
    // a second fixed model, not a generic model-config system.  It exercises
    // the real Qwen3.5 hybrid text path on ordinary developer hardware.
    static constexpr int kVocabSize = 248320;
    static constexpr int kHiddenSize = 1024;
    static constexpr int kIntermediateSize = 3584;
    static constexpr int kLayers = 24;
    static constexpr int kAttentionHeads = 8;
    static constexpr int kKvHeads = 2;
    static constexpr int kAttentionHeadDim = 256;
    static constexpr int kRotaryDim = 64;
    static constexpr int kDeltaKeyHeads = 16;
    static constexpr int kDeltaValueHeads = 16;
    static constexpr int kDeltaKeyDim = 128;
    static constexpr int kDeltaValueDim = 128;
    static constexpr const char* kDeltaAuxiliaryDtype = "F32";
    static constexpr bool kTiedWordEmbeddings = true;
    static constexpr const char* kModelName = "Qwen3.5-0.8B development checkpoint";
#else
    static constexpr int kVocabSize = 248320;
    static constexpr int kHiddenSize = 5120;
    static constexpr int kIntermediateSize = 17408;
    static constexpr int kLayers = 64;
    static constexpr int kAttentionHeads = 24;
    static constexpr int kKvHeads = 4;
    static constexpr int kAttentionHeadDim = 256;
    static constexpr int kRotaryDim = 64;
    static constexpr int kDeltaKeyHeads = 16;
    static constexpr int kDeltaValueHeads = 48;
    static constexpr int kDeltaKeyDim = 128;
    static constexpr int kDeltaValueDim = 128;
    static constexpr const char* kDeltaAuxiliaryDtype = "F32";
    static constexpr bool kTiedWordEmbeddings = false;
    static constexpr const char* kModelName = "Qwen3.8-27B";
#endif
    static constexpr int kFullAttentionInterval = 4;
    static constexpr int kDeltaConvKernel = 4;

    static constexpr int kAttentionSize = kAttentionHeads * kAttentionHeadDim;
    static constexpr int kAttentionQProjectionSize = 2 * kAttentionSize;
    static constexpr int kAttentionKvSize = kKvHeads * kAttentionHeadDim;
    static constexpr int kDeltaQkSize = kDeltaKeyHeads * kDeltaKeyDim;
    static constexpr int kDeltaOutputSize = kDeltaValueHeads * kDeltaValueDim;
    static constexpr int kDeltaQkvSize = 2 * kDeltaQkSize + kDeltaOutputSize;

    static constexpr float kRmsNormEps = 1.0e-6f;
    static constexpr float kRopeTheta = 10000000.0f;

    static bool is_deltanet_layer(int layer) {
        return layer >= 0 && layer < kLayers && (layer % kFullAttentionInterval) != 3;
    }

    static constexpr int kDeltaLayers = kLayers - kLayers / kFullAttentionInterval;
    static constexpr int kAttentionLayers = kLayers - kDeltaLayers;
    static constexpr int kTextTensorCount = (kTiedWordEmbeddings ? 2 : 3) + kDeltaLayers * 14 + kAttentionLayers * 11;
};

static_assert(Config::kAttentionHeads % Config::kKvHeads == 0, "GQA heads must divide evenly");
static_assert(Config::kDeltaValueHeads % Config::kDeltaKeyHeads == 0, "DeltaNet QK heads must repeat evenly");
#ifndef QWEN38_TINY_MODEL
#ifdef QWEN38_DEV_08B_MODEL
static_assert(Config::kAttentionSize == 2048 && Config::kAttentionQProjectionSize == 4096,
              "Qwen3.5-0.8B attention projection size changed");
static_assert(Config::kDeltaQkvSize == 6144 && Config::kDeltaOutputSize == 2048,
              "Qwen3.5-0.8B DeltaNet projection size changed");
static_assert(Config::kTextTensorCount == 320, "Qwen3.5-0.8B text tensor count changed");
#else
static_assert(Config::kAttentionSize == 6144 && Config::kAttentionQProjectionSize == 12288,
              "Qwen3.8 attention projection size changed");
static_assert(Config::kDeltaQkvSize == 10240 && Config::kDeltaOutputSize == 6144,
              "Qwen3.8 DeltaNet projection size changed");
static_assert(Config::kTextTensorCount == 851, "Qwen3.8 text tensor count changed");
#endif
#endif

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "qwen38: %s\n", message);
    std::exit(1);
}

[[noreturn]] void fail(const std::string& message) {
    fail(message.c_str());
}

void require(bool condition, const char* message) {
    if (!condition) fail(message);
}

bool nearly_equal(float a, float b, float tolerance = 1.0e-5f) {
    return std::fabs(a - b) <= tolerance;
}

// Safetensors stores the released checkpoint as BF16. The CPU reference performs scalar
// reference arithmetic in FP32 after this exact BF16-to-FP32 expansion.
float bf16_to_f32(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float f32_at_le(const uint8_t* bytes) {
    const uint32_t bits = static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
                          (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    // Round to nearest, ties to even, before retaining the high 16 bits.
    const uint32_t rounding_bias = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

float sigmoid(float x) {
    if (x >= 0.0f) {
        const float e = std::exp(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = std::exp(x);
    return e / (1.0f + e);
}

float silu(float x) {
    return x * sigmoid(x);
}

float softplus(float x) {
    // Stable enough for DeltaNet's a + dt_bias input.
    return x > 20.0f ? x : std::log1pf(std::exp(x));
}

// Row-major W[out_dim, in_dim] times x[in_dim].  This is intentionally a
// scalar GEMV: it is the CPU correctness oracle for future BLAS/GPU paths.
void matvec_cpu(const float* weight, int out_dim, int in_dim, const float* x, float* out) {
    for (int row = 0; row < out_dim; ++row) {
        const float* w = weight + static_cast<size_t>(row) * in_dim;
        float sum = 0.0f;
        for (int col = 0; col < in_dim; ++col) sum += w[col] * x[col];
        out[row] = sum;
    }
}

// Qwen3.5/Qwen3.8's ordinary RMSNorm stores an offset: (1 + weight), not
// the more common direct weight multiplier.
void rmsnorm_plus_cpu(const float* x, const float* weight, int size, float eps, float* out) {
    float square_sum = 0.0f;
    for (int i = 0; i < size; ++i) square_sum += x[i] * x[i];
    const float scale = 1.0f / std::sqrt(square_sum / size + eps);
    for (int i = 0; i < size; ++i) out[i] = x[i] * scale * (1.0f + weight[i]);
}

// DeltaNet's post-recurrence norm is different: direct norm.weight followed
// by the SiLU output gate z.
void rmsnorm_gated_cpu(const float* x, const float* weight, const float* z, int size, float eps,
                       float* out) {
    float square_sum = 0.0f;
    for (int i = 0; i < size; ++i) square_sum += x[i] * x[i];
    const float scale = 1.0f / std::sqrt(square_sum / size + eps);
    for (int i = 0; i < size; ++i) out[i] = x[i] * scale * weight[i] * silu(z[i]);
}

void swiglu_cpu(const float* gate, const float* up, int size, float* out) {
    for (int i = 0; i < size; ++i) out[i] = silu(gate[i]) * up[i];
}

void l2norm_inplace_cpu(float* x, int size, float eps = Config::kRmsNormEps) {
    float square_sum = 0.0f;
    for (int i = 0; i < size; ++i) square_sum += x[i] * x[i];
    const float scale = 1.0f / std::sqrt(square_sum + eps);
    for (int i = 0; i < size; ++i) x[i] *= scale;
}

// Applies Qwen's half-rotation layout to the first rotary_dim channels.  cos
// and sin are already calculated for the current position in FP32.
void rope_cpu(float* x, int head_dim, int rotary_dim, const float* cos, const float* sin) {
    require(rotary_dim > 0 && rotary_dim <= head_dim && (rotary_dim % 2) == 0,
            "invalid RoPE dimensions");
    const int half = rotary_dim / 2;
    std::vector<float> original(x, x + rotary_dim);
    for (int i = 0; i < half; ++i) {
        x[i] = original[i] * cos[i] - original[i + half] * sin[i];
        x[i + half] = original[i + half] * cos[i + half] + original[i] * sin[i + half];
    }
}

// A reference decode-time depthwise causal convolution.  state contains the
// preceding kernel_size - 1 pre-convolution values for each channel.
void depthwise_conv_step_cpu(const float* input, const float* weight, int channels, int kernel_size,
                             std::vector<float>* state, float* out) {
    require(kernel_size >= 2, "causal convolution needs a history slot");
    const int history = kernel_size - 1;
    require(state->size() == static_cast<size_t>(channels) * history, "bad convolution state shape");

    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        float* past = state->data() + static_cast<size_t>(c) * history;
        const float* w = weight + static_cast<size_t>(c) * kernel_size;
        for (int i = 0; i < history; ++i) sum += past[i] * w[i];
        sum += input[c] * w[history];
        for (int i = 0; i + 1 < history; ++i) past[i] = past[i + 1];
        past[history - 1] = input[c];
        out[c] = silu(sum);
    }
}

struct DeltaNetShape {
    int heads;
    int key_dim;
    int value_dim;
};

struct DeltaNetState {
    DeltaNetShape shape;
// Logical layout: [head][key_dim][value_dim]. Always FP32 in the CPU reference.
    std::vector<float> recurrent;

    explicit DeltaNetState(DeltaNetShape new_shape)
        : shape(new_shape),
          recurrent(static_cast<size_t>(new_shape.heads) * new_shape.key_dim * new_shape.value_dim, 0.0f) {}

    float* at(int head, int key_index, int value_index) {
        const size_t offset = (static_cast<size_t>(head) * shape.key_dim + key_index) * shape.value_dim + value_index;
        return &recurrent[offset];
    }
};

// The literal Gated DeltaNet recurrence. q and k are expected to be L2
// normalized, and q must already have its 1/sqrt(key_dim) attention scale.
void gated_delta_recurrence_cpu(const DeltaNetShape& shape, const float* q, const float* k,
                                const float* value, const float* log_decay, const float* beta,
                                DeltaNetState* state, float* out) {
    require(state->shape.heads == shape.heads && state->shape.key_dim == shape.key_dim &&
                state->shape.value_dim == shape.value_dim,
            "bad recurrent state shape");

    std::vector<float> memory(shape.value_dim);
    for (int h = 0; h < shape.heads; ++h) {
        const float* q_head = q + static_cast<size_t>(h) * shape.key_dim;
        const float* k_head = k + static_cast<size_t>(h) * shape.key_dim;
        const float* v_head = value + static_cast<size_t>(h) * shape.value_dim;
        float* out_head = out + static_cast<size_t>(h) * shape.value_dim;
        const float decay = std::exp(log_decay[h]);

        for (int key_index = 0; key_index < shape.key_dim; ++key_index) {
            for (int value_index = 0; value_index < shape.value_dim; ++value_index) {
                *state->at(h, key_index, value_index) *= decay;
            }
        }

        for (int value_index = 0; value_index < shape.value_dim; ++value_index) {
            float sum = 0.0f;
            for (int key_index = 0; key_index < shape.key_dim; ++key_index) {
                sum += k_head[key_index] * *state->at(h, key_index, value_index);
            }
            memory[value_index] = sum;
        }

        for (int key_index = 0; key_index < shape.key_dim; ++key_index) {
            for (int value_index = 0; value_index < shape.value_dim; ++value_index) {
                const float delta = beta[h] * (v_head[value_index] - memory[value_index]);
                *state->at(h, key_index, value_index) += k_head[key_index] * delta;
            }
        }

        for (int value_index = 0; value_index < shape.value_dim; ++value_index) {
            float sum = 0.0f;
            for (int key_index = 0; key_index < shape.key_dim; ++key_index) {
                sum += q_head[key_index] * *state->at(h, key_index, value_index);
            }
            out_head[value_index] = sum;
        }
    }
}

// A readable batch=1 attention decode primitive. keys and values use the
// cache layout [tokens][kv_heads][head_dim]; query is [query_heads][head_dim].
void causal_attention_decode_cpu(const float* query, int query_heads, const float* keys, const float* values,
                                 int kv_heads, int tokens, int head_dim, float scale, float* out) {
    require(query_heads % kv_heads == 0, "GQA heads must divide exactly");
    std::vector<float> scores(tokens);
    const int queries_per_kv = query_heads / kv_heads;

    for (int h = 0; h < query_heads; ++h) {
        const int kv_head = h / queries_per_kv;
        const float* q = query + static_cast<size_t>(h) * head_dim;
        float maximum = -INFINITY;
        for (int t = 0; t < tokens; ++t) {
            const float* k = keys + (static_cast<size_t>(t) * kv_heads + kv_head) * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) score += q[d] * k[d];
            scores[t] = score * scale;
            maximum = std::max(maximum, scores[t]);
        }

        float denominator = 0.0f;
        for (int t = 0; t < tokens; ++t) {
            scores[t] = std::exp(scores[t] - maximum);
            denominator += scores[t];
        }

        float* o = out + static_cast<size_t>(h) * head_dim;
        std::fill(o, o + head_dim, 0.0f);
        for (int t = 0; t < tokens; ++t) {
            const float probability = scores[t] / denominator;
            const float* v = values + (static_cast<size_t>(t) * kv_heads + kv_head) * head_dim;
            for (int d = 0; d < head_dim; ++d) o[d] += probability * v[d];
        }
    }
}

// --------------------------------------------------------------------------
// Minimal safetensors reader
// --------------------------------------------------------------------------
//
// The official checkpoint is an index plus 18 safetensors shards.  A
// safetensors file starts with a little-endian u64 header length, then a JSON
// tensor table, followed by raw tensor bytes.  We only need that small,
// stable subset of JSON here, so an external JSON dependency would make the reference
// less readable without buying us anything.

class MappedFile {
  public:
    explicit MappedFile(const std::filesystem::path& path) : path_(path.string()) {
        fd_ = open(path_.c_str(), O_RDONLY);
        if (fd_ < 0) fail("cannot open " + path_ + ": " + std::strerror(errno));

        struct stat status {};
        if (fstat(fd_, &status) != 0) fail("cannot stat " + path_ + ": " + std::strerror(errno));
        if (status.st_size <= 0) fail("empty file: " + path_);
        size_ = static_cast<size_t>(status.st_size);

        void* mapped = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapped == MAP_FAILED) fail("cannot mmap " + path_ + ": " + std::strerror(errno));
        data_ = static_cast<const uint8_t*>(mapped);
    }

    ~MappedFile() {
        if (data_ != nullptr) munmap(const_cast<uint8_t*>(data_), size_);
        if (fd_ >= 0) close(fd_);
    }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }

  private:
    std::string path_;
    int fd_ = -1;
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
};

uint64_t read_u64_le(const uint8_t* bytes) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= static_cast<uint64_t>(bytes[i]) << (i * 8);
    return value;
}

struct TensorInfo {
    std::string name;
    std::string dtype;
    std::vector<uint64_t> shape;
    uint64_t data_begin = 0;  // Relative to the start of safetensors data, not the file.
    uint64_t data_end = 0;
};

class SafetensorsHeaderParser {
  public:
    explicit SafetensorsHeaderParser(const std::string& text) : text_(text) {}

    std::vector<TensorInfo> parse() {
        std::vector<TensorInfo> tensors;
        expect('{');
        if (consume('}')) {
            finish();
            return tensors;
        }
        while (true) {
            const std::string name = parse_string();
            expect(':');
            if (name == "__metadata__") {
                skip_value();
            } else {
                tensors.push_back(parse_tensor(name));
            }
            if (consume('}')) break;
            expect(',');
        }
        finish();
        return tensors;
    }

  private:
    const std::string& text_;
    size_t pos_ = 0;

    [[noreturn]] void error(const char* message) const {
        fail("invalid safetensors JSON near byte " + std::to_string(pos_) + ": " + message);
    }

    void skip_whitespace() {
        while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\n' || text_[pos_] == '\r' ||
                                       text_[pos_] == '\t')) {
            ++pos_;
        }
    }

    bool consume(char wanted) {
        skip_whitespace();
        if (pos_ < text_.size() && text_[pos_] == wanted) {
            ++pos_;
            return true;
        }
        return false;
    }

    void expect(char wanted) {
        if (!consume(wanted)) error("unexpected JSON token");
    }

    std::string parse_string() {
        skip_whitespace();
        if (pos_ >= text_.size() || text_[pos_] != '"') error("expected JSON string");
        ++pos_;
        std::string value;
        while (pos_ < text_.size()) {
            const char c = text_[pos_++];
            if (c == '"') return value;
            if (c != '\\') {
                value.push_back(c);
                continue;
            }
            if (pos_ >= text_.size()) error("unfinished JSON escape");
            const char escaped = text_[pos_++];
            switch (escaped) {
                case '"': value.push_back('"'); break;
                case '\\': value.push_back('\\'); break;
                case '/': value.push_back('/'); break;
                case 'b': value.push_back('\b'); break;
                case 'f': value.push_back('\f'); break;
                case 'n': value.push_back('\n'); break;
                case 'r': value.push_back('\r'); break;
                case 't': value.push_back('\t'); break;
                case 'u':
                    // Tensor names and safetensors fields are ASCII.  Still
                    // consume a valid unicode escape when skipping metadata.
                    if (pos_ + 4 > text_.size()) error("short unicode escape");
                    pos_ += 4;
                    value.push_back('?');
                    break;
                default: error("invalid JSON escape");
            }
        }
        error("unterminated JSON string");
    }

    uint64_t parse_uint() {
        skip_whitespace();
        if (pos_ >= text_.size() || text_[pos_] < '0' || text_[pos_] > '9') error("expected unsigned integer");
        uint64_t value = 0;
        while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
            const uint64_t digit = static_cast<uint64_t>(text_[pos_] - '0');
            if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10) error("integer overflow");
            value = value * 10 + digit;
            ++pos_;
        }
        return value;
    }

    std::vector<uint64_t> parse_uint_array() {
        std::vector<uint64_t> values;
        expect('[');
        if (consume(']')) return values;
        while (true) {
            values.push_back(parse_uint());
            if (consume(']')) break;
            expect(',');
        }
        return values;
    }

    void skip_value() {
        skip_whitespace();
        if (pos_ >= text_.size()) error("missing JSON value");
        if (text_[pos_] == '"') {
            parse_string();
            return;
        }
        if (text_[pos_] == '{') {
            ++pos_;
            if (consume('}')) return;
            while (true) {
                parse_string();
                expect(':');
                skip_value();
                if (consume('}')) return;
                expect(',');
            }
        }
        if (text_[pos_] == '[') {
            ++pos_;
            if (consume(']')) return;
            while (true) {
                skip_value();
                if (consume(']')) return;
                expect(',');
            }
        }
        // JSON number, true, false, or null.  Its exact value is irrelevant.
        const size_t begin = pos_;
        while (pos_ < text_.size() && text_[pos_] != ',' && text_[pos_] != '}' && text_[pos_] != ']' &&
               text_[pos_] != ' ' && text_[pos_] != '\n' && text_[pos_] != '\r' && text_[pos_] != '\t') {
            ++pos_;
        }
        if (pos_ == begin) error("invalid JSON value");
    }

    TensorInfo parse_tensor(const std::string& name) {
        TensorInfo tensor;
        tensor.name = name;
        bool has_dtype = false;
        bool has_shape = false;
        bool has_offsets = false;

        expect('{');
        if (consume('}')) error("empty tensor description");
        while (true) {
            const std::string field = parse_string();
            expect(':');
            if (field == "dtype") {
                tensor.dtype = parse_string();
                has_dtype = true;
            } else if (field == "shape") {
                tensor.shape = parse_uint_array();
                has_shape = true;
            } else if (field == "data_offsets") {
                const std::vector<uint64_t> offsets = parse_uint_array();
                if (offsets.size() != 2) error("data_offsets must have two integers");
                tensor.data_begin = offsets[0];
                tensor.data_end = offsets[1];
                has_offsets = true;
            } else {
                skip_value();
            }
            if (consume('}')) break;
            expect(',');
        }
        if (!has_dtype || !has_shape || !has_offsets || tensor.data_end < tensor.data_begin) {
            error("incomplete tensor description");
        }
        return tensor;
    }

    void finish() {
        skip_whitespace();
        if (pos_ != text_.size()) error("trailing JSON data");
    }
};

std::vector<TensorInfo> parse_safetensors_header(const uint8_t* bytes, size_t size, uint64_t* data_offset) {
    require(size >= 8, "safetensors file is shorter than its header prefix");
    const uint64_t header_size = read_u64_le(bytes);
    if (header_size > size - 8) fail("safetensors header extends past end of file");
    *data_offset = 8 + header_size;
    const std::string header(reinterpret_cast<const char*>(bytes + 8), static_cast<size_t>(header_size));
    return SafetensorsHeaderParser(header).parse();
}

bool starts_with(const std::string& value, const char* prefix) {
    const size_t prefix_size = std::strlen(prefix);
    return value.size() >= prefix_size && value.compare(0, prefix_size, prefix) == 0;
}

bool is_text_tensor(const std::string& name) {
    return name == "lm_head.weight" || starts_with(name, "model.language_model.");
}

const TensorInfo* find_tensor(const std::vector<TensorInfo>& tensors, const std::string& name) {
    for (const TensorInfo& tensor : tensors) {
        if (tensor.name == name) return &tensor;
    }
    return nullptr;
}

void require_tensor(const std::vector<TensorInfo>& tensors, const std::string& name,
                    std::vector<uint64_t> expected_shape, const char* expected_dtype = "BF16") {
    const TensorInfo* tensor = find_tensor(tensors, name);
    if (tensor == nullptr) fail("missing required Qwen text tensor: " + name);
    if (tensor->dtype != expected_dtype) fail(std::string("expected ") + expected_dtype + " tensor: " + name);
    if (tensor->shape != expected_shape) fail("unexpected shape for tensor: " + name);
}

void validate_text_schema(const std::vector<TensorInfo>& tensors) {
    require_tensor(tensors, "model.language_model.embed_tokens.weight", {Config::kVocabSize, Config::kHiddenSize});
    require_tensor(tensors, "model.language_model.norm.weight", {Config::kHiddenSize});
    if (!Config::kTiedWordEmbeddings) {
        require_tensor(tensors, "lm_head.weight", {Config::kVocabSize, Config::kHiddenSize});
    }

    for (int layer = 0; layer < Config::kLayers; ++layer) {
        const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
        require_tensor(tensors, prefix + "input_layernorm.weight", {Config::kHiddenSize});
        require_tensor(tensors, prefix + "post_attention_layernorm.weight", {Config::kHiddenSize});
        require_tensor(tensors, prefix + "mlp.gate_proj.weight", {Config::kIntermediateSize, Config::kHiddenSize});
        require_tensor(tensors, prefix + "mlp.up_proj.weight", {Config::kIntermediateSize, Config::kHiddenSize});
        require_tensor(tensors, prefix + "mlp.down_proj.weight", {Config::kHiddenSize, Config::kIntermediateSize});

        if (Config::is_deltanet_layer(layer)) {
            const std::string linear = prefix + "linear_attn.";
            require_tensor(tensors, linear + "A_log", {Config::kDeltaValueHeads}, Config::kDeltaAuxiliaryDtype);
            require_tensor(tensors, linear + "dt_bias", {Config::kDeltaValueHeads});
            require_tensor(tensors, linear + "conv1d.weight", {Config::kDeltaQkvSize, 1, Config::kDeltaConvKernel});
            require_tensor(tensors, linear + "in_proj_a.weight", {Config::kDeltaValueHeads, Config::kHiddenSize});
            require_tensor(tensors, linear + "in_proj_b.weight", {Config::kDeltaValueHeads, Config::kHiddenSize});
            require_tensor(tensors, linear + "in_proj_qkv.weight", {Config::kDeltaQkvSize, Config::kHiddenSize});
            require_tensor(tensors, linear + "in_proj_z.weight", {Config::kDeltaOutputSize, Config::kHiddenSize});
            require_tensor(tensors, linear + "norm.weight", {Config::kDeltaValueDim}, Config::kDeltaAuxiliaryDtype);
            require_tensor(tensors, linear + "out_proj.weight", {Config::kHiddenSize, Config::kDeltaOutputSize});
        } else {
            const std::string attention = prefix + "self_attn.";
            require_tensor(tensors, attention + "q_norm.weight", {Config::kAttentionHeadDim});
            require_tensor(tensors, attention + "k_norm.weight", {Config::kAttentionHeadDim});
            require_tensor(tensors, attention + "q_proj.weight", {Config::kAttentionQProjectionSize, Config::kHiddenSize});
            require_tensor(tensors, attention + "k_proj.weight", {Config::kAttentionKvSize, Config::kHiddenSize});
            require_tensor(tensors, attention + "v_proj.weight", {Config::kAttentionKvSize, Config::kHiddenSize});
            require_tensor(tensors, attention + "o_proj.weight", {Config::kHiddenSize, Config::kAttentionSize});
        }
    }
}

struct CheckpointStats {
    int shards = 0;
    int tensors = 0;
    int text_tensors = 0;
    int vision_tensors = 0;
    int mtp_tensors = 0;
    uint64_t text_bytes = 0;
    std::vector<TensorInfo> text_schema;
};

void inspect_safetensors_shard(const std::filesystem::path& path, CheckpointStats* stats) {
    MappedFile file(path);
    uint64_t data_offset = 0;
    const std::vector<TensorInfo> tensors = parse_safetensors_header(file.data(), file.size(), &data_offset);
    const uint64_t data_size = static_cast<uint64_t>(file.size()) - data_offset;

    ++stats->shards;
    for (const TensorInfo& tensor : tensors) {
        if (tensor.data_end > data_size) fail("tensor extends beyond shard data: " + tensor.name);
        ++stats->tensors;
        if (is_text_tensor(tensor.name)) {
            ++stats->text_tensors;
            stats->text_bytes += tensor.data_end - tensor.data_begin;
            stats->text_schema.push_back(tensor);
        } else if (starts_with(tensor.name, "model.visual.")) {
            ++stats->vision_tensors;
        } else if (starts_with(tensor.name, "mtp.")) {
            ++stats->mtp_tensors;
        }
    }
}

void inspect_checkpoint(const char* directory_name) {
    namespace fs = std::filesystem;
    const fs::path directory(directory_name);
    if (!fs::is_directory(directory)) fail("not a checkpoint directory: " + directory.string());

    std::vector<fs::path> shards;
    for (const fs::directory_entry& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        if (entry.path().extension() == ".safetensors") shards.push_back(entry.path());
    }
    std::sort(shards.begin(), shards.end());
    if (shards.empty()) fail("no .safetensors weight shards found");

    CheckpointStats stats;
    for (const fs::path& shard : shards) inspect_safetensors_shard(shard, &stats);
    validate_text_schema(stats.text_schema);
    require(stats.text_tensors == Config::kTextTensorCount, "unexpected Qwen text tensor count");

    const double text_gib = static_cast<double>(stats.text_bytes) / (1024.0 * 1024.0 * 1024.0);
    std::printf("checkpoint: %s\n", directory.c_str());
    std::printf("shards: %d, tensors: %d (text %d, vision %d, mtp %d)\n", stats.shards, stats.tensors,
                stats.text_tensors, stats.vision_tensors, stats.mtp_tensors);
    std::printf("text tensors: %.2f GiB\n", text_gib);
    std::printf("%s text schema: valid\n", Config::kModelName);
}

// --------------------------------------------------------------------------
// Qwen text model: mapped weights + scalar CPU forward
// --------------------------------------------------------------------------

struct Tensor {
    TensorInfo info;
    const uint8_t* data = nullptr;

    uint64_t elements() const {
        uint64_t result = 1;
        for (uint64_t dimension : info.shape) {
            if (dimension != 0 && result > std::numeric_limits<uint64_t>::max() / dimension) {
                fail("tensor element count overflow: " + info.name);
            }
            result *= dimension;
        }
        return result;
    }
};

uint16_t read_u16_le(const uint8_t* bytes) {
    return static_cast<uint16_t>(bytes[0]) | (static_cast<uint16_t>(bytes[1]) << 8);
}

inline float bf16_at(const Tensor& tensor, uint64_t element) {
    return bf16_to_f32(read_u16_le(tensor.data + element * 2));
}

// Official Qwen checkpoints deliberately retain DeltaNet's decay parameter
// and post-recurrence norm in FP32. All other model weights are BF16. Keeping
// this conversion at the tensor boundary makes that mixed-dtype fact explicit.
inline float scalar_at(const Tensor& tensor, uint64_t element) {
    if (tensor.info.dtype == "BF16") return bf16_at(tensor, element);
    if (tensor.info.dtype == "F32") return f32_at_le(tensor.data + element * 4);
    fail("unsupported scalar dtype: " + tensor.info.dtype + " in " + tensor.info.name);
}

class Checkpoint {
  public:
    explicit Checkpoint(const std::filesystem::path& directory) {
        namespace fs = std::filesystem;
        if (!fs::is_directory(directory)) fail("not a checkpoint directory: " + directory.string());

        std::vector<fs::path> shard_paths;
        for (const fs::directory_entry& entry : fs::directory_iterator(directory)) {
            if (!entry.is_regular_file()) continue;
            const std::string name = entry.path().filename().string();
            if (entry.path().extension() == ".safetensors") {
                shard_paths.push_back(entry.path());
            }
        }
        std::sort(shard_paths.begin(), shard_paths.end());
        if (shard_paths.empty()) fail("no model-*.safetensors shards found");

        for (const fs::path& path : shard_paths) add_shard(path);
    }

    Checkpoint(const Checkpoint&) = delete;
    Checkpoint& operator=(const Checkpoint&) = delete;

    const Tensor& tensor(const std::string& name) const {
        for (const Tensor& candidate : tensors_) {
            if (candidate.info.name == name) return candidate;
        }
        fail("missing tensor: " + name);
    }

    std::vector<TensorInfo> text_tensor_info() const {
        std::vector<TensorInfo> result;
        for (const Tensor& tensor : tensors_) {
            if (is_text_tensor(tensor.info.name)) result.push_back(tensor.info);
        }
        return result;
    }

  private:
    std::vector<std::unique_ptr<MappedFile>> files_;
    std::vector<Tensor> tensors_;

    void add_shard(const std::filesystem::path& path) {
        std::unique_ptr<MappedFile> file(new MappedFile(path));
        uint64_t data_offset = 0;
        const std::vector<TensorInfo> infos = parse_safetensors_header(file->data(), file->size(), &data_offset);
        const uint64_t data_size = static_cast<uint64_t>(file->size()) - data_offset;
        for (const TensorInfo& info : infos) {
            if (info.data_end > data_size) fail("tensor extends beyond shard data: " + info.name);
            Tensor tensor;
            tensor.info = info;
            tensor.data = file->data() + data_offset + info.data_begin;
            tensors_.push_back(std::move(tensor));
        }
        files_.push_back(std::move(file));
    }
};

void require_bf16_tensor(const Tensor& tensor, const char* role) {
    if (tensor.info.dtype != "BF16") fail(std::string(role) + " must be BF16: " + tensor.info.name);
}

void require_tensor_dtype(const Tensor& tensor, const char* role, const char* expected_dtype) {
    if (tensor.info.dtype != expected_dtype) {
        fail(std::string(role) + " must be " + expected_dtype + ": " + tensor.info.name);
    }
}

struct LinearWeight {
    const Tensor* weight = nullptr;  // [out_dim, in_dim], BF16
    int out_dim = 0;
    int in_dim = 0;
};

struct NormWeight {
    const Tensor* weight = nullptr;  // [size], BF16
    int size = 0;
};

struct MlpWeight {
    LinearWeight gate;
    LinearWeight up;
    LinearWeight down;
};

struct DeltaNetWeight {
    LinearWeight qkv;
    LinearWeight z;
    LinearWeight a;
    LinearWeight b;
    LinearWeight out;
    const Tensor* conv = nullptr;
    const Tensor* a_log = nullptr;
    const Tensor* dt_bias = nullptr;
    NormWeight norm;
};

struct AttentionWeight {
    LinearWeight q;
    LinearWeight k;
    LinearWeight v;
    LinearWeight out;
    NormWeight q_norm;
    NormWeight k_norm;
};

struct LayerWeight {
    bool is_deltanet = false;
    NormWeight input_norm;
    NormWeight post_attention_norm;
    MlpWeight mlp;
    DeltaNetWeight deltanet;
    AttentionWeight attention;
};

LinearWeight bind_linear(const Checkpoint& checkpoint, const std::string& name, int out_dim, int in_dim) {
    const Tensor& tensor = checkpoint.tensor(name);
    require_bf16_tensor(tensor, "linear weight");
    if (tensor.info.shape != std::vector<uint64_t>({static_cast<uint64_t>(out_dim), static_cast<uint64_t>(in_dim)})) {
        fail("unexpected linear shape: " + name);
    }
    return {&tensor, out_dim, in_dim};
}

NormWeight bind_norm(const Checkpoint& checkpoint, const std::string& name, int size, const char* dtype = "BF16") {
    const Tensor& tensor = checkpoint.tensor(name);
    require_tensor_dtype(tensor, "norm weight", dtype);
    if (tensor.info.shape != std::vector<uint64_t>({static_cast<uint64_t>(size)})) {
        fail("unexpected norm shape: " + name);
    }
    return {&tensor, size};
}

class TextModel {
  public:
    explicit TextModel(const std::filesystem::path& checkpoint_directory) : checkpoint_(checkpoint_directory) {
        const std::vector<TensorInfo> text_info = checkpoint_.text_tensor_info();
        validate_text_schema(text_info);
        require(text_info.size() == Config::kTextTensorCount, "unexpected Qwen text tensor count");

        embedding_ = &checkpoint_.tensor("model.language_model.embed_tokens.weight");
        lm_head_ = Config::kTiedWordEmbeddings ? embedding_ : &checkpoint_.tensor("lm_head.weight");
        final_norm_ = bind_norm(checkpoint_, "model.language_model.norm.weight", Config::kHiddenSize);

        layers_.resize(Config::kLayers);
        for (int layer = 0; layer < Config::kLayers; ++layer) bind_layer(layer, &layers_[layer]);
    }

    const Tensor& embedding() const { return *embedding_; }
    const Tensor& lm_head() const { return *lm_head_; }
    const NormWeight& final_norm() const { return final_norm_; }
    const LayerWeight& layer(int index) const { return layers_[index]; }

  private:
    Checkpoint checkpoint_;
    const Tensor* embedding_ = nullptr;
    const Tensor* lm_head_ = nullptr;
    NormWeight final_norm_;
    std::vector<LayerWeight> layers_;

    void bind_layer(int layer_index, LayerWeight* layer) {
        const std::string prefix = "model.language_model.layers." + std::to_string(layer_index) + ".";
        layer->is_deltanet = Config::is_deltanet_layer(layer_index);
        layer->input_norm = bind_norm(checkpoint_, prefix + "input_layernorm.weight", Config::kHiddenSize);
        layer->post_attention_norm = bind_norm(checkpoint_, prefix + "post_attention_layernorm.weight", Config::kHiddenSize);
        layer->mlp.gate = bind_linear(checkpoint_, prefix + "mlp.gate_proj.weight", Config::kIntermediateSize,
                                      Config::kHiddenSize);
        layer->mlp.up = bind_linear(checkpoint_, prefix + "mlp.up_proj.weight", Config::kIntermediateSize,
                                    Config::kHiddenSize);
        layer->mlp.down = bind_linear(checkpoint_, prefix + "mlp.down_proj.weight", Config::kHiddenSize,
                                      Config::kIntermediateSize);

        if (layer->is_deltanet) {
            const std::string linear = prefix + "linear_attn.";
            layer->deltanet.qkv = bind_linear(checkpoint_, linear + "in_proj_qkv.weight", Config::kDeltaQkvSize,
                                              Config::kHiddenSize);
            layer->deltanet.z = bind_linear(checkpoint_, linear + "in_proj_z.weight", Config::kDeltaOutputSize,
                                            Config::kHiddenSize);
            layer->deltanet.a = bind_linear(checkpoint_, linear + "in_proj_a.weight", Config::kDeltaValueHeads,
                                            Config::kHiddenSize);
            layer->deltanet.b = bind_linear(checkpoint_, linear + "in_proj_b.weight", Config::kDeltaValueHeads,
                                            Config::kHiddenSize);
            layer->deltanet.out = bind_linear(checkpoint_, linear + "out_proj.weight", Config::kHiddenSize,
                                              Config::kDeltaOutputSize);
            layer->deltanet.conv = &checkpoint_.tensor(linear + "conv1d.weight");
            layer->deltanet.a_log = &checkpoint_.tensor(linear + "A_log");
            layer->deltanet.dt_bias = &checkpoint_.tensor(linear + "dt_bias");
            layer->deltanet.norm = bind_norm(checkpoint_, linear + "norm.weight", Config::kDeltaValueDim,
                                             Config::kDeltaAuxiliaryDtype);
        } else {
            const std::string attention = prefix + "self_attn.";
            layer->attention.q = bind_linear(checkpoint_, attention + "q_proj.weight",
                                              Config::kAttentionQProjectionSize, Config::kHiddenSize);
            layer->attention.k = bind_linear(checkpoint_, attention + "k_proj.weight", Config::kAttentionKvSize,
                                              Config::kHiddenSize);
            layer->attention.v = bind_linear(checkpoint_, attention + "v_proj.weight", Config::kAttentionKvSize,
                                              Config::kHiddenSize);
            layer->attention.out = bind_linear(checkpoint_, attention + "o_proj.weight", Config::kHiddenSize,
                                                Config::kAttentionSize);
            layer->attention.q_norm = bind_norm(checkpoint_, attention + "q_norm.weight", Config::kAttentionHeadDim);
            layer->attention.k_norm = bind_norm(checkpoint_, attention + "k_norm.weight", Config::kAttentionHeadDim);
        }
    }
};

void matvec_bf16_cpu(const LinearWeight& linear, const float* x, float* out) {
    const Tensor& weight = *linear.weight;
    for (int row = 0; row < linear.out_dim; ++row) {
        float sum = 0.0f;
        const uint64_t row_offset = static_cast<uint64_t>(row) * linear.in_dim;
        for (int col = 0; col < linear.in_dim; ++col) sum += bf16_at(weight, row_offset + col) * x[col];
        out[row] = sum;
    }
}

void copy_bf16_row_cpu(const Tensor& tensor, int row, int row_size, float* out) {
    require(row >= 0 && static_cast<uint64_t>(row) < tensor.info.shape[0], "token id outside embedding vocabulary");
    const uint64_t offset = static_cast<uint64_t>(row) * row_size;
    for (int column = 0; column < row_size; ++column) out[column] = bf16_at(tensor, offset + column);
}

void rmsnorm_plus_bf16_cpu(const float* x, const NormWeight& norm, float* out) {
    float square_sum = 0.0f;
    for (int i = 0; i < norm.size; ++i) square_sum += x[i] * x[i];
    const float scale = 1.0f / std::sqrt(square_sum / norm.size + Config::kRmsNormEps);
    for (int i = 0; i < norm.size; ++i) out[i] = x[i] * scale * (1.0f + bf16_at(*norm.weight, i));
}

void rmsnorm_gated_bf16_cpu(const float* x, const NormWeight& norm, const float* z, float* out) {
    float square_sum = 0.0f;
    for (int i = 0; i < norm.size; ++i) square_sum += x[i] * x[i];
    const float scale = 1.0f / std::sqrt(square_sum / norm.size + Config::kRmsNormEps);
    for (int i = 0; i < norm.size; ++i) out[i] = x[i] * scale * scalar_at(*norm.weight, i) * silu(z[i]);
}

void depthwise_conv_bf16_step_cpu(const float* input, const Tensor& weight, int channels, int kernel_size,
                                  std::vector<float>* state, float* out) {
    const int history = kernel_size - 1;
    require(state->size() == static_cast<size_t>(channels) * history, "bad DeltaNet convolution state shape");
    for (int channel = 0; channel < channels; ++channel) {
        float* past = state->data() + static_cast<size_t>(channel) * history;
        float sum = 0.0f;
        const uint64_t offset = static_cast<uint64_t>(channel) * kernel_size;
        for (int i = 0; i < history; ++i) sum += past[i] * bf16_at(weight, offset + i);
        sum += input[channel] * bf16_at(weight, offset + history);
        for (int i = 0; i + 1 < history; ++i) past[i] = past[i + 1];
        past[history - 1] = input[channel];
        out[channel] = silu(sum);
    }
}

#ifdef QWEN38_TINY_MODEL
using CpuAttentionCacheElement = uint16_t;

float cpu_attention_cache_to_f32(CpuAttentionCacheElement value) { return bf16_to_f32(value); }
CpuAttentionCacheElement f32_to_cpu_attention_cache(float value) { return f32_to_bf16(value); }
#else
// The scalar CPU oracle deliberately retains attention K/V in FP32. This
// matches the official FP32 reference trace and prevents BF16 cache rounding
// from hiding mathematical mistakes behind accumulated decode error. GPU
// backends may use BF16/FP16 cache after they pass this oracle.
using CpuAttentionCacheElement = float;

float cpu_attention_cache_to_f32(CpuAttentionCacheElement value) { return value; }
CpuAttentionCacheElement f32_to_cpu_attention_cache(float value) { return value; }
#endif

struct AttentionCache {
    // Logical layout: [token][kv_head][head_dim]. See the type alias above
    // for why CPU reference and tiny CUDA-parity builds use different dtypes.
    std::vector<CpuAttentionCacheElement> keys;
    std::vector<CpuAttentionCacheElement> values;

    int tokens() const {
        const int token_size = Config::kKvHeads * Config::kAttentionHeadDim;
        require(keys.size() == values.size() && keys.size() % token_size == 0, "corrupt attention cache");
        return static_cast<int>(keys.size() / token_size);
    }

    void append(const float* key, const float* value) {
        const int token_size = Config::kKvHeads * Config::kAttentionHeadDim;
        const size_t previous_size = keys.size();
        keys.resize(previous_size + token_size);
        values.resize(previous_size + token_size);
        for (int i = 0; i < token_size; ++i) {
            keys[previous_size + i] = f32_to_cpu_attention_cache(key[i]);
            values[previous_size + i] = f32_to_cpu_attention_cache(value[i]);
        }
    }
};

void causal_attention_decode_cache_cpu(const float* query, const AttentionCache& cache, float* out) {
    const int tokens = cache.tokens();
    require(tokens > 0, "attention cache is empty after appending current token");
    const int query_heads = Config::kAttentionHeads;
    const int kv_heads = Config::kKvHeads;
    const int head_dim = Config::kAttentionHeadDim;
    const int queries_per_kv = query_heads / kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(tokens);

    for (int head = 0; head < query_heads; ++head) {
        const int kv_head = head / queries_per_kv;
        const float* q = query + static_cast<size_t>(head) * head_dim;
        float maximum = -INFINITY;
        for (int token = 0; token < tokens; ++token) {
            const CpuAttentionCacheElement* k =
                cache.keys.data() + (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
            float score = 0.0f;
            for (int dimension = 0; dimension < head_dim; ++dimension) {
                score += q[dimension] * cpu_attention_cache_to_f32(k[dimension]);
            }
            scores[token] = score * scale;
            maximum = std::max(maximum, scores[token]);
        }

        float denominator = 0.0f;
        for (int token = 0; token < tokens; ++token) {
            scores[token] = std::exp(scores[token] - maximum);
            denominator += scores[token];
        }

        float* o = out + static_cast<size_t>(head) * head_dim;
        std::fill(o, o + head_dim, 0.0f);
        for (int token = 0; token < tokens; ++token) {
            const float probability = scores[token] / denominator;
            const CpuAttentionCacheElement* v =
                cache.values.data() + (static_cast<size_t>(token) * kv_heads + kv_head) * head_dim;
            for (int dimension = 0; dimension < head_dim; ++dimension) {
                o[dimension] += probability * cpu_attention_cache_to_f32(v[dimension]);
            }
        }
    }
}

struct RuntimeState {
    int position = 0;
    std::vector<std::vector<float>> conv_states;
    std::vector<std::unique_ptr<DeltaNetState>> recurrent_states;
    std::vector<AttentionCache> attention_caches;

    RuntimeState() : conv_states(Config::kLayers), recurrent_states(Config::kLayers), attention_caches(Config::kLayers) {
        const DeltaNetShape shape = {Config::kDeltaValueHeads, Config::kDeltaKeyDim, Config::kDeltaValueDim};
        for (int layer = 0; layer < Config::kLayers; ++layer) {
            if (!Config::is_deltanet_layer(layer)) continue;
            conv_states[layer].assign(Config::kDeltaQkvSize * (Config::kDeltaConvKernel - 1), 0.0f);
            recurrent_states[layer].reset(new DeltaNetState(shape));
        }
    }
};

struct ForwardWorkspace {
    std::vector<float> hidden = std::vector<float>(Config::kHiddenSize);
    std::vector<float> normalized = std::vector<float>(Config::kHiddenSize);
    std::vector<float> mixer = std::vector<float>(Config::kHiddenSize);
    std::vector<float> mlp_gate = std::vector<float>(Config::kIntermediateSize);
    std::vector<float> mlp_up = std::vector<float>(Config::kIntermediateSize);

    std::vector<float> delta_qkv = std::vector<float>(Config::kDeltaQkvSize);
    std::vector<float> delta_z = std::vector<float>(Config::kDeltaOutputSize);
    std::vector<float> delta_a = std::vector<float>(Config::kDeltaValueHeads);
    std::vector<float> delta_b = std::vector<float>(Config::kDeltaValueHeads);
    std::vector<float> delta_q = std::vector<float>(Config::kDeltaOutputSize);
    std::vector<float> delta_k = std::vector<float>(Config::kDeltaOutputSize);
    std::vector<float> delta_log_decay = std::vector<float>(Config::kDeltaValueHeads);
    std::vector<float> delta_beta = std::vector<float>(Config::kDeltaValueHeads);
    std::vector<float> delta_out = std::vector<float>(Config::kDeltaOutputSize);

    // q_proj is stored per head as [Q(head_dim), gate(head_dim)], not as one
    // contiguous Q block followed by one contiguous gate block.
    std::vector<float> attention_q_projection = std::vector<float>(Config::kAttentionQProjectionSize);
    std::vector<float> attention_q = std::vector<float>(Config::kAttentionSize);
    std::vector<float> attention_gate = std::vector<float>(Config::kAttentionSize);
    std::vector<float> attention_k = std::vector<float>(Config::kAttentionKvSize);
    std::vector<float> attention_v = std::vector<float>(Config::kAttentionKvSize);
    std::vector<float> attention_out = std::vector<float>(Config::kAttentionSize);
    std::vector<float> rope_cos = std::vector<float>(Config::kRotaryDim);
    std::vector<float> rope_sin = std::vector<float>(Config::kRotaryDim);

    std::vector<float> logits = std::vector<float>(Config::kVocabSize);
};

// A trace is a development-only correctness artifact: one little-endian F32
// file per named point in the last token's forward pass. It gives the Python
// oracle a stable comparison surface without turning the forward path into a
// generic graph executor.
class TraceWriter {
  public:
    explicit TraceWriter(const std::filesystem::path& directory) : directory_(directory) {
        namespace fs = std::filesystem;
        if (fs::exists(directory_)) {
            require(fs::is_directory(directory_), "trace path is not a directory");
            require(fs::is_empty(directory_), "refusing to overwrite a non-empty trace directory");
        } else {
            fs::create_directories(directory_);
        }
    }

    void write_f32(const std::string& name, const float* values, size_t count) const {
        const std::filesystem::path path = directory_ / (name + ".f32");
        std::ofstream output(path, std::ios::binary);
        if (!output) fail("cannot create trace file: " + path.string());
        for (size_t index = 0; index < count; ++index) {
            uint32_t bits;
            std::memcpy(&bits, &values[index], sizeof(bits));
            const uint8_t bytes[4] = {static_cast<uint8_t>(bits), static_cast<uint8_t>(bits >> 8),
                                      static_cast<uint8_t>(bits >> 16), static_cast<uint8_t>(bits >> 24)};
            output.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
        }
        if (!output) fail("failed while writing trace file: " + path.string());
    }

    void write_manifest(const std::vector<int>& tokens) const {
        const std::filesystem::path path = directory_ / "manifest.txt";
        std::ofstream output(path);
        if (!output) fail("cannot create trace manifest: " + path.string());
        output << "qwen38-trace-v1\nmodel=" << Config::kModelName << "\ntokens=";
        for (size_t index = 0; index < tokens.size(); ++index) {
            if (index) output << ',';
            output << tokens[index];
        }
        output << "\nhidden_size=" << Config::kHiddenSize << "\nvocab_size=" << Config::kVocabSize << '\n';
        if (!output) fail("failed while writing trace manifest: " + path.string());
    }

  private:
    std::filesystem::path directory_;
};

void add_inplace(float* destination, const float* source, int size) {
    for (int i = 0; i < size; ++i) destination[i] += source[i];
}

void make_text_rope(int position, std::vector<float>* cos, std::vector<float>* sin) {
    const int half = Config::kRotaryDim / 2;
    for (int i = 0; i < half; ++i) {
        const float inv_frequency = 1.0f / std::pow(Config::kRopeTheta, static_cast<float>(2 * i) / Config::kRotaryDim);
        const float angle = position * inv_frequency;
        (*cos)[i] = (*cos)[i + half] = std::cos(angle);
        (*sin)[i] = (*sin)[i + half] = std::sin(angle);
    }
}

void split_attention_q_and_gate_cpu(const float* projected, int heads, int head_dim, float* query, float* gate) {
    for (int head = 0; head < heads; ++head) {
        const size_t projected_offset = static_cast<size_t>(head) * head_dim * 2;
        const size_t output_offset = static_cast<size_t>(head) * head_dim;
        std::memcpy(query + output_offset, projected + projected_offset, head_dim * sizeof(float));
        std::memcpy(gate + output_offset, projected + projected_offset + head_dim, head_dim * sizeof(float));
    }
}

void run_mlp_cpu(const MlpWeight& mlp, const float* input, ForwardWorkspace* workspace, float* out) {
    matvec_bf16_cpu(mlp.gate, input, workspace->mlp_gate.data());
    matvec_bf16_cpu(mlp.up, input, workspace->mlp_up.data());
    swiglu_cpu(workspace->mlp_gate.data(), workspace->mlp_up.data(), Config::kIntermediateSize,
               workspace->mlp_gate.data());
    matvec_bf16_cpu(mlp.down, workspace->mlp_gate.data(), out);
}

void run_deltanet_cpu(const DeltaNetWeight& weights, std::vector<float>* conv_state, DeltaNetState* recurrent_state,
                      const float* input, ForwardWorkspace* workspace, float* out) {
    matvec_bf16_cpu(weights.qkv, input, workspace->delta_qkv.data());
    matvec_bf16_cpu(weights.z, input, workspace->delta_z.data());
    matvec_bf16_cpu(weights.a, input, workspace->delta_a.data());
    matvec_bf16_cpu(weights.b, input, workspace->delta_b.data());
    depthwise_conv_bf16_step_cpu(workspace->delta_qkv.data(), *weights.conv, Config::kDeltaQkvSize,
                                 Config::kDeltaConvKernel,
                                 conv_state, workspace->delta_qkv.data());

    const float* q_small = workspace->delta_qkv.data();
    const float* k_small = q_small + Config::kDeltaKeyHeads * Config::kDeltaKeyDim;
    const float* value = k_small + Config::kDeltaKeyHeads * Config::kDeltaKeyDim;
    for (int head = 0; head < Config::kDeltaValueHeads; ++head) {
        const int qk_head = head / (Config::kDeltaValueHeads / Config::kDeltaKeyHeads);
        std::memcpy(workspace->delta_q.data() + static_cast<size_t>(head) * Config::kDeltaKeyDim,
                    q_small + static_cast<size_t>(qk_head) * Config::kDeltaKeyDim,
                    Config::kDeltaKeyDim * sizeof(float));
        std::memcpy(workspace->delta_k.data() + static_cast<size_t>(head) * Config::kDeltaKeyDim,
                    k_small + static_cast<size_t>(qk_head) * Config::kDeltaKeyDim,
                    Config::kDeltaKeyDim * sizeof(float));
        l2norm_inplace_cpu(workspace->delta_q.data() + static_cast<size_t>(head) * Config::kDeltaKeyDim,
                           Config::kDeltaKeyDim);
        l2norm_inplace_cpu(workspace->delta_k.data() + static_cast<size_t>(head) * Config::kDeltaKeyDim,
                           Config::kDeltaKeyDim);
        for (int dimension = 0; dimension < Config::kDeltaKeyDim; ++dimension) {
            workspace->delta_q[static_cast<size_t>(head) * Config::kDeltaKeyDim + dimension] /=
                std::sqrt(static_cast<float>(Config::kDeltaKeyDim));
        }
        workspace->delta_beta[head] = sigmoid(workspace->delta_b[head]);
        workspace->delta_log_decay[head] =
            -std::exp(scalar_at(*weights.a_log, head)) *
            softplus(workspace->delta_a[head] + scalar_at(*weights.dt_bias, head));
    }

    const DeltaNetShape shape = {Config::kDeltaValueHeads, Config::kDeltaKeyDim, Config::kDeltaValueDim};
    gated_delta_recurrence_cpu(shape, workspace->delta_q.data(), workspace->delta_k.data(), value,
                               workspace->delta_log_decay.data(), workspace->delta_beta.data(), recurrent_state,
                               workspace->delta_out.data());
    for (int head = 0; head < Config::kDeltaValueHeads; ++head) {
        const size_t offset = static_cast<size_t>(head) * Config::kDeltaValueDim;
        rmsnorm_gated_bf16_cpu(workspace->delta_out.data() + offset, weights.norm, workspace->delta_z.data() + offset,
                               workspace->delta_out.data() + offset);
    }
    matvec_bf16_cpu(weights.out, workspace->delta_out.data(), out);
}

void run_attention_cpu(const AttentionWeight& weights, AttentionCache* cache, int position, const float* input,
                       ForwardWorkspace* workspace, float* out) {
    matvec_bf16_cpu(weights.q, input, workspace->attention_q_projection.data());
    matvec_bf16_cpu(weights.k, input, workspace->attention_k.data());
    matvec_bf16_cpu(weights.v, input, workspace->attention_v.data());

    float* query = workspace->attention_q.data();
    float* gate = workspace->attention_gate.data();
    split_attention_q_and_gate_cpu(workspace->attention_q_projection.data(), Config::kAttentionHeads,
                                   Config::kAttentionHeadDim, query, gate);
    for (int head = 0; head < Config::kAttentionHeads; ++head) {
        rmsnorm_plus_bf16_cpu(query + static_cast<size_t>(head) * Config::kAttentionHeadDim, weights.q_norm,
                               query + static_cast<size_t>(head) * Config::kAttentionHeadDim);
    }
    for (int head = 0; head < Config::kKvHeads; ++head) {
        rmsnorm_plus_bf16_cpu(workspace->attention_k.data() + static_cast<size_t>(head) * Config::kAttentionHeadDim,
                               weights.k_norm,
                               workspace->attention_k.data() + static_cast<size_t>(head) * Config::kAttentionHeadDim);
    }

    make_text_rope(position, &workspace->rope_cos, &workspace->rope_sin);
    for (int head = 0; head < Config::kAttentionHeads; ++head) {
        rope_cpu(query + static_cast<size_t>(head) * Config::kAttentionHeadDim, Config::kAttentionHeadDim,
                 Config::kRotaryDim, workspace->rope_cos.data(), workspace->rope_sin.data());
    }
    for (int head = 0; head < Config::kKvHeads; ++head) {
        rope_cpu(workspace->attention_k.data() + static_cast<size_t>(head) * Config::kAttentionHeadDim,
                 Config::kAttentionHeadDim, Config::kRotaryDim, workspace->rope_cos.data(), workspace->rope_sin.data());
    }

    cache->append(workspace->attention_k.data(), workspace->attention_v.data());
    causal_attention_decode_cache_cpu(query, *cache, workspace->attention_out.data());
    for (int i = 0; i < Config::kAttentionSize; ++i) workspace->attention_out[i] *= sigmoid(gate[i]);
    matvec_bf16_cpu(weights.out, workspace->attention_out.data(), out);
}

void forward_token_cpu(const TextModel& model, RuntimeState* state, int token_id, ForwardWorkspace* workspace,
                       const TraceWriter* trace = nullptr) {
    require(token_id >= 0 && token_id < Config::kVocabSize, "token id outside model vocabulary");
    require(state->position < 262144, "the CPU reference does not implement context extension beyond 262144 tokens");
    copy_bf16_row_cpu(model.embedding(), token_id, Config::kHiddenSize, workspace->hidden.data());
    if (trace) trace->write_f32("embedding", workspace->hidden.data(), Config::kHiddenSize);

    for (int layer_index = 0; layer_index < Config::kLayers; ++layer_index) {
        const LayerWeight& layer = model.layer(layer_index);
        const std::string layer_name = trace ? "layers." + std::to_string(layer_index) + "." : "";
        rmsnorm_plus_bf16_cpu(workspace->hidden.data(), layer.input_norm, workspace->normalized.data());
        if (trace) trace->write_f32(layer_name + "input_norm", workspace->normalized.data(), Config::kHiddenSize);
        if (layer.is_deltanet) {
            run_deltanet_cpu(layer.deltanet, &state->conv_states[layer_index], state->recurrent_states[layer_index].get(),
                             workspace->normalized.data(), workspace, workspace->mixer.data());
        } else {
            run_attention_cpu(layer.attention, &state->attention_caches[layer_index], state->position,
                              workspace->normalized.data(), workspace, workspace->mixer.data());
        }
        if (trace) trace->write_f32(layer_name + "mixer", workspace->mixer.data(), Config::kHiddenSize);
        add_inplace(workspace->hidden.data(), workspace->mixer.data(), Config::kHiddenSize);
        if (trace) trace->write_f32(layer_name + "after_mixer_residual", workspace->hidden.data(), Config::kHiddenSize);

        rmsnorm_plus_bf16_cpu(workspace->hidden.data(), layer.post_attention_norm, workspace->normalized.data());
        if (trace) trace->write_f32(layer_name + "post_norm", workspace->normalized.data(), Config::kHiddenSize);
        run_mlp_cpu(layer.mlp, workspace->normalized.data(), workspace, workspace->mixer.data());
        if (trace) trace->write_f32(layer_name + "mlp", workspace->mixer.data(), Config::kHiddenSize);
        add_inplace(workspace->hidden.data(), workspace->mixer.data(), Config::kHiddenSize);
        if (trace) trace->write_f32(layer_name + "after_mlp_residual", workspace->hidden.data(), Config::kHiddenSize);
    }

    rmsnorm_plus_bf16_cpu(workspace->hidden.data(), model.final_norm(), workspace->normalized.data());
    if (trace) trace->write_f32("final_norm", workspace->normalized.data(), Config::kHiddenSize);
    const LinearWeight lm_head = {&model.lm_head(), Config::kVocabSize, Config::kHiddenSize};
    matvec_bf16_cpu(lm_head, workspace->normalized.data(), workspace->logits.data());
    if (trace) trace->write_f32("logits", workspace->logits.data(), Config::kVocabSize);
    ++state->position;
}

int argmax(const std::vector<float>& values) {
    require(!values.empty(), "argmax of empty vector");
    int best = 0;
    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] > values[best]) best = static_cast<int>(i);
    }
    return best;
}

std::vector<int> parse_token_ids(const char* input) {
    std::vector<int> tokens;
    uint64_t value = 0;
    bool in_number = false;
    for (const char* cursor = input;; ++cursor) {
        const char c = *cursor;
        if (c >= '0' && c <= '9') {
            if (value > (static_cast<uint64_t>(Config::kVocabSize) - 1) / 10) fail("token id is too large");
            value = value * 10 + static_cast<uint64_t>(c - '0');
            in_number = true;
            continue;
        }
        if (c == ',' || c == '\0') {
            if (!in_number) fail("expected comma-separated token ids");
            if (value >= Config::kVocabSize) fail("token id outside model vocabulary");
            tokens.push_back(static_cast<int>(value));
            value = 0;
            in_number = false;
            if (c == '\0') break;
            continue;
        }
        fail("token ids must be decimal numbers separated by commas");
    }
    return tokens;
}

// --------------------------------------------------------------------------
// Tiny fake checkpoint: end-to-end validation without 50 GiB of real weights
// --------------------------------------------------------------------------

struct FakeTensor {
    std::string name;
    std::vector<uint64_t> shape;
    std::vector<uint16_t> data;
};

uint64_t element_count(const std::vector<uint64_t>& shape) {
    uint64_t result = 1;
    for (uint64_t dimension : shape) {
        require(dimension != 0 && result <= std::numeric_limits<uint64_t>::max() / dimension,
                "fake tensor element count overflow");
        result *= dimension;
    }
    return result;
}

uint32_t stable_name_hash(const std::string& name) {
    uint32_t hash = 2166136261u;
    for (unsigned char character : name) hash = (hash ^ character) * 16777619u;
    return hash;
}

float fake_weight_value(const std::string& name, uint64_t index) {
    if (name.find("linear_attn.norm.weight") != std::string::npos) return 1.0f;
    if (name.find("A_log") != std::string::npos) return -1.0f;
    if (name.find("dt_bias") != std::string::npos) return 0.0f;
    if (name.find("layernorm.weight") != std::string::npos || name == "model.language_model.norm.weight" ||
        name.find("q_norm.weight") != std::string::npos || name.find("k_norm.weight") != std::string::npos) {
        return 0.0f;  // Qwen's ordinary norm stores its scale as 1 + weight.
    }
    const uint32_t bits = stable_name_hash(name) + static_cast<uint32_t>(index * 1103515245u);
    return static_cast<float>(static_cast<int>(bits % 2001u) - 1000) * 0.0005f;
}

void add_fake_tensor(std::vector<FakeTensor>* tensors, const std::string& name, std::vector<uint64_t> shape) {
    FakeTensor tensor;
    tensor.name = name;
    tensor.shape = std::move(shape);
    const uint64_t elements = element_count(tensor.shape);
    require(elements <= static_cast<uint64_t>(std::numeric_limits<size_t>::max()), "fake tensor is too large");
    tensor.data.resize(static_cast<size_t>(elements));
    for (uint64_t i = 0; i < elements; ++i) tensor.data[static_cast<size_t>(i)] = f32_to_bf16(fake_weight_value(name, i));
    tensors->push_back(std::move(tensor));
}

std::vector<FakeTensor> make_fake_text_tensors() {
    std::vector<FakeTensor> tensors;
    add_fake_tensor(&tensors, "model.language_model.embed_tokens.weight", {Config::kVocabSize, Config::kHiddenSize});
    add_fake_tensor(&tensors, "model.language_model.norm.weight", {Config::kHiddenSize});
    add_fake_tensor(&tensors, "lm_head.weight", {Config::kVocabSize, Config::kHiddenSize});

    for (int layer = 0; layer < Config::kLayers; ++layer) {
        const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
        add_fake_tensor(&tensors, prefix + "input_layernorm.weight", {Config::kHiddenSize});
        add_fake_tensor(&tensors, prefix + "post_attention_layernorm.weight", {Config::kHiddenSize});
        add_fake_tensor(&tensors, prefix + "mlp.gate_proj.weight", {Config::kIntermediateSize, Config::kHiddenSize});
        add_fake_tensor(&tensors, prefix + "mlp.up_proj.weight", {Config::kIntermediateSize, Config::kHiddenSize});
        add_fake_tensor(&tensors, prefix + "mlp.down_proj.weight", {Config::kHiddenSize, Config::kIntermediateSize});
        if (Config::is_deltanet_layer(layer)) {
            const std::string linear = prefix + "linear_attn.";
            add_fake_tensor(&tensors, linear + "A_log", {Config::kDeltaValueHeads});
            add_fake_tensor(&tensors, linear + "dt_bias", {Config::kDeltaValueHeads});
            add_fake_tensor(&tensors, linear + "conv1d.weight", {Config::kDeltaQkvSize, 1, Config::kDeltaConvKernel});
            add_fake_tensor(&tensors, linear + "in_proj_a.weight", {Config::kDeltaValueHeads, Config::kHiddenSize});
            add_fake_tensor(&tensors, linear + "in_proj_b.weight", {Config::kDeltaValueHeads, Config::kHiddenSize});
            add_fake_tensor(&tensors, linear + "in_proj_qkv.weight", {Config::kDeltaQkvSize, Config::kHiddenSize});
            add_fake_tensor(&tensors, linear + "in_proj_z.weight", {Config::kDeltaOutputSize, Config::kHiddenSize});
            add_fake_tensor(&tensors, linear + "norm.weight", {Config::kDeltaValueDim});
            add_fake_tensor(&tensors, linear + "out_proj.weight", {Config::kHiddenSize, Config::kDeltaOutputSize});
        } else {
            const std::string attention = prefix + "self_attn.";
            add_fake_tensor(&tensors, attention + "q_norm.weight", {Config::kAttentionHeadDim});
            add_fake_tensor(&tensors, attention + "k_norm.weight", {Config::kAttentionHeadDim});
            add_fake_tensor(&tensors, attention + "q_proj.weight", {Config::kAttentionQProjectionSize, Config::kHiddenSize});
            add_fake_tensor(&tensors, attention + "k_proj.weight", {Config::kAttentionKvSize, Config::kHiddenSize});
            add_fake_tensor(&tensors, attention + "v_proj.weight", {Config::kAttentionKvSize, Config::kHiddenSize});
            add_fake_tensor(&tensors, attention + "o_proj.weight", {Config::kHiddenSize, Config::kAttentionSize});
        }
    }
    require(tensors.size() == Config::kTextTensorCount, "fake checkpoint tensor count is wrong");
    return tensors;
}

void write_u64_le(std::ofstream* file, uint64_t value) {
    uint8_t bytes[8];
    for (int i = 0; i < 8; ++i) bytes[i] = static_cast<uint8_t>(value >> (8 * i));
    file->write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

std::string fake_safetensors_header(const std::vector<FakeTensor>& tensors) {
    std::string header = "{";
    uint64_t offset = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const FakeTensor& tensor = tensors[i];
        if (i) header += ',';
        const uint64_t bytes = static_cast<uint64_t>(tensor.data.size()) * sizeof(uint16_t);
        header += '\"' + tensor.name + "\":{\"dtype\":\"BF16\",\"shape\":[";
        for (size_t dimension = 0; dimension < tensor.shape.size(); ++dimension) {
            if (dimension) header += ',';
            header += std::to_string(tensor.shape[dimension]);
        }
        header += "],\"data_offsets\":[" + std::to_string(offset) + ',' + std::to_string(offset + bytes) + "]}";
        offset += bytes;
    }
    header += '}';
    return header;
}

void make_fake_checkpoint(const char* directory_name) {
#ifndef QWEN38_TINY_MODEL
    (void)directory_name;
    fail("--make-fake is available only in the qwen38_tiny build");
#else
    namespace fs = std::filesystem;
    const fs::path directory(directory_name);
    if (fs::exists(directory)) {
        require(fs::is_directory(directory), "fake checkpoint path is not a directory");
        require(fs::is_empty(directory), "refusing to write fake checkpoint into a non-empty directory");
    } else {
        fs::create_directories(directory);
    }
    const fs::path path = directory / "model-00001-of-00001.safetensors";
    const std::vector<FakeTensor> tensors = make_fake_text_tensors();
    const std::string header = fake_safetensors_header(tensors);
    std::ofstream file(path, std::ios::binary);
    if (!file) fail("cannot create fake checkpoint: " + path.string());
    write_u64_le(&file, header.size());
    file.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const FakeTensor& tensor : tensors) {
        for (uint16_t value : tensor.data) {
            const uint8_t bytes[2] = {static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8)};
            file.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
        }
    }
    if (!file) fail("failed while writing fake checkpoint: " + path.string());
    std::printf("fake checkpoint: %s (%zu tensors, %.1f KiB)\n", path.c_str(), tensors.size(),
                static_cast<double>(file.tellp()) / 1024.0);
#endif
}

void run_forward_tokens(const char* checkpoint_directory, const std::vector<int>& tokens) {
    require(!tokens.empty(), "forward requires at least one input token");
    std::puts("loading text weights with mmap (this maps the checkpoint but does not copy all weights into RAM)...");
    std::fflush(stdout);
    TextModel model(checkpoint_directory);
    RuntimeState state;
    ForwardWorkspace workspace;
    for (int token : tokens) forward_token_cpu(model, &state, token, &workspace);
    const int next_token = argmax(workspace.logits);
    std::printf("forward: %d input token(s), next token id %d, logit %.6f\n", state.position, next_token,
                workspace.logits[next_token]);
}

void run_forward(const char* checkpoint_directory, const char* token_string) {
    run_forward_tokens(checkpoint_directory, parse_token_ids(token_string));
}

void run_trace(const char* checkpoint_directory, const char* token_string, const char* output_directory) {
    const std::vector<int> tokens = parse_token_ids(token_string);
    require(!tokens.empty(), "trace requires at least one input token");
    std::puts("loading text weights with mmap (this maps the checkpoint but does not copy all weights into RAM)...");
    std::fflush(stdout);
    TextModel model(checkpoint_directory);
    RuntimeState state;
    ForwardWorkspace workspace;
    TraceWriter trace(output_directory);
    for (size_t index = 0; index < tokens.size(); ++index) {
        forward_token_cpu(model, &state, tokens[index], &workspace, index + 1 == tokens.size() ? &trace : nullptr);
    }
    trace.write_manifest(tokens);
    std::printf("trace: %zu input token(s), final-token tensors written to %s\n", tokens.size(), output_directory);
}

int parse_positive_int(const char* input, const char* option_name) {
    if (!input || !input[0]) fail(std::string(option_name) + " requires a positive integer");
    uint64_t value = 0;
    for (const char* cursor = input; *cursor; ++cursor) {
        if (*cursor < '0' || *cursor > '9') fail(std::string(option_name) + " requires a positive integer");
        value = value * 10 + static_cast<uint64_t>(*cursor - '0');
        if (value > 1000000) fail(std::string(option_name) + " is unreasonably large");
    }
    if (value == 0) fail(std::string(option_name) + " requires a positive integer");
    return static_cast<int>(value);
}

struct SamplingOptions {
    float temperature = 0.0f;  // 0 is deliberately greedy and deterministic.
    int top_k = 0;             // 0 means the whole vocabulary.
    float top_p = 1.0f;
    uint64_t seed = 1;
};

float parse_probability(const char* input, const char* option_name, bool allow_zero) {
    char* end = nullptr;
    const float value = std::strtof(input, &end);
    if (!end || *end != '\0' || !std::isfinite(value) || value > 1.0f || (allow_zero ? value < 0.0f : value <= 0.0f)) {
        fail(std::string(option_name) + " must be in (0, 1]");
    }
    return value;
}

float parse_nonnegative_float(const char* input, const char* option_name) {
    char* end = nullptr;
    const float value = std::strtof(input, &end);
    if (!end || *end != '\0' || !std::isfinite(value) || value < 0.0f) {
        fail(std::string(option_name) + " requires a non-negative finite number");
    }
    return value;
}

uint64_t parse_u64(const char* input, const char* option_name) {
    if (!input || !input[0]) fail(std::string(option_name) + " requires an integer");
    uint64_t value = 0;
    for (const char* cursor = input; *cursor; ++cursor) {
        if (*cursor < '0' || *cursor > '9') fail(std::string(option_name) + " requires an integer");
        if (value > (std::numeric_limits<uint64_t>::max() - static_cast<uint64_t>(*cursor - '0')) / 10) {
            fail(std::string(option_name) + " is too large");
        }
        value = value * 10 + static_cast<uint64_t>(*cursor - '0');
    }
    return value;
}

SamplingOptions parse_sampling_options(int argc, char** argv, int first_option) {
    SamplingOptions options;
    for (int index = first_option; index < argc; ++index) {
        const char* option = argv[index];
        if (std::strcmp(option, "--temperature") == 0) {
            if (++index == argc) fail("--temperature requires a value");
            options.temperature = parse_nonnegative_float(argv[index], "--temperature");
        } else if (std::strcmp(option, "--top-k") == 0) {
            if (++index == argc) fail("--top-k requires a value");
            options.top_k = parse_positive_int(argv[index], "--top-k");
        } else if (std::strcmp(option, "--top-p") == 0) {
            if (++index == argc) fail("--top-p requires a value");
            options.top_p = parse_probability(argv[index], "--top-p", false);
        } else if (std::strcmp(option, "--seed") == 0) {
            if (++index == argc) fail("--seed requires a value");
            options.seed = parse_u64(argv[index], "--seed");
        } else {
            fail("unknown sampling option: " + std::string(option));
        }
    }
    return options;
}

uint64_t next_random_u64(uint64_t* state) {
    // splitmix64: small, deterministic, and sufficient for educational sampling.
    *state += 0x9e3779b97f4a7c15ull;
    uint64_t value = *state;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

int sample_token(const std::vector<float>& logits, const SamplingOptions& options, uint64_t* random_state) {
    if (options.temperature == 0.0f) return argmax(logits);

    std::vector<int> candidates(logits.size());
    for (int i = 0; i < static_cast<int>(candidates.size()); ++i) candidates[i] = i;
    const auto greater_logit = [&logits](int left, int right) { return logits[left] > logits[right]; };
    const int top_k = options.top_k == 0 ? static_cast<int>(candidates.size()) : std::min(options.top_k, static_cast<int>(candidates.size()));
    if (top_k < static_cast<int>(candidates.size())) {
        std::nth_element(candidates.begin(), candidates.begin() + top_k, candidates.end(), greater_logit);
        candidates.resize(top_k);
    }
    std::sort(candidates.begin(), candidates.end(), greater_logit);

    const float maximum = logits[candidates.front()] / options.temperature;
    std::vector<float> probabilities(candidates.size());
    float total = 0.0f;
    for (size_t i = 0; i < candidates.size(); ++i) {
        probabilities[i] = std::exp(logits[candidates[i]] / options.temperature - maximum);
        total += probabilities[i];
    }
    float kept_total = 0.0f;
    size_t kept = 0;
    for (; kept < probabilities.size(); ++kept) {
        kept_total += probabilities[kept];
        if (kept_total / total >= options.top_p) {
            ++kept;
            break;
        }
    }
    const float draw = static_cast<float>((next_random_u64(random_state) >> 40) * (1.0 / 16777216.0)) * kept_total;
    float cumulative = 0.0f;
    for (size_t i = 0; i < kept; ++i) {
        cumulative += probabilities[i];
        if (draw < cumulative) return candidates[i];
    }
    return candidates[kept - 1];
}

// Sampling is deliberately separate from prefill/decode: changing a sampling
// strategy never changes the model execution or cache update path.
std::vector<int> generate_tokens(const char* checkpoint_directory, const std::vector<int>& prompt, int new_tokens,
                                 const SamplingOptions& options, int stop_token = -1) {
    require(!prompt.empty(), "generation requires at least one prompt token");
    std::puts("loading text weights with mmap (this maps the checkpoint but does not copy all weights into RAM)...");
    std::fflush(stdout);
    TextModel model(checkpoint_directory);
    RuntimeState state;
    ForwardWorkspace workspace;

    for (int token : prompt) forward_token_cpu(model, &state, token, &workspace);  // prefill
    std::vector<int> generated;
    generated.reserve(new_tokens);
    uint64_t random_state = options.seed;
    for (int step = 0; step < new_tokens; ++step) {
        const int token = sample_token(workspace.logits, options, &random_state);
        if (token == stop_token) break;
        generated.push_back(token);
        if (step + 1 < new_tokens) forward_token_cpu(model, &state, token, &workspace);  // decode
    }
    std::printf("prefill: %zu token(s), generated: %zu/%d token(s), evaluated: %d token(s)\n", prompt.size(),
                generated.size(), new_tokens, state.position);
    return generated;
}

void run_generate(const char* checkpoint_directory, const char* token_string, int new_tokens, const SamplingOptions& options) {
    const std::vector<int> prompt = parse_token_ids(token_string);
    const std::vector<int> generated = generate_tokens(checkpoint_directory, prompt, new_tokens, options);
    std::printf("generated:");
    for (int token : generated) std::printf(" %d", token);
    std::putchar('\n');
}

#ifdef QWEN38_WITH_TOKENIZER
void print_token_ids(const std::vector<int>& token_ids) {
    std::printf("ids:");
    for (int token_id : token_ids) std::printf(" %d", token_id);
    std::putchar('\n');
}

void run_tokenize(const char* checkpoint_directory, const char* text) {
    QwenTokenizer tokenizer(checkpoint_directory);
    const std::vector<int> token_ids = tokenizer.encode(text ? text : "");
    std::printf("tokenize: %zu token(s)\n", token_ids.size());
    print_token_ids(token_ids);
    std::printf("decoded: %s\n", tokenizer.decode(token_ids).c_str());
}

void run_forward_text(const char* checkpoint_directory, const char* text) {
    QwenTokenizer tokenizer(checkpoint_directory);
    const std::vector<int> token_ids = tokenizer.encode(text ? text : "");
    std::printf("tokenize: %zu token(s)\n", token_ids.size());
    print_token_ids(token_ids);
    run_forward_tokens(checkpoint_directory, token_ids);
}

void run_generate_text(const char* checkpoint_directory, const char* text, int new_tokens, const SamplingOptions& options) {
    QwenTokenizer tokenizer(checkpoint_directory);
    const std::vector<int> prompt = tokenizer.encode(text ? text : "");
    const std::vector<int> generated = generate_tokens(checkpoint_directory, prompt, new_tokens, options);
    print_token_ids(generated);
    std::printf("generated text: %s\n", tokenizer.decode(generated).c_str());
}

// This is intentionally not a generic Jinja or chat-template runtime. It is
// the exact no-tools, one-user-turn shape from Qwen3.5's shipped template with
// thinking disabled. Multi-turn/tool/vision messages remain outside text M1.
std::string qwen_one_turn_chat_prompt(std::string_view user_text) {
    return "<|im_start|>user\n" + std::string(user_text) +
           "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
}

void run_chat_tokenize(const char* checkpoint_directory, const char* user_text) {
    QwenTokenizer tokenizer(checkpoint_directory);
    const std::vector<int> token_ids = tokenizer.encode(qwen_one_turn_chat_prompt(user_text ? user_text : ""));
    std::printf("chat tokenize: %zu token(s)\n", token_ids.size());
    print_token_ids(token_ids);
}

void run_generate_chat(const char* checkpoint_directory, const char* user_text, int new_tokens,
                       const SamplingOptions& options) {
    QwenTokenizer tokenizer(checkpoint_directory);
    const std::vector<int> prompt = tokenizer.encode(qwen_one_turn_chat_prompt(user_text ? user_text : ""));
    // The official simple-chat template terminates an assistant message with
    // <|im_end|>; do not return that control token as user-visible text.
    const std::vector<int> generated = generate_tokens(checkpoint_directory, prompt, new_tokens, options, 248046);
    print_token_ids(generated);
    std::printf("generated chat text: %s\n", tokenizer.decode(generated).c_str());
}
#endif

void test_config() {
    int deltanet_layers = 0;
    for (int layer = 0; layer < Config::kLayers; ++layer) {
        if (Config::is_deltanet_layer(layer)) ++deltanet_layers;
    }
    require(deltanet_layers == Config::kDeltaLayers, "Qwen hybrid DeltaNet layer count is wrong");
    require(!Config::is_deltanet_layer(3), "Qwen hybrid attention interval is wrong");
    if (Config::kLayers > 4) require(Config::is_deltanet_layer(4), "Qwen hybrid DeltaNet layer pattern is wrong");
}

void test_bfloat16() {
    const float original = 1.234375f;  // exactly representable in BF16
    require(nearly_equal(bf16_to_f32(f32_to_bf16(original)), original), "BF16 conversion failed");

    const uint16_t raw_weight[] = {f32_to_bf16(1.0f), f32_to_bf16(2.0f), f32_to_bf16(3.0f), f32_to_bf16(4.0f)};
    Tensor tensor;
    tensor.info.name = "test.weight";
    tensor.info.dtype = "BF16";
    tensor.info.shape = {2, 2};
    tensor.data = reinterpret_cast<const uint8_t*>(raw_weight);
    const LinearWeight linear = {&tensor, 2, 2};
    const float x[] = {5.0f, 6.0f};
    float out[2] = {};
    matvec_bf16_cpu(linear, x, out);
    require(nearly_equal(out[0], 17.0f) && nearly_equal(out[1], 39.0f), "BF16 GEMV failed");
}

void test_norms_and_swiglu() {
    const float x[] = {3.0f, 4.0f};
    const float zero_weight[] = {0.0f, 0.0f};
    float out[2] = {};
    rmsnorm_plus_cpu(x, zero_weight, 2, 0.0f, out);
    require(nearly_equal(out[0], 3.0f / std::sqrt(12.5f)), "ordinary RMSNorm value 0 failed");
    require(nearly_equal(out[1], 4.0f / std::sqrt(12.5f)), "ordinary RMSNorm value 1 failed");

    const float gate[] = {0.0f, 1.0f};
    const float up[] = {2.0f, 3.0f};
    swiglu_cpu(gate, up, 2, out);
    require(nearly_equal(out[0], 0.0f), "SwiGLU value 0 failed");
    require(nearly_equal(out[1], silu(1.0f) * 3.0f), "SwiGLU value 1 failed");
}

void test_rope_and_conv() {
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float cos[] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float sin[] = {1.0f, 1.0f, 1.0f, 1.0f};
    rope_cpu(x, 4, 4, cos, sin);
    require(nearly_equal(x[0], -3.0f) && nearly_equal(x[1], -4.0f) && nearly_equal(x[2], 1.0f) &&
                nearly_equal(x[3], 2.0f),
            "RoPE half rotation failed");

    const float input[] = {1.0f};
    const float weight[] = {0.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> state(3, 0.0f);
    float out[1] = {};
    depthwise_conv_step_cpu(input, weight, 1, 4, &state, out);
    require(nearly_equal(out[0], silu(1.0f)), "depthwise causal convolution failed");
    require(nearly_equal(state[2], 1.0f), "convolution history update failed");
}

void test_delta_recurrence() {
    const DeltaNetShape shape = {1, 2, 2};
    DeltaNetState state(shape);
    // q is already scaled; q/k are deliberately simple unit vectors here.
    const float q[] = {1.0f, 0.0f};
    const float k[] = {1.0f, 0.0f};
    const float value[] = {3.0f, -2.0f};
    const float log_decay[] = {-1.0f};
    const float beta[] = {0.5f};
    float out[2] = {};
    gated_delta_recurrence_cpu(shape, q, k, value, log_decay, beta, &state, out);
    require(nearly_equal(out[0], 1.5f) && nearly_equal(out[1], -1.0f), "DeltaNet first update failed");
}

void test_attention() {
    // Two query heads share one KV head.  Each query should favor the matching
    // cached key/value pair, demonstrating GQA mapping and FP32 softmax.
    const float query[] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float keys[] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float values[] = {10.0f, 0.0f, 0.0f, 20.0f};
    float out[4] = {};
    causal_attention_decode_cpu(query, 2, keys, values, 1, 2, 2, 1.0f, out);
    require(out[0] > 7.0f && out[1] > 5.0f && out[2] > 2.0f && out[3] > 14.0f,
            "causal GQA attention failed");

    const float projected[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float unpacked_q[4] = {};
    float unpacked_gate[4] = {};
    split_attention_q_and_gate_cpu(projected, 2, 2, unpacked_q, unpacked_gate);
    require(nearly_equal(unpacked_q[0], 1.0f) && nearly_equal(unpacked_q[1], 2.0f) &&
                nearly_equal(unpacked_q[2], 5.0f) && nearly_equal(unpacked_gate[0], 3.0f) &&
                nearly_equal(unpacked_gate[3], 8.0f),
            "attention Q/gate projection layout failed");

    std::vector<float> cache_key(Config::kKvHeads * Config::kAttentionHeadDim, 0.0f);
    std::vector<float> cache_value(Config::kKvHeads * Config::kAttentionHeadDim, 0.0f);
    AttentionCache cache;
    cache.append(cache_key.data(), cache_value.data());
    require(cache.tokens() == 1, "attention cache token accounting failed");
}

void test_safetensors_header() {
    const std::string header =
        R"({"__metadata__":{"format":"pt"},"scalar":{"dtype":"BF16","shape":[],"data_offsets":[0,2]},"matrix":{"dtype":"BF16","shape":[2,3],"data_offsets":[2,14]}})";
    const std::vector<TensorInfo> tensors = SafetensorsHeaderParser(header).parse();
    require(tensors.size() == 2, "safetensors parser tensor count failed");
    require(tensors[0].name == "scalar" && tensors[0].dtype == "BF16" && tensors[0].shape.empty() &&
                tensors[0].data_end == 2,
            "safetensors parser scalar failed");
    require(tensors[1].name == "matrix" && tensors[1].shape == std::vector<uint64_t>({2, 3}) &&
                tensors[1].data_begin == 2 && tensors[1].data_end == 14,
            "safetensors parser matrix failed");
}

void test_sampling() {
    const std::vector<float> logits = {1.0f, 3.0f, 2.0f};
    SamplingOptions greedy;
    uint64_t random_state = 1;
    require(sample_token(logits, greedy, &random_state) == 1, "greedy sampling failed");

    SamplingOptions top_one;
    top_one.temperature = 0.7f;
    top_one.top_k = 1;
    require(sample_token(logits, top_one, &random_state) == 1, "top-k sampling failed");
}

void run_self_test() {
    test_config();
    test_bfloat16();
    test_norms_and_swiglu();
    test_rope_and_conv();
    test_delta_recurrence();
    test_attention();
    test_safetensors_header();
    test_sampling();
    std::puts("self-test: passed (config, bf16, norm, SwiGLU, RoPE, conv, DeltaNet, GQA attention, safetensors, sampling)");
}

void print_description() {
    std::printf("%s text config: %d layers, hidden %d, %d DeltaNet + %d full attention layers\n",
                Config::kModelName, Config::kLayers, Config::kHiddenSize, Config::kDeltaLayers,
                Config::kAttentionLayers);
    std::printf("attention: Q=%d x %d, KV=%d x %d, RoPE=%d\n", Config::kAttentionHeads,
                Config::kAttentionHeadDim, Config::kKvHeads, Config::kAttentionHeadDim, Config::kRotaryDim);
    std::printf("DeltaNet: QK=%d x %d, V=%d x %d, conv=%d, recurrent state=FP32\n", Config::kDeltaKeyHeads,
                Config::kDeltaKeyDim, Config::kDeltaValueHeads, Config::kDeltaValueDim, Config::kDeltaConvKernel);
    std::printf("embedding/lm_head: %s\n", Config::kTiedWordEmbeddings ? "tied" : "untied");
}

void print_usage(const char* program) {
    std::printf("usage: %s [--self-test|--describe|--inspect <checkpoint-dir>|--forward <checkpoint-dir> <id,id,...>]\n",
                program);
    std::printf("       %s --trace <checkpoint-dir> <id,id,...> <empty-output-directory>\n", program);
    std::printf("       %s --generate <checkpoint-dir> <id,id,...> <new-token-count> [--temperature T --top-k K --top-p P --seed N]\n",
                program);
#ifdef QWEN38_TINY_MODEL
    std::printf("       %s --make-fake <empty-directory>\n", program);
#endif
#ifdef QWEN38_WITH_TOKENIZER
    std::printf("       %s [--tokenize <checkpoint-dir> <text>|--forward-text <checkpoint-dir> <text>]\n", program);
    std::printf("       %s --generate-text <checkpoint-dir> <text> <new-token-count> [--temperature T --top-k K --top-p P --seed N]\n",
                program);
    std::printf("       %s [--chat-tokenize <checkpoint-dir> <user-text>|--generate-chat <checkpoint-dir> <user-text> <new-token-count> [sampling options]]\n",
                program);
    std::puts("Tokenizer build: official tokenizer.json through the Hugging Face tokenizer C ABI.");
#else
    std::puts("This dependency-free build accepts token IDs first. Build with CMake to enable text tokenization.");
#endif
}

}  // namespace qwen38

#ifndef QWEN38_NO_MAIN
int main(int argc, char** argv) {
    if (argc == 1 || std::strcmp(argv[1], "--self-test") == 0) {
        qwen38::run_self_test();
        return 0;
    }
    if (std::strcmp(argv[1], "--describe") == 0) {
        qwen38::print_description();
        return 0;
    }
    if (std::strcmp(argv[1], "--inspect") == 0) {
        if (argc != 3) qwen38::fail("--inspect requires a checkpoint directory");
        qwen38::inspect_checkpoint(argv[2]);
        return 0;
    }
    if (std::strcmp(argv[1], "--forward") == 0) {
        if (argc != 4) qwen38::fail("--forward requires a checkpoint directory and comma-separated token ids");
        qwen38::run_forward(argv[2], argv[3]);
        return 0;
    }
    if (std::strcmp(argv[1], "--trace") == 0) {
        if (argc != 5) qwen38::fail("--trace requires a checkpoint directory, comma-separated token ids, and output directory");
        qwen38::run_trace(argv[2], argv[3], argv[4]);
        return 0;
    }
    if (std::strcmp(argv[1], "--generate") == 0) {
        if (argc < 5) qwen38::fail("--generate requires a checkpoint directory, token ids, and token count");
        qwen38::run_generate(argv[2], argv[3], qwen38::parse_positive_int(argv[4], "--generate"),
                              qwen38::parse_sampling_options(argc, argv, 5));
        return 0;
    }
    if (std::strcmp(argv[1], "--make-fake") == 0) {
        if (argc != 3) qwen38::fail("--make-fake requires an empty output directory");
        qwen38::make_fake_checkpoint(argv[2]);
        return 0;
    }
#ifdef QWEN38_WITH_TOKENIZER
    if (std::strcmp(argv[1], "--tokenize") == 0) {
        if (argc != 4) qwen38::fail("--tokenize requires a checkpoint directory and text");
        qwen38::run_tokenize(argv[2], argv[3]);
        return 0;
    }
    if (std::strcmp(argv[1], "--forward-text") == 0) {
        if (argc != 4) qwen38::fail("--forward-text requires a checkpoint directory and text");
        qwen38::run_forward_text(argv[2], argv[3]);
        return 0;
    }
    if (std::strcmp(argv[1], "--generate-text") == 0) {
        if (argc < 5) qwen38::fail("--generate-text requires a checkpoint directory, text, and token count");
        qwen38::run_generate_text(argv[2], argv[3], qwen38::parse_positive_int(argv[4], "--generate-text"),
                                   qwen38::parse_sampling_options(argc, argv, 5));
        return 0;
    }
    if (std::strcmp(argv[1], "--chat-tokenize") == 0) {
        if (argc != 4) qwen38::fail("--chat-tokenize requires a checkpoint directory and user text");
        qwen38::run_chat_tokenize(argv[2], argv[3]);
        return 0;
    }
    if (std::strcmp(argv[1], "--generate-chat") == 0) {
        if (argc < 5) qwen38::fail("--generate-chat requires a checkpoint directory, user text, and token count");
        qwen38::run_generate_chat(argv[2], argv[3], qwen38::parse_positive_int(argv[4], "--generate-chat"),
                                   qwen38::parse_sampling_options(argc, argv, 5));
        return 0;
    }
#endif
    qwen38::print_usage(argv[0]);
    return 1;
}
#endif

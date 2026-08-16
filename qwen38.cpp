// qwen38.cpp -- 小而直接的 Qwen3.8 CPU reference。
//
// 阅读顺序建议：先跳到 forward_token_cpu() 看完整的一次 token 前向，
// 再回来看本文件前半部分的数学、权重加载与状态定义。这里刻意没有 Tensor
// 框架、计算图或虚函数 backend：每个函数都应能直接对应一条模型公式。
// 后续 CUDA / Metal 优化会以这份易读 CPU 实现为 correctness oracle，而不是
// 取代它。

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

// 这里是固定模型的“编译期说明书”。默认值来自 Qwen/Qwen3.8-27B 的
// text_config；视觉编码器和 MTP head 有意不属于 text-only 执行路径。
// 这不是运行时可变的通用 Config：换模型应显式新增一个固定分支，避免让
// 可读的模型专用代码退化成抽象框架。
struct Config {
#ifdef QWEN38_TINY_MODEL
    // tiny 假权重保留 “3 个 DeltaNet : 1 个 attention” 的层模式，但缩小
    // 全部维度。它只用于 CPU/CUDA 端到端控制流测试，不具有语言能力。
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
    // 日常真实开发 checkpoint：Qwen/Qwen3.5-0.8B。它同样是固定分支，
    // 不是“任意模型配置系统”；用途是在普通开发机上跑真实 hybrid text 路径。
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
    // 每 4 层的最后一层走完整 attention；其余 3 层走 DeltaNet。
    static constexpr int kFullAttentionInterval = 4;
    // DeltaNet 的逐通道 causal convolution 使用长度为 4 的 kernel。
    static constexpr int kDeltaConvKernel = 4;

    // 以下是从基础超参数推导出的张量长度。保留名字而不是在代码里散落
    // “1024/2048/...” 等魔数，读 forward 时才能看出每段向量代表什么。
    static constexpr int kAttentionSize = kAttentionHeads * kAttentionHeadDim;
    static constexpr int kAttentionQProjectionSize = 2 * kAttentionSize;
    static constexpr int kAttentionKvSize = kKvHeads * kAttentionHeadDim;
    static constexpr int kDeltaQkSize = kDeltaKeyHeads * kDeltaKeyDim;
    static constexpr int kDeltaOutputSize = kDeltaValueHeads * kDeltaValueDim;
    static constexpr int kDeltaQkvSize = 2 * kDeltaQkSize + kDeltaOutputSize;

    // RMSNorm 的稳定项，以及 Qwen 使用的 RoPE base theta。
    static constexpr float kRmsNormEps = 1.0e-6f;
    static constexpr float kRopeTheta = 10000000.0f;

    static bool is_deltanet_layer(int layer) {
        // layer 3、7、11... 是 attention；0、1、2... 是 DeltaNet。
        return layer >= 0 && layer < kLayers && (layer % kFullAttentionInterval) != 3;
    }

    // 统计值既用于 --describe，也用于 safetensors schema 的完整性检查。
    static constexpr int kDeltaLayers = kLayers - kLayers / kFullAttentionInterval;
    static constexpr int kAttentionLayers = kLayers - kDeltaLayers;
    static constexpr int kTextTensorCount = (kTiedWordEmbeddings ? 2 : 3) + kDeltaLayers * 14 + kAttentionLayers * 11;
};

// 这些不变量若不成立，下面 GQA/DeltaNet 的“一个小头复制给多个 value head”
// 索引公式就会错误；因此在编译期立刻拒绝。
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
    // runtime 不尝试恢复坏 checkpoint/坏命令行；报出局部原因后立即退出，
    // 这样错误不会传播成“看似正常但数值错误”的生成结果。
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

// safetensors 里的主权重是 BF16。CPU reference 先按位将其扩展为 FP32，
// 再做标量运算：这是最容易检查数学正确性的精度路径。
float bf16_to_f32(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float f32_at_le(const uint8_t* bytes) {
    // safetensors 规定字节序为 little-endian；不能直接把任意 uint8_t* 强转
    // 成 float*，否则会有对齐/可移植性问题。
    const uint32_t bits = static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
                          (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    // 保留高 16 bit 前先做 round-to-nearest-even，与常见 BF16 转换一致。
    const uint32_t rounding_bias = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

float sigmoid(float x) {
    // 分开正负半轴，避免 exp(+很大数) 溢出。
    if (x >= 0.0f) {
        const float e = std::exp(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = std::exp(x);
    return e / (1.0f + e);
}

float silu(float x) {
    // SiLU(x) = x * sigmoid(x)，用于 SwiGLU、卷积和 DeltaNet 输出门。
    return x * sigmoid(x);
}

float softplus(float x) {
    // DeltaNet 的 a + dt_bias 会进入 softplus；大正数时直接返回 x，避免
    // 无意义地计算 exp(x) 并溢出。
    return x > 20.0f ? x : std::log1pf(std::exp(x));
}

// 行主序 W[out_dim, in_dim] 乘 x[in_dim]。这是故意朴素的 scalar GEMV：
// 慢，但每一项 sum += W[row,col]*x[col] 都可直接对照线性层公式，是以后
// BLAS/GPU 路径的数值 oracle。
void matvec_cpu(const float* weight, int out_dim, int in_dim, const float* x, float* out) {
    for (int row = 0; row < out_dim; ++row) {
        const float* w = weight + static_cast<size_t>(row) * in_dim;
        float sum = 0.0f;
        for (int col = 0; col < in_dim; ++col) sum += w[col] * x[col];
        out[row] = sum;
    }
}

// Qwen3.5/Qwen3.8 的普通 RMSNorm 权重保存的是“偏移量”：
// y = x / rms(x) * (1 + weight)，不是更常见的直接乘 weight。
void rmsnorm_plus_cpu(const float* x, const float* weight, int size, float eps, float* out) {
    float square_sum = 0.0f;
    for (int i = 0; i < size; ++i) square_sum += x[i] * x[i];
    const float scale = 1.0f / std::sqrt(square_sum / size + eps);
    for (int i = 0; i < size; ++i) out[i] = x[i] * scale * (1.0f + weight[i]);
}

// DeltaNet recurrence 之后的 norm 是另一种形式：直接乘 norm.weight，
// 再乘 SiLU(z) 输出门；不要和上面的 ordinary RMSNorm 混用。
void rmsnorm_gated_cpu(const float* x, const float* weight, const float* z, int size, float eps,
                       float* out) {
    float square_sum = 0.0f;
    for (int i = 0; i < size; ++i) square_sum += x[i] * x[i];
    const float scale = 1.0f / std::sqrt(square_sum / size + eps);
    for (int i = 0; i < size; ++i) out[i] = x[i] * scale * weight[i] * silu(z[i]);
}

void swiglu_cpu(const float* gate, const float* up, int size, float* out) {
    // FFN: down( SiLU(gate_proj(x)) * up_proj(x) )。
    for (int i = 0; i < size; ++i) out[i] = silu(gate[i]) * up[i];
}

void l2norm_inplace_cpu(float* x, int size, float eps = Config::kRmsNormEps) {
    // DeltaNet 的 q/k 是 L2 单位向量；q 之后还会额外乘 attention scale。
    float square_sum = 0.0f;
    for (int i = 0; i < size; ++i) square_sum += x[i] * x[i];
    const float scale = 1.0f / std::sqrt(square_sum + eps);
    for (int i = 0; i < size; ++i) x[i] *= scale;
}

// 对前 rotary_dim 个通道应用 Qwen 的 RoPE half-rotation。cos/sin 已按当前
// position 在 FP32 中算好；注意配对关系是 [0..half) 与 [half..rotary_dim)，
// 不是相邻元素两两配对。
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

// decode 时的 depthwise causal convolution。每个 channel 独立卷积；state
// 为该 channel 保存 kernel_size - 1 个“卷积前”的历史输入。
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
        // 滑动窗口左移，再把当前 input 写到最后一个历史槽。
        for (int i = 0; i + 1 < history; ++i) past[i] = past[i + 1];
        past[history - 1] = input[c];
        out[c] = silu(sum);
    }
}

struct DeltaNetShape {
    // 一个 DeltaNet state 矩阵的三个轴：[head][key_dim][value_dim]。
    int heads;
    int key_dim;
    int value_dim;
};

struct DeltaNetState {
    DeltaNetShape shape;
    // 逻辑布局 [head][key_dim][value_dim]。即使权重为 BF16，recurrent state
    // 在 CPU reference 中始终为 FP32，以减少跨 token 累积误差。
    std::vector<float> recurrent;

    explicit DeltaNetState(DeltaNetShape new_shape)
        : shape(new_shape),
          recurrent(static_cast<size_t>(new_shape.heads) * new_shape.key_dim * new_shape.value_dim, 0.0f) {}

    float* at(int head, int key_index, int value_index) {
        // 将三维逻辑坐标换成连续 vector 的地址；返回指针便于原地更新 S。
        const size_t offset = (static_cast<size_t>(head) * shape.key_dim + key_index) * shape.value_dim + value_index;
        return &recurrent[offset];
    }
};

// 逐字面实现 Gated DeltaNet recurrence。输入 q/k 已 L2 normalize，q 还应
// 已乘 1/sqrt(key_dim)。每个 head 的状态 S 是固定大小矩阵，不随 context
// 长度增长，这是它与 attention KV cache 最本质的区别。
void gated_delta_recurrence_cpu(const DeltaNetShape& shape, const float* q, const float* k,
                                const float* value, const float* log_decay, const float* beta,
                                DeltaNetState* state, float* out) {
    require(state->shape.heads == shape.heads && state->shape.key_dim == shape.key_dim &&
                state->shape.value_dim == shape.value_dim,
            "bad recurrent state shape");

    // memory = k^T S，长度仅为 value_dim，可被各 head 复用。
    std::vector<float> memory(shape.value_dim);
    for (int h = 0; h < shape.heads; ++h) {
        const float* q_head = q + static_cast<size_t>(h) * shape.key_dim;
        const float* k_head = k + static_cast<size_t>(h) * shape.key_dim;
        const float* v_head = value + static_cast<size_t>(h) * shape.value_dim;
        float* out_head = out + static_cast<size_t>(h) * shape.value_dim;
        const float decay = std::exp(log_decay[h]);

        // (1) 衰减旧状态：S <- exp(log_decay) * S。
        for (int key_index = 0; key_index < shape.key_dim; ++key_index) {
            for (int value_index = 0; value_index < shape.value_dim; ++value_index) {
                *state->at(h, key_index, value_index) *= decay;
            }
        }

        // (2) 用当前 k 从旧/已衰减状态读出 memory = k^T S。
        for (int value_index = 0; value_index < shape.value_dim; ++value_index) {
            float sum = 0.0f;
            for (int key_index = 0; key_index < shape.key_dim; ++key_index) {
                sum += k_head[key_index] * *state->at(h, key_index, value_index);
            }
            memory[value_index] = sum;
        }

        // (3) 以 gated delta 写回：S += k outer (beta * (v - memory))。
        for (int key_index = 0; key_index < shape.key_dim; ++key_index) {
            for (int value_index = 0; value_index < shape.value_dim; ++value_index) {
                const float delta = beta[h] * (v_head[value_index] - memory[value_index]);
                *state->at(h, key_index, value_index) += k_head[key_index] * delta;
            }
        }

        // (4) 当前 query 从更新后的状态读取 out = q^T S。
        for (int value_index = 0; value_index < shape.value_dim; ++value_index) {
            float sum = 0.0f;
            for (int key_index = 0; key_index < shape.key_dim; ++key_index) {
                sum += q_head[key_index] * *state->at(h, key_index, value_index);
            }
            out_head[value_index] = sum;
        }
    }
}

// 易读的 batch=1 attention decode primitive。keys/values 的 cache 布局为
// [tokens][kv_heads][head_dim]，query 为 [query_heads][head_dim]。
void causal_attention_decode_cpu(const float* query, int query_heads, const float* keys, const float* values,
                                 int kv_heads, int tokens, int head_dim, float scale, float* out) {
    require(query_heads % kv_heads == 0, "GQA heads must divide exactly");
    std::vector<float> scores(tokens);
    // GQA：连续 queries_per_kv 个 Q head 共用一个 KV head。
    const int queries_per_kv = query_heads / kv_heads;

    for (int h = 0; h < query_heads; ++h) {
        const int kv_head = h / queries_per_kv;
        const float* q = query + static_cast<size_t>(h) * head_dim;
        float maximum = -INFINITY;
        // 第一遍：计算缩放点积，并保留最大值供稳定 softmax 使用。
        for (int t = 0; t < tokens; ++t) {
            const float* k = keys + (static_cast<size_t>(t) * kv_heads + kv_head) * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) score += q[d] * k[d];
            scores[t] = score * scale;
            maximum = std::max(maximum, scores[t]);
        }

        // 第二遍：exp(score - max) 并求分母。
        float denominator = 0.0f;
        for (int t = 0; t < tokens; ++t) {
            scores[t] = std::exp(scores[t] - maximum);
            denominator += scores[t];
        }

        // 第三遍：概率加权所有历史 V，得到该 Q head 的输出。
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
// 最小 safetensors 读取器
// --------------------------------------------------------------------------
//
// 官方 checkpoint 是 index 加若干 safetensors shard。一个 safetensors 文件：
//   [8 字节 little-endian JSON 长度][JSON tensor 表][连续原始 tensor 字节]
// 本项目只需这个稳定的小子集，因此手写小 parser 比引入完整 JSON 依赖更容易
// 阅读。重点：这里解析的是“tensor 在文件哪里”，不是把 27B 权重复制进 RAM。

class MappedFile {
  public:
    explicit MappedFile(const std::filesystem::path& path) : path_(path.string()) {
        // open + fstat 先确认文件存在且非空；之后 mmap 建立只读虚拟映射。
        fd_ = open(path_.c_str(), O_RDONLY);
        if (fd_ < 0) fail("cannot open " + path_ + ": " + std::strerror(errno));

        struct stat status {};
        if (fstat(fd_, &status) != 0) fail("cannot stat " + path_ + ": " + std::strerror(errno));
        if (status.st_size <= 0) fail("empty file: " + path_);
        size_ = static_cast<size_t>(status.st_size);

        // MAP_PRIVATE 表示 runtime 永不修改 checkpoint；物理页按访问时加载。
        void* mapped = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapped == MAP_FAILED) fail("cannot mmap " + path_ + ": " + std::strerror(errno));
        data_ = static_cast<const uint8_t*>(mapped);
    }

    ~MappedFile() {
        // RAII：Checkpoint 析构时自动解除映射并关闭 fd。
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
    // safetensors 头部长度固定为 8 字节 little-endian 无符号整数。
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= static_cast<uint64_t>(bytes[i]) << (i * 8);
    return value;
}

struct TensorInfo {
    // JSON header 中单个 tensor 的元数据；data_begin/data_end 都相对“原始
    // tensor 数据区”起点，而不是相对整个文件起点。
    std::string name;
    std::string dtype;
    std::vector<uint64_t> shape;
    uint64_t data_begin = 0;
    uint64_t data_end = 0;
};

class SafetensorsHeaderParser {
  public:
    explicit SafetensorsHeaderParser(const std::string& text) : text_(text) {}

    std::vector<TensorInfo> parse() {
        // 顶层 JSON 是 { tensor_name: {dtype, shape, data_offsets}, ... }。
        // __metadata__ 对 forward 无用，直接跳过其完整 JSON value。
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
        // 给出 byte offset，坏 header 时定位比笼统的“JSON error”更容易。
        fail("invalid safetensors JSON near byte " + std::to_string(pos_) + ": " + message);
    }

    void skip_whitespace() {
        while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\n' || text_[pos_] == '\r' ||
                                       text_[pos_] == '\t')) {
            ++pos_;
        }
    }

    bool consume(char wanted) {
        // “如果当前正好是 wanted 则前进”是递归下降 parser 的基础操作。
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
                    // tensor 名和我们关心的字段均为 ASCII；仍需吞掉合法的
                    // unicode escape，才能正确跳过 metadata 中的任意内容。
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
        // shape 与 data_offsets 都是非负整数；这里同时显式防止 uint64 溢出。
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
        // 用于 shape:[...] 和 data_offsets:[begin,end] 两种字段。
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
        // 未知字段或 metadata 只需保证 JSON 结构被完整跳过，无需理解语义。
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
        // JSON number、true、false 或 null；对未知字段来说具体值无关紧要。
        const size_t begin = pos_;
        while (pos_ < text_.size() && text_[pos_] != ',' && text_[pos_] != '}' && text_[pos_] != ']' &&
               text_[pos_] != ' ' && text_[pos_] != '\n' && text_[pos_] != '\r' && text_[pos_] != '\t') {
            ++pos_;
        }
        if (pos_ == begin) error("invalid JSON value");
    }

    TensorInfo parse_tensor(const std::string& name) {
        // 只接受 runtime 必需的 dtype / shape / data_offsets 三项；任意
        // 额外字段可跳过，缺项或反向 offset 则拒绝该 checkpoint。
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
    // bytes[0..7] 是 header 长度；data_offset 返回原始 tensor 数据区起点。
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
    // Qwen 多模态 checkpoint 同时带 visual / MTP；text-only runtime 只取
    // language_model.* 与（若 untied）顶层 lm_head.weight。
    return name == "lm_head.weight" || starts_with(name, "model.language_model.");
}

const TensorInfo* find_tensor(const std::vector<TensorInfo>& tensors, const std::string& name) {
    // schema 检查阶段 tensor 数量不大，线性查找最直白；真正执行时会由
    // TextModel 把结果绑定为指针，不会在每个 token 内反复查找。
    for (const TensorInfo& tensor : tensors) {
        if (tensor.name == name) return &tensor;
    }
    return nullptr;
}

void require_tensor(const std::vector<TensorInfo>& tensors, const std::string& name,
                    std::vector<uint64_t> expected_shape, const char* expected_dtype = "BF16") {
    // 检查“名字、dtype、shape”三个维度，任何一个不符都不能继续 forward。
    const TensorInfo* tensor = find_tensor(tensors, name);
    if (tensor == nullptr) fail("missing required Qwen text tensor: " + name);
    if (tensor->dtype != expected_dtype) fail(std::string("expected ") + expected_dtype + " tensor: " + name);
    if (tensor->shape != expected_shape) fail("unexpected shape for tensor: " + name);
}

void validate_text_schema(const std::vector<TensorInfo>& tensors) {
    // 这份逐 tensor 的白名单是模型专用设计的核心：它既解释每层需要什么，
    // 也防止把相似但结构不同的 Qwen checkpoint 误当作目标模型。
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
            // DeltaNet 层：QKV + z/a/b 门控投影、卷积、FP32 辅助参数、out。
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
            // 完整 attention 层：Q/K/V/O 投影与每个 head 的 Q/K RMSNorm。
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
    // --inspect 只读 header 时收集的统计；用于报告 text/vision/MTP 的边界。
    int shards = 0;
    int tensors = 0;
    int text_tensors = 0;
    int vision_tensors = 0;
    int mtp_tensors = 0;
    uint64_t text_bytes = 0;
    std::vector<TensorInfo> text_schema;
};

void inspect_safetensors_shard(const std::filesystem::path& path, CheckpointStats* stats) {
    // 此函数没有构造 TextModel，也不执行 GEMV；它只扫描一个 shard 的 header。
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
            // 视觉权重被统计但不会纳入 text schema。
            ++stats->vision_tensors;
        } else if (starts_with(tensor.name, "mtp.")) {
            // MTP（multi-token prediction）同样被忽略，不影响普通 next token。
            ++stats->mtp_tensors;
        }
    }
}

void inspect_checkpoint(const char* directory_name) {
    // --inspect 的用户入口：遍历所有 shard，校验 schema，然后打印体积。
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
// Qwen text model：映射权重 + 标量 CPU 前向
// --------------------------------------------------------------------------

struct Tensor {
    // Tensor 不拥有 data；data 指向某个 MappedFile 内的原始字节。
    TensorInfo info;
    const uint8_t* data = nullptr;

    uint64_t elements() const {
        // shape 各维相乘，并防止恶意/损坏 header 导致乘法溢出。
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
    // BF16 每元素 2 字节；先按 little-endian 取 uint16，再扩展为 FP32。
    return bf16_to_f32(read_u16_le(tensor.data + element * 2));
}

// 官方 Qwen checkpoint 特意把 DeltaNet 衰减参数与 recurrence 后 norm 留为
// FP32，其他主权重大多是 BF16。把 mixed dtype 分支集中在 tensor 边界，
// 可避免前向公式里到处出现“这是 BF16 还是 F32”的噪声。
inline float scalar_at(const Tensor& tensor, uint64_t element) {
    if (tensor.info.dtype == "BF16") return bf16_at(tensor, element);
    if (tensor.info.dtype == "F32") return f32_at_le(tensor.data + element * 4);
    fail("unsupported scalar dtype: " + tensor.info.dtype + " in " + tensor.info.name);
}

class Checkpoint {
  public:
    explicit Checkpoint(const std::filesystem::path& directory) {
        // 读取目录中所有 .safetensors shard，按文件名排序以保证确定性。
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
        // 这是模型构造期的按名字查找，不在 token 热路径调用。
        for (const Tensor& candidate : tensors_) {
            if (candidate.info.name == name) return candidate;
        }
        fail("missing tensor: " + name);
    }

    std::vector<TensorInfo> text_tensor_info() const {
        // 先筛掉 vision/MTP，交给固定 text schema 进行严格验证。
        std::vector<TensorInfo> result;
        for (const Tensor& tensor : tensors_) {
            if (is_text_tensor(tensor.info.name)) result.push_back(tensor.info);
        }
        return result;
    }

  private:
    // files_ 必须比 tensors_ 活得久：后者的 data 指针借用前者的 mmap 区域。
    std::vector<std::unique_ptr<MappedFile>> files_;
    std::vector<Tensor> tensors_;

    void add_shard(const std::filesystem::path& path) {
        // 把 header 里的相对 offset 转换成实际内存地址；这里不复制权重。
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
    // 线性层 W 的形状为 [out_dim, in_dim]，主权重必须是 BF16。
    const Tensor* weight = nullptr;
    int out_dim = 0;
    int in_dim = 0;
};

struct NormWeight {
    // norm 向量是 [size]；DeltaNet gated norm 可例外为 F32。
    const Tensor* weight = nullptr;
    int size = 0;
};

struct MlpWeight {
    // SwiGLU FFN 的三个投影：gate/up 扩维，down 投回 hidden。
    LinearWeight gate;
    LinearWeight up;
    LinearWeight down;
};

struct DeltaNetWeight {
    // DeltaNet mixer 的全部“权重视图”。每项只是指向 checkpoint mmap 的指针。
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
    // attention mixer 的 Q/K/V/O 与 Q/K 两个 per-head norm。
    LinearWeight q;
    LinearWeight k;
    LinearWeight v;
    LinearWeight out;
    NormWeight q_norm;
    NormWeight k_norm;
};

struct LayerWeight {
    // 每层总是有两次 RMSNorm 和一个 MLP；mixer 二选一。
    bool is_deltanet = false;
    NormWeight input_norm;
    NormWeight post_attention_norm;
    MlpWeight mlp;
    DeltaNetWeight deltanet;
    AttentionWeight attention;
};

LinearWeight bind_linear(const Checkpoint& checkpoint, const std::string& name, int out_dim, int in_dim) {
    // 绑定时立刻检查形状，运行 forward 时才不需要猜张量布局。
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
        // 模型构造只做一次：完整 schema 校验 + 将每个名字绑定到明确字段。
        const std::vector<TensorInfo> text_info = checkpoint_.text_tensor_info();
        validate_text_schema(text_info);
        require(text_info.size() == Config::kTextTensorCount, "unexpected Qwen text tensor count");

        embedding_ = &checkpoint_.tensor("model.language_model.embed_tokens.weight");
        // 0.8B 使用 tied embedding/lm_head；27B 使用独立 lm_head。
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
        // 这一段是“官方 tensor 名称 → 可读的 LayerWeight 字段”的字典。
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
            // 注意 in_proj_qkv 的输出按 [small Q, small K, value] 拼接。
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
            // q_proj 的输出每个 head 按 [Q, gate] 排列，后续会显式 split。
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
    // 和前面的通用 matvec_cpu 同一公式，只是每次读 W 时即时 BF16→FP32。
    const Tensor& weight = *linear.weight;
    for (int row = 0; row < linear.out_dim; ++row) {
        float sum = 0.0f;
        const uint64_t row_offset = static_cast<uint64_t>(row) * linear.in_dim;
        for (int col = 0; col < linear.in_dim; ++col) sum += bf16_at(weight, row_offset + col) * x[col];
        out[row] = sum;
    }
}

void copy_bf16_row_cpu(const Tensor& tensor, int row, int row_size, float* out) {
    // embedding lookup：取 token_id 对应的一整行 [hidden]，并扩展到 FP32。
    require(row >= 0 && static_cast<uint64_t>(row) < tensor.info.shape[0], "token id outside embedding vocabulary");
    const uint64_t offset = static_cast<uint64_t>(row) * row_size;
    for (int column = 0; column < row_size; ++column) out[column] = bf16_at(tensor, offset + column);
}

void rmsnorm_plus_bf16_cpu(const float* x, const NormWeight& norm, float* out) {
    // 普通 norm 的权重是 BF16，公式仍是 x/rms(x) * (1+w)。
    float square_sum = 0.0f;
    for (int i = 0; i < norm.size; ++i) square_sum += x[i] * x[i];
    const float scale = 1.0f / std::sqrt(square_sum / norm.size + Config::kRmsNormEps);
    for (int i = 0; i < norm.size; ++i) out[i] = x[i] * scale * (1.0f + bf16_at(*norm.weight, i));
}

void rmsnorm_gated_bf16_cpu(const float* x, const NormWeight& norm, const float* z, float* out) {
    // gated DeltaNet norm：norm 可能是 BF16 或 F32，因此用 scalar_at。
    float square_sum = 0.0f;
    for (int i = 0; i < norm.size; ++i) square_sum += x[i] * x[i];
    const float scale = 1.0f / std::sqrt(square_sum / norm.size + Config::kRmsNormEps);
    for (int i = 0; i < norm.size; ++i) out[i] = x[i] * scale * scalar_at(*norm.weight, i) * silu(z[i]);
}

void depthwise_conv_bf16_step_cpu(const float* input, const Tensor& weight, int channels, int kernel_size,
                                  std::vector<float>* state, float* out) {
    // 与上面 float 版公式相同；区别仅在卷积权重从 mmap BF16 即时读取。
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
// tiny fixture 故意将 cache 写回 BF16，以覆盖量化/舍入存储这一控制流。
using CpuAttentionCacheElement = uint16_t;

float cpu_attention_cache_to_f32(CpuAttentionCacheElement value) { return bf16_to_f32(value); }
CpuAttentionCacheElement f32_to_cpu_attention_cache(float value) { return f32_to_bf16(value); }
#else
// 标量 CPU oracle 有意将 attention K/V 保持 FP32。这样才能与官方 FP32
// trace 对齐，不会让 BF16 cache 的累积舍入误差掩盖数学错误。GPU backend
// 在先通过这个 oracle 后，可以再选择 BF16/FP16 cache 优化显存。
using CpuAttentionCacheElement = float;

float cpu_attention_cache_to_f32(CpuAttentionCacheElement value) { return value; }
CpuAttentionCacheElement f32_to_cpu_attention_cache(float value) { return value; }
#endif

struct AttentionCache {
    // 逻辑布局 [token][kv_head][head_dim]。上面的 alias 决定底层元素是
    // FP32（真实 CPU oracle）还是 BF16（tiny fixture）。
    std::vector<CpuAttentionCacheElement> keys;
    std::vector<CpuAttentionCacheElement> values;

    int tokens() const {
        // 用连续数组长度反推 cache 中已有多少 token，并顺便检查一致性。
        const int token_size = Config::kKvHeads * Config::kAttentionHeadDim;
        require(keys.size() == values.size() && keys.size() % token_size == 0, "corrupt attention cache");
        return static_cast<int>(keys.size() / token_size);
    }

    void append(const float* key, const float* value) {
        // 每次 forward 当前 token 先写入 cache，随后它也可 attend 到自己。
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
    // 这是实际模型使用的 cache 版 attention；三遍 softmax 与前面的教学版
    // causal_attention_decode_cpu 相同，只是从 AttentionCache 取 K/V。
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
    // 唯一跨 token 存活的数据。position 是 RoPE 位置；三类容器分别服务
    // DeltaNet conv、DeltaNet recurrent matrix、完整 attention KV cache。
    int position = 0;
    std::vector<std::vector<float>> conv_states;
    std::vector<std::unique_ptr<DeltaNetState>> recurrent_states;
    std::vector<AttentionCache> attention_caches;

    RuntimeState() : conv_states(Config::kLayers), recurrent_states(Config::kLayers), attention_caches(Config::kLayers) {
        // 只有 DeltaNet 层分配 conv/recurrent；attention 层的 cache 按 token
        // 追加，因此起初为空。
        const DeltaNetShape shape = {Config::kDeltaValueHeads, Config::kDeltaKeyDim, Config::kDeltaValueDim};
        for (int layer = 0; layer < Config::kLayers; ++layer) {
            if (!Config::is_deltanet_layer(layer)) continue;
            conv_states[layer].assign(Config::kDeltaQkvSize * (Config::kDeltaConvKernel - 1), 0.0f);
            recurrent_states[layer].reset(new DeltaNetState(shape));
        }
    }
};

struct ForwardWorkspace {
    // 单个 forward 重复使用的临时向量，绝不跨 token 保存语义状态。
    // 预先分配可让核心函数专注于数学，而不用每层 new/delete。
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

    // q_proj 按 head 存储 [Q(head_dim), gate(head_dim)]，不是“所有 Q 后
    // 跟所有 gate”；attention 前会调用 split 函数拆成两个连续向量。
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

// trace 是开发期 correctness 工具：最后一个 token forward 的每个命名检查点
// 写成一个 little-endian F32 文件。Python 官方 reference 因此可逐层比较，
// 但 forward 本身仍保持直接的手写控制流，而非变成 computation graph。
class TraceWriter {
  public:
    explicit TraceWriter(const std::filesystem::path& directory) : directory_(directory) {
        // 避免误覆盖旧 trace；一次对比应对应一个干净目录。
        namespace fs = std::filesystem;
        if (fs::exists(directory_)) {
            require(fs::is_directory(directory_), "trace path is not a directory");
            require(fs::is_empty(directory_), "refusing to overwrite a non-empty trace directory");
        } else {
            fs::create_directories(directory_);
        }
    }

    void write_f32(const std::string& name, const float* values, size_t count) const {
        // 不依赖 NumPy 格式：每个 float 显式按 little-endian 写出，Python
        // 端可用 numpy.fromfile(..., dtype="<f4") 直接读取。
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
        // manifest 记录本次 trace 对应的 Config 与输入 token，方便复现。
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
    // residual connection：destination <- destination + source。
    for (int i = 0; i < size; ++i) destination[i] += source[i];
}

void make_text_rope(int position, std::vector<float>* cos, std::vector<float>* sin) {
    // 计算当前 position 的频率表；同一张表被该层所有 Q/K head 共享。
    const int half = Config::kRotaryDim / 2;
    for (int i = 0; i < half; ++i) {
        const float inv_frequency = 1.0f / std::pow(Config::kRopeTheta, static_cast<float>(2 * i) / Config::kRotaryDim);
        const float angle = position * inv_frequency;
        (*cos)[i] = (*cos)[i + half] = std::cos(angle);
        (*sin)[i] = (*sin)[i + half] = std::sin(angle);
    }
}

void split_attention_q_and_gate_cpu(const float* projected, int heads, int head_dim, float* query, float* gate) {
    // 将每 head [Q, gate] 的交错布局，变为两个连续 [heads, head_dim] 向量。
    for (int head = 0; head < heads; ++head) {
        const size_t projected_offset = static_cast<size_t>(head) * head_dim * 2;
        const size_t output_offset = static_cast<size_t>(head) * head_dim;
        std::memcpy(query + output_offset, projected + projected_offset, head_dim * sizeof(float));
        std::memcpy(gate + output_offset, projected + projected_offset + head_dim, head_dim * sizeof(float));
    }
}

void run_mlp_cpu(const MlpWeight& mlp, const float* input, ForwardWorkspace* workspace, float* out) {
    // 两个上投影可概念上并行；reference 为清晰而串行计算，然后 SwiGLU
    // 原地复用 mlp_gate 缓冲区，最后 down_proj 回到 hidden size。
    matvec_bf16_cpu(mlp.gate, input, workspace->mlp_gate.data());
    matvec_bf16_cpu(mlp.up, input, workspace->mlp_up.data());
    swiglu_cpu(workspace->mlp_gate.data(), workspace->mlp_up.data(), Config::kIntermediateSize,
               workspace->mlp_gate.data());
    matvec_bf16_cpu(mlp.down, workspace->mlp_gate.data(), out);
}

void run_deltanet_cpu(const DeltaNetWeight& weights, std::vector<float>* conv_state, DeltaNetState* recurrent_state,
                      const float* input, ForwardWorkspace* workspace, float* out) {
    // DeltaNet mixer 的执行顺序：投影 → causal conv → 准备 q/k/门控 →
    // recurrent update/read → gated norm → out_proj。
    matvec_bf16_cpu(weights.qkv, input, workspace->delta_qkv.data());
    matvec_bf16_cpu(weights.z, input, workspace->delta_z.data());
    matvec_bf16_cpu(weights.a, input, workspace->delta_a.data());
    matvec_bf16_cpu(weights.b, input, workspace->delta_b.data());
    depthwise_conv_bf16_step_cpu(workspace->delta_qkv.data(), *weights.conv, Config::kDeltaQkvSize,
                                 Config::kDeltaConvKernel,
                                 conv_state, workspace->delta_qkv.data());

    // in_proj_qkv 输出拼接为 [small Q][small K][V]。
    const float* q_small = workspace->delta_qkv.data();
    const float* k_small = q_small + Config::kDeltaKeyHeads * Config::kDeltaKeyDim;
    const float* value = k_small + Config::kDeltaKeyHeads * Config::kDeltaKeyDim;
    for (int head = 0; head < Config::kDeltaValueHeads; ++head) {
        // value head 数可能大于 key head 数；qk_head 表示当前 value head
        // 应复用哪一个小 Q/K head。
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
        // q 的额外 1/sqrt(d) 缩放与 attention 的缩放作用相同。
        for (int dimension = 0; dimension < Config::kDeltaKeyDim; ++dimension) {
            workspace->delta_q[static_cast<size_t>(head) * Config::kDeltaKeyDim + dimension] /=
                std::sqrt(static_cast<float>(Config::kDeltaKeyDim));
        }
        // beta 控制写入状态的幅度；log_decay 控制旧状态的遗忘速度。
        workspace->delta_beta[head] = sigmoid(workspace->delta_b[head]);
        workspace->delta_log_decay[head] =
            -std::exp(scalar_at(*weights.a_log, head)) *
            softplus(workspace->delta_a[head] + scalar_at(*weights.dt_bias, head));
    }

    // 真正的 recurrent state update；它是整个 DeltaNet 的“跨 token 记忆”。
    const DeltaNetShape shape = {Config::kDeltaValueHeads, Config::kDeltaKeyDim, Config::kDeltaValueDim};
    gated_delta_recurrence_cpu(shape, workspace->delta_q.data(), workspace->delta_k.data(), value,
                               workspace->delta_log_decay.data(), workspace->delta_beta.data(), recurrent_state,
                               workspace->delta_out.data());
    for (int head = 0; head < Config::kDeltaValueHeads; ++head) {
        // 每个 value head 独立做 gated RMSNorm，再拼回 delta_out。
        const size_t offset = static_cast<size_t>(head) * Config::kDeltaValueDim;
        rmsnorm_gated_bf16_cpu(workspace->delta_out.data() + offset, weights.norm, workspace->delta_z.data() + offset,
                               workspace->delta_out.data() + offset);
    }
    matvec_bf16_cpu(weights.out, workspace->delta_out.data(), out);
}

void run_attention_cpu(const AttentionWeight& weights, AttentionCache* cache, int position, const float* input,
                       ForwardWorkspace* workspace, float* out) {
    // attention mixer：Q/K/V 投影 → Q/K norm + RoPE → cache → softmax
    // attention → gate → O projection。
    matvec_bf16_cpu(weights.q, input, workspace->attention_q_projection.data());
    matvec_bf16_cpu(weights.k, input, workspace->attention_k.data());
    matvec_bf16_cpu(weights.v, input, workspace->attention_v.data());

    // q_proj 同时输出 query 与 sigmoid gate；K/V 不带这个 gate。
    float* query = workspace->attention_q.data();
    float* gate = workspace->attention_gate.data();
    split_attention_q_and_gate_cpu(workspace->attention_q_projection.data(), Config::kAttentionHeads,
                                   Config::kAttentionHeadDim, query, gate);
    for (int head = 0; head < Config::kAttentionHeads; ++head) {
        // q_norm 是每个 Q head 重复使用的 [head_dim] 参数。
        rmsnorm_plus_bf16_cpu(query + static_cast<size_t>(head) * Config::kAttentionHeadDim, weights.q_norm,
                               query + static_cast<size_t>(head) * Config::kAttentionHeadDim);
    }
    for (int head = 0; head < Config::kKvHeads; ++head) {
        // k_norm 的 head 数较少，因为 GQA 中 KV head 被多组 Q 共享。
        rmsnorm_plus_bf16_cpu(workspace->attention_k.data() + static_cast<size_t>(head) * Config::kAttentionHeadDim,
                               weights.k_norm,
                               workspace->attention_k.data() + static_cast<size_t>(head) * Config::kAttentionHeadDim);
    }

    // 同一 token position 的 cos/sin 同时作用于 Q 和 K，V 不旋转。
    make_text_rope(position, &workspace->rope_cos, &workspace->rope_sin);
    for (int head = 0; head < Config::kAttentionHeads; ++head) {
        rope_cpu(query + static_cast<size_t>(head) * Config::kAttentionHeadDim, Config::kAttentionHeadDim,
                 Config::kRotaryDim, workspace->rope_cos.data(), workspace->rope_sin.data());
    }
    for (int head = 0; head < Config::kKvHeads; ++head) {
        rope_cpu(workspace->attention_k.data() + static_cast<size_t>(head) * Config::kAttentionHeadDim,
                 Config::kAttentionHeadDim, Config::kRotaryDim, workspace->rope_cos.data(), workspace->rope_sin.data());
    }

    // 先 append 当前 K/V，再 decode，因此当前 token 可以 attend to 自己。
    cache->append(workspace->attention_k.data(), workspace->attention_v.data());
    causal_attention_decode_cache_cpu(query, *cache, workspace->attention_out.data());
    for (int i = 0; i < Config::kAttentionSize; ++i) workspace->attention_out[i] *= sigmoid(gate[i]);
    matvec_bf16_cpu(weights.out, workspace->attention_out.data(), out);
}

void forward_token_cpu(const TextModel& model, RuntimeState* state, int token_id, ForwardWorkspace* workspace,
                       const TraceWriter* trace = nullptr) {
    // 这是最值得反复阅读的函数：它就是“一次 token 的完整 Qwen text forward”。
    require(token_id >= 0 && token_id < Config::kVocabSize, "token id outside model vocabulary");
    require(state->position < 262144, "the CPU reference does not implement context extension beyond 262144 tokens");
    // 0) token ID → embedding，作为第 0 层的 residual stream / hidden。
    copy_bf16_row_cpu(model.embedding(), token_id, Config::kHiddenSize, workspace->hidden.data());
    if (trace) trace->write_f32("embedding", workspace->hidden.data(), Config::kHiddenSize);

    for (int layer_index = 0; layer_index < Config::kLayers; ++layer_index) {
        // 每层严格遵循：pre-norm → mixer → residual → post-norm → MLP → residual。
        const LayerWeight& layer = model.layer(layer_index);
        const std::string layer_name = trace ? "layers." + std::to_string(layer_index) + "." : "";
        rmsnorm_plus_bf16_cpu(workspace->hidden.data(), layer.input_norm, workspace->normalized.data());
        if (trace) trace->write_f32(layer_name + "input_norm", workspace->normalized.data(), Config::kHiddenSize);
        if (layer.is_deltanet) {
            // DeltaNet 状态只存在于这种层；attention cache 在这里不会动。
            run_deltanet_cpu(layer.deltanet, &state->conv_states[layer_index], state->recurrent_states[layer_index].get(),
                             workspace->normalized.data(), workspace, workspace->mixer.data());
        } else {
            // 完整 attention 层只更新该层的 KV cache；DeltaNet state 不会动。
            run_attention_cpu(layer.attention, &state->attention_caches[layer_index], state->position,
                              workspace->normalized.data(), workspace, workspace->mixer.data());
        }
        if (trace) trace->write_f32(layer_name + "mixer", workspace->mixer.data(), Config::kHiddenSize);
        // 第一次 residual：hidden <- hidden + mixer(hidden)。
        add_inplace(workspace->hidden.data(), workspace->mixer.data(), Config::kHiddenSize);
        if (trace) trace->write_f32(layer_name + "after_mixer_residual", workspace->hidden.data(), Config::kHiddenSize);

        // 第二个 pre-norm 只服务 FFN，输入是第一次 residual 后的 hidden。
        rmsnorm_plus_bf16_cpu(workspace->hidden.data(), layer.post_attention_norm, workspace->normalized.data());
        if (trace) trace->write_f32(layer_name + "post_norm", workspace->normalized.data(), Config::kHiddenSize);
        run_mlp_cpu(layer.mlp, workspace->normalized.data(), workspace, workspace->mixer.data());
        if (trace) trace->write_f32(layer_name + "mlp", workspace->mixer.data(), Config::kHiddenSize);
        // 第二次 residual：hidden <- hidden + MLP(hidden)。
        add_inplace(workspace->hidden.data(), workspace->mixer.data(), Config::kHiddenSize);
        if (trace) trace->write_f32(layer_name + "after_mlp_residual", workspace->hidden.data(), Config::kHiddenSize);
    }

    // 所有层结束：final norm 后用 lm_head 做词表大小 GEMV，得到 logits。
    rmsnorm_plus_bf16_cpu(workspace->hidden.data(), model.final_norm(), workspace->normalized.data());
    if (trace) trace->write_f32("final_norm", workspace->normalized.data(), Config::kHiddenSize);
    const LinearWeight lm_head = {&model.lm_head(), Config::kVocabSize, Config::kHiddenSize};
    matvec_bf16_cpu(lm_head, workspace->normalized.data(), workspace->logits.data());
    if (trace) trace->write_f32("logits", workspace->logits.data(), Config::kVocabSize);
    // 只有完整 forward 成功后才推进 position，下一 token 的 RoPE 会使用它。
    ++state->position;
}

int argmax(const std::vector<float>& values) {
    // greedy decode：返回 logits 最大的 vocabulary index。
    require(!values.empty(), "argmax of empty vector");
    int best = 0;
    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] > values[best]) best = static_cast<int>(i);
    }
    return best;
}

std::vector<int> parse_token_ids(const char* input) {
    // dependency-free CLI 使用 "12,34,56" 形式 token ID；逐字符解析可
    // 在读取模型前就报出格式/溢出错误。
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
// tiny 假 checkpoint：无需 50 GiB 真实权重的端到端测试
// --------------------------------------------------------------------------

struct FakeTensor {
    // 与真实 safetensors 的 TensorInfo 类似，但这里实际持有可写的 BF16 数据。
    std::string name;
    std::vector<uint64_t> shape;
    std::vector<uint16_t> data;
};

uint64_t element_count(const std::vector<uint64_t>& shape) {
    // fake checkpoint 同样检查 shape 乘积，避免测试辅助代码绕过安全边界。
    uint64_t result = 1;
    for (uint64_t dimension : shape) {
        require(dimension != 0 && result <= std::numeric_limits<uint64_t>::max() / dimension,
                "fake tensor element count overflow");
        result *= dimension;
    }
    return result;
}

uint32_t stable_name_hash(const std::string& name) {
    // FNV 风格的稳定 hash：同一个 tensor 名在每次运行得到同一组假权重。
    uint32_t hash = 2166136261u;
    for (unsigned char character : name) hash = (hash ^ character) * 16777619u;
    return hash;
}

float fake_weight_value(const std::string& name, uint64_t index) {
    // 假权重不追求语言能力；只要求数值有限、非退化，并让特殊参数符合
    // 公式约束（例如 ordinary norm 的 weight=0 对应乘以 1）。
    if (name.find("linear_attn.norm.weight") != std::string::npos) return 1.0f;
    if (name.find("A_log") != std::string::npos) return -1.0f;
    if (name.find("dt_bias") != std::string::npos) return 0.0f;
    if (name.find("layernorm.weight") != std::string::npos || name == "model.language_model.norm.weight" ||
        name.find("q_norm.weight") != std::string::npos || name.find("k_norm.weight") != std::string::npos) {
        return 0.0f;  // Qwen ordinary norm 实际使用 1 + weight。
    }
    const uint32_t bits = stable_name_hash(name) + static_cast<uint32_t>(index * 1103515245u);
    return static_cast<float>(static_cast<int>(bits % 2001u) - 1000) * 0.0005f;
}

void add_fake_tensor(std::vector<FakeTensor>* tensors, const std::string& name, std::vector<uint64_t> shape) {
    // 按真实 tensor 名/shape 生成确定性的 BF16 内容。
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
    // 这里故意重复 validate_text_schema 的结构：若模型 schema 改变，fake
    // fixture 也必须同步变化，避免测试运行了错误的旧结构。
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
    // safetensors 文件开头的 header 长度编码。
    uint8_t bytes[8];
    for (int i = 0; i < 8; ++i) bytes[i] = static_cast<uint8_t>(value >> (8 * i));
    file->write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

std::string fake_safetensors_header(const std::vector<FakeTensor>& tensors) {
    // 生成足够小的合法 safetensors JSON header；offset 累积的是 data 区字节。
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
    // 不覆盖已有目录内容，避免测试命令误伤用户的真实 checkpoint。
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
    // 文件布局严格遵循 [u64 header size][header JSON][raw BF16 bytes]。
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
    // CLI --forward 的最小入口：顺序处理所有输入 token，报告最后 logits 的
    // greedy next token。顺序调用本身就是 prefill。
    require(!tokens.empty(), "forward requires at least one input token");
    std::puts("loading text weights with mmap (this maps the checkpoint but does not copy all weights into RAM)...");
    std::fflush(stdout);
    // TextModel 建立 mmap 权重视图；state/workspace 随本次请求新建。
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
    // 与 run_forward_tokens 相同地 prefill，但只为“最后一个输入 token”写
    // trace；前面的 token 仍会更新 state，保证 decode 等价性。
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
    // CLI 中 token count、top-k 等使用的防溢出正整数 parser。
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
    // temperature=0 明确表示 greedy、可复现；top_k=0 表示不截断词表。
    float temperature = 0.0f;
    int top_k = 0;
    float top_p = 1.0f;
    uint64_t seed = 1;
};

float parse_probability(const char* input, const char* option_name, bool allow_zero) {
    // top-p 等概率参数的公共验证：拒绝 NaN、inf 和超出 [0,1] 的输入。
    char* end = nullptr;
    const float value = std::strtof(input, &end);
    if (!end || *end != '\0' || !std::isfinite(value) || value > 1.0f || (allow_zero ? value < 0.0f : value <= 0.0f)) {
        fail(std::string(option_name) + " must be in (0, 1]");
    }
    return value;
}

float parse_nonnegative_float(const char* input, const char* option_name) {
    // temperature 唯一允许 0，因为 0 是刻意定义的 greedy sentinel。
    char* end = nullptr;
    const float value = std::strtof(input, &end);
    if (!end || *end != '\0' || !std::isfinite(value) || value < 0.0f) {
        fail(std::string(option_name) + " requires a non-negative finite number");
    }
    return value;
}

uint64_t parse_u64(const char* input, const char* option_name) {
    // seed 使用完整 uint64，保证不同平台的采样序列可复现。
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
    // 生成参数刻意只解析本项目定义的少量 flag；未知参数立即报错。
    // 这避免为了一个教材项目引入通用 CLI 框架。
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
    // SplitMix64：小巧、可复现、统计性质足够用于推理时的采样。
    // 给定相同 seed、prompt 和权重，生成的 token 序列也应完全相同。
    *state += 0x9e3779b97f4a7c15ull;
    uint64_t value = *state;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

int sample_token(const std::vector<float>& logits, const SamplingOptions& options, uint64_t* random_state) {
    // temperature=0 是贪婪解码：直接取最大 logit，不涉及随机数。
    if (options.temperature == 0.0f) return argmax(logits);

    // 初始候选集是整个词表；top-k 会在下面先截断它。
    std::vector<int> candidates(logits.size());
    for (int i = 0; i < static_cast<int>(candidates.size()); ++i) candidates[i] = i;
    const auto greater_logit = [&logits](int left, int right) { return logits[left] > logits[right]; };
    const int top_k = options.top_k == 0 ? static_cast<int>(candidates.size()) : std::min(options.top_k, static_cast<int>(candidates.size()));
    if (top_k < static_cast<int>(candidates.size())) {
        // nth_element 不要求完整排序，适合只保留概率最高的 k 个 token。
        std::nth_element(candidates.begin(), candidates.begin() + top_k, candidates.end(), greater_logit);
        candidates.resize(top_k);
    }
    // top-p 需要从最高概率开始累积，因此此处再进行完整排序。
    std::sort(candidates.begin(), candidates.end(), greater_logit);

    // softmax(logits / temperature)。减 maximum 保证 exp 不会数值溢出。
    const float maximum = logits[candidates.front()] / options.temperature;
    std::vector<float> probabilities(candidates.size());
    float total = 0.0f;
    for (size_t i = 0; i < candidates.size(); ++i) {
        probabilities[i] = std::exp(logits[candidates[i]] / options.temperature - maximum);
        total += probabilities[i];
    }
    // nucleus / top-p：保留累计概率刚刚覆盖 p 的最小前缀。
    float kept_total = 0.0f;
    size_t kept = 0;
    for (; kept < probabilities.size(); ++kept) {
        kept_total += probabilities[kept];
        if (kept_total / total >= options.top_p) {
            ++kept;
            break;
        }
    }
    // 在保留下来的离散分布上按累计概率抽样。
    const float draw = static_cast<float>((next_random_u64(random_state) >> 40) * (1.0 / 16777216.0)) * kept_total;
    float cumulative = 0.0f;
    for (size_t i = 0; i < kept; ++i) {
        cumulative += probabilities[i];
        if (draw < cumulative) return candidates[i];
    }
    return candidates[kept - 1];
}

// 将采样和 prefill/decode 刻意分开：替换采样策略不会改变模型计算或 cache 更新路径。
std::vector<int> generate_tokens(const char* checkpoint_directory, const std::vector<int>& prompt, int new_tokens,
                                 const SamplingOptions& options, int stop_token = -1) {
    require(!prompt.empty(), "generation requires at least one prompt token");
    std::puts("loading text weights with mmap (this maps the checkpoint but does not copy all weights into RAM)...");
    std::fflush(stdout);
    TextModel model(checkpoint_directory);
    RuntimeState state;
    ForwardWorkspace workspace;

    // prefill：顺序喂入整段 prompt，逐 token 建好所有层的 KV / recurrent cache。
    // 这里不做并行 prompt GEMM，但数学结果与真正的 prefill 相同，更便于参考实现阅读。
    for (int token : prompt) forward_token_cpu(model, &state, token, &workspace);
    std::vector<int> generated;
    generated.reserve(new_tokens);
    uint64_t random_state = options.seed;
    for (int step = 0; step < new_tokens; ++step) {
        // 当前 logits 预测“下一个” token；抽到 token 后，下一轮才把它送回模型。
        const int token = sample_token(workspace.logits, options, &random_state);
        // 终止 token 仅作为控制符，不把它放进用户可见的输出。
        if (token == stop_token) break;
        generated.push_back(token);
        // 最后一次抽样之后无需计算下一份 logits，因此少做一次 forward。
        if (step + 1 < new_tokens) forward_token_cpu(model, &state, token, &workspace);
    }
    std::printf("prefill: %zu token(s), generated: %zu/%d token(s), evaluated: %d token(s)\n", prompt.size(),
                generated.size(), new_tokens, state.position);
    return generated;
}

void run_generate(const char* checkpoint_directory, const char* token_string, int new_tokens, const SamplingOptions& options) {
    // 无 tokenizer 的底层入口：直接以 token id 为 prompt，
    // 用于快速验证 loader、forward、cache 与 sampler 是否连通。
    const std::vector<int> prompt = parse_token_ids(token_string);
    const std::vector<int> generated = generate_tokens(checkpoint_directory, prompt, new_tokens, options);
    std::printf("generated:");
    for (int token : generated) std::printf(" %d", token);
    std::putchar('\n');
}

#ifdef QWEN38_WITH_TOKENIZER
// Tokenizer 不属于核心 inference 数学。启用此选项后，使用 libtokenizers-cpp
// 读取官方 tokenizer.json；模型主流程仍只接受/产生整数 token id。
void print_token_ids(const std::vector<int>& token_ids) {
    std::printf("ids:");
    for (int token_id : token_ids) std::printf(" %d", token_id);
    std::putchar('\n');
}

void run_tokenize(const char* checkpoint_directory, const char* text) {
    // 单独暴露编码/解码命令，方便将 tokenizer 与模型计算问题隔离排查。
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
    // encode 后仍调用同一个 token-level forward；末尾 logits 预测文本的下一个 token。
    run_forward_tokens(checkpoint_directory, token_ids);
}

void run_generate_text(const char* checkpoint_directory, const char* text, int new_tokens, const SamplingOptions& options) {
    QwenTokenizer tokenizer(checkpoint_directory);
    const std::vector<int> prompt = tokenizer.encode(text ? text : "");
    // 此低层 text 命令不擅自拼 chat template；prompt 按用户提供的原样编码。
    const std::vector<int> generated = generate_tokens(checkpoint_directory, prompt, new_tokens, options);
    print_token_ids(generated);
    std::printf("generated text: %s\n", tokenizer.decode(generated).c_str());
}

// 这里刻意不是通用 Jinja/chat-template runtime；只实现 Qwen3.5 官方模板中
// 禁用思考、无工具、单用户回合的精确形状。多轮、工具、视觉消息仍在 text M1 范围外。
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
    // 官方简单 chat 模板以 <|im_end|> 终止 assistant 回合；遇到它即停止，
    // 以免把下一条控制消息误解码到用户输出中。
    const std::vector<int> generated = generate_tokens(checkpoint_directory, prompt, new_tokens, options, 248046);
    print_token_ids(generated);
    std::printf("generated chat text: %s\n", tokenizer.decode(generated).c_str());
}
#endif

void test_config() {
    // 配置测试：开发用小模型的层模式与每层派生维度必须可自洽。
    int deltanet_layers = 0;
    for (int layer = 0; layer < Config::kLayers; ++layer) {
        if (Config::is_deltanet_layer(layer)) ++deltanet_layers;
    }
    require(deltanet_layers == Config::kDeltaLayers, "Qwen hybrid DeltaNet layer count is wrong");
    require(!Config::is_deltanet_layer(3), "Qwen hybrid attention interval is wrong");
    if (Config::kLayers > 4) require(Config::is_deltanet_layer(4), "Qwen hybrid DeltaNet layer pattern is wrong");
}

void test_bfloat16() {
    // BF16 读取、标量转换以及权重×向量的参考 GEMV。
    const float original = 1.234375f;  // 此数可被 BF16 精确表示。
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
    // RMSNorm 以及 Qwen MLP 的 gate/up/down 路径。
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
    // RoPE 旋转和 DeltaNet 前面的因果 depthwise convolution。
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
    // 单头 recurrence：检查 beta=1 时的可手算状态更新与输出。
    const DeltaNetShape shape = {1, 2, 2};
    DeltaNetState state(shape);
    // q 已做过缩放；这里刻意选用易手算的 q/k 单位向量。
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
    // causal attention：未来位置不能参与当前 token 的 softmax。
    // 两个 query head 共享一个 KV head；它们分别偏向对应的 key/value，
    // 以验证 GQA 映射与 FP32 softmax。
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
    // parser 不需要真实大权重，也可验证 safetensors JSON header 的字段处理。
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
    // 同 seed 必须可复现；temperature=0 必须退化为 argmax。
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
    // 这些是极小的白盒回归测试，不是模型质量 benchmark。
    // 它们先保证“数学积木”正确，再由端到端 checkpoint 测试验证组合结果。
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
    // --describe 是给人读的版本；--inspect 是给某个 checkpoint 读的版本。
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
    // 命令保持扁平：每个子命令直接对应教材中的一项可观察能力。
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

}  // namespace qwen38（实现细节全部收在这个命名空间内）

#ifndef QWEN38_NO_MAIN
int main(int argc, char** argv) {
    // qwen38.cpp 也会被 CUDA translation unit include；QWEN38_NO_MAIN
    // 让同一份 reference/loader 代码可复用，而不会重复定义程序入口。
    // 先处理无需加载权重的命令，使新环境的 smoke test 成本最低。
    if (argc == 1 || std::strcmp(argv[1], "--self-test") == 0) {
        qwen38::run_self_test();
        return 0;
    }
    if (std::strcmp(argv[1], "--describe") == 0) {
        qwen38::print_description();
        return 0;
    }
    if (std::strcmp(argv[1], "--inspect") == 0) {
        // --inspect 只解析 safetensors header，不分配模型权重，也不运行 forward。
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

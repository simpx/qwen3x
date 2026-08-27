// engine.cpp -- Qwen3.5-0.8B production runtime 的 plain C++ CPU engine。
//
// 数学实现延续 00-lessons/09 和 02-cpu-0.8b。这个文件只保留四块：
//   1. Model：固定权重；
//   2. State / Work：跨 token 历史与单次 forward 临时量；
//   3. deltanet / attention / ffn：三个已经学过的 branch；
//   4. forward：完整主干。
// File/Reader 只负责 mmap 权重；文件末尾用 internal.h 的窄接口暴露 Model/State。
// Session、token timeline、prefix、checkpoint 策略和 sampling 全部位于 runtime.cpp。
//
//   token id
//     -> embedding                              -> hidden[H]
//     -> 24 * {RMSNorm -> mixer -> residual
//              RMSNorm -> FFN   -> residual}    -> hidden[H]
//     -> final RMSNorm                          -> hidden[H]
//     -> tied lm_head                           -> logits[V]
//
// 真实 shape：hidden[1024]，FFN intermediate[3584]，logits[248320]；
// Attention Q[8,256]、K/V[2,256]；DeltaNet Q/K/V[16,128]、S[16,128,128]。
// 每四层固定为 DeltaNet、DeltaNet、DeltaNet、Attention。

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "internal.h"
#include "qwen35.h"

namespace qwen35 {

// 这些常数直接来自 Qwen/Qwen3.5-0.8B 的 text_config。不是运行时 Config；
// 换模型意味着提供另一份明确实现，而不是把这里演化成通用 Tensor 框架。
// V=词表大小，H=每个 token 的 hidden width，I=FFN intermediate width，N=层数。
constexpr int V = 248320, H = 1024, I = 3584, N = 24;
// full attention：AH 个 query head、KVH 个共享 key/value head、每 head AD 通道；
// 只旋转前 RD 个通道。AH/KVH 是 GQA 的压缩比来源。
constexpr int AH = 8, KVH = 2, AD = 256, RD = 64;
// DeltaNet：KH 个小 Q/K head，VH 个 value head；S 的单 head shape 是 [KD,VD]；
// CK 是 causal depthwise convolution 的 kernel 宽度。
constexpr int KH = 16, VH = 16, KD = 128, VD = 128, CK = 4;
// 以下是经常一起出现的平铺宽度。DQKV 的布局固定为 [small_Q | small_K | V]。
constexpr int AS = AH * AD, KV_WIDTH = KVH * AD;
constexpr int DQK = KH * KD, DO = VH * VD, DQKV = 2 * DQK + DO;
// checkpoint 的最大 position。每个 Session 在创建时选择不超过它的 context_size，
// State/Work 只按该 Session 的实际上限分配，而不是无条件分配 262K。
constexpr int MAX_CONTEXT = 262144;
constexpr uint32_t MODEL_FORMAT_VERSION = 1;
constexpr int END_OF_TEXT_TOKEN = 248044, IM_END_TOKEN = 248046;
using FP32 = float;
using BF16 = uint16_t;
static_assert(sizeof(FP32) == 4 && std::numeric_limits<FP32>::is_iec559);
static_assert(sizeof(BF16) == 2);
constexpr FP32 EPS = 1e-6f, THETA = 10000000.0f;

[[noreturn]] void die(const char* text) {
    // library 不能结束宿主 Python 进程；异常会在 C ABI 边界转换成错误码和文字。
    throw std::runtime_error(text);
}

FP32 f32(BF16 value) {
    // BF16 与 FP32 共用最高 16 bit；左移后按位复制即可得到精确的 FP32 表示。
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    FP32 out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

BF16 bf16(FP32 value) {
    // 只用于 self-test。真实模型权重由 pack_weights.py 原样复制，不会走这里。
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t bias = 0x7fffu + ((bits >> 16) & 1u);  // round-to-nearest-even
    return static_cast<BF16>((bits + bias) >> 16);
}

FP32 sigmoid(FP32 x) {
    if (x >= 0.0f) return 1.0f / (1.0f + std::exp(-x));
    const FP32 e = std::exp(x);
    return e / (1.0f + e);
}

FP32 silu(FP32 x) { return x * sigmoid(x); }  // SwiGLU 与 DeltaNet gate 共用的 SiLU。
FP32 softplus(FP32 x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

// Linear 始终表示没有 bias 的 row-major W[rows,cols]。指针直接指向 mmap 权重页。
struct Linear { const BF16* w = nullptr; int rows = 0, cols = 0; };
struct DeltaWeights {
    // DeltaNet 的四个输入投影、depthwise conv 参数、衰减参数、head norm 和输出投影。
    Linear qkv, z, a, b, out;
    const BF16* conv = nullptr;
    const FP32* alog = nullptr;
    const BF16* dt = nullptr;
    const FP32* norm = nullptr;
};
struct AttentionWeights {
    // q 同时产出 query 和 attention gate，所以有 2*AS 行；k/v 是 GQA 的小宽度。
    Linear q, k, v, out;
    const BF16* qnorm = nullptr;
    const BF16* knorm = nullptr;
};
struct Layer {
    // 每一层共有输入 RMSNorm、mixer 分支、post-mixer RMSNorm 和 FFN；delta/attention 只有
    // 其中一个被填充。bool 的目的只是保持 forward 的 if 一眼可读。
    bool uses_deltanet = false;
    const BF16* input_norm = nullptr;
    const BF16* post_norm = nullptr;
    Linear gate, up, down;
    DeltaWeights delta;
    AttentionWeights attention;
};

struct Model {
    // 同一张 [V,H] 表既用于开头的 token -> hidden 查表，也在结尾作为 tied lm_head：
    // final hidden 与表的每一行点积，得到 V 个“下一个 token”分数。
    const BF16* embedding = nullptr;
    const BF16* final_norm = nullptr;
    std::array<Layer, N> layer {};
};

// 两个 FP32 向量：[count] dot [count] -> 标量。
FP32 dot_f32(const FP32* left, const FP32* right, int count) {
#if defined(__AVX512F__)
    __m512 sums = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= count; i += 16) {
        sums = _mm512_fmadd_ps(_mm512_loadu_ps(left + i),
                               _mm512_loadu_ps(right + i), sums);
    }
    FP32 sum = _mm512_reduce_add_ps(sums);
    for (; i < count; ++i) sum += left[i] * right[i];
    return sum;
#else
    FP32 sum = 0.0f;
    for (int i = 0; i < count; ++i) sum += left[i] * right[i];
    return sum;
#endif
}

// BF16 权重行与 FP32 输入：[count] dot [count] -> FP32 标量。
FP32 dot_bf16(const BF16* weight, const FP32* input, int count) {
#if defined(__AVX512F__) && defined(__AVX512BW__)
    __m512 sums = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= count; i += 16) {
        const __m256i packed = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(weight + i));
        __m512i bits = _mm512_cvtepu16_epi32(packed);
        bits = _mm512_slli_epi32(bits, 16);
        const __m512 values = _mm512_castsi512_ps(bits);
        sums = _mm512_fmadd_ps(values, _mm512_loadu_ps(input + i), sums);
    }
    FP32 sum = _mm512_reduce_add_ps(sums);
    for (; i < count; ++i) sum += f32(weight[i]) * input[i];
    return sum;
#else
    FP32 sum = 0.0f;
    for (int i = 0; i < count; ++i) sum += f32(weight[i]) * input[i];
    return sum;
#endif
}

// GEMV：W[rows,cols] @ input[cols] -> output[rows]；遍历输出行，每行做一次 dot。
void mv(const Linear& weight, const FP32* input, FP32* output) {
    for (int row = 0; row < weight.rows; ++row) {
        const BF16* weight_row = weight.w + static_cast<size_t>(row) * weight.cols;
        output[row] = dot_bf16(weight_row, input, weight.cols);
    }
}

// Embedding lookup：table[V,H] + token 标量 -> out[H]；out=table[token,:]。
void embed(const BF16* table, int token, FP32* out) {
    if (token < 0 || token >= V) die("token outside vocabulary");
    // embedding table 的 shape 是 [V,H]；token id 唯一决定应复制的那一行。
    const BF16* row = table + static_cast<size_t>(token) * H;
    for (int i = 0; i < H; ++i) out[i] = f32(row[i]);
}

// Qwen RMSNorm：x[n] + weight[n] -> out[n]；out_i=x_i/sqrt(mean(x^2)+eps)*(1+weight_i)。
void rms(const FP32* x, const BF16* weight, int n, FP32* out) {
    // RMSNorm(x)_i = x_i / RMS(x) * (1 + weight_i)。它逐 token、逐向量工作，
    // 不跨 position，也不减均值。
    FP32 square = 0.0f;
    for (int i = 0; i < n; ++i) square += x[i] * x[i];
    const FP32 scale = 1.0f / std::sqrt(square / n + EPS);
    for (int i = 0; i < n; ++i) out[i] = x[i] * scale * (1.0f + f32(weight[i]));
}

// DeltaNet gated RMSNorm：x[VD],weight[VD],z[VD] -> out[VD]。
void gated_rms(const FP32* x, const FP32* weight, const FP32* z, FP32* out) {
    // DeltaNet head 内部专用：readout [VD] 先 RMSNorm，再乘 learned norm.weight 和
    // SiLU(z)。这里的 weight 是 checkpoint 中的 F32，和 ordinary RMSNorm 的 1+w
    // 约定不同；不要把两个 norm 函数合并后丢掉这一区别。
    FP32 square = 0.0f;
    for (int i = 0; i < VD; ++i) square += x[i] * x[i];
    const FP32 scale = 1.0f / std::sqrt(square / VD + EPS);
    for (int i = 0; i < VD; ++i) out[i] = x[i] * scale * weight[i] * silu(z[i]);
}

// L2 normalize：x[n] -> x[n]；x_i <- x_i/sqrt(sum(x^2)+eps)。
void l2(FP32* x, int n) {
    // L2Norm 令 q/k 的 Euclidean norm 接近 1；与 RMSNorm 的分母定义不同。
    FP32 square = 0.0f;
    for (int i = 0; i < n; ++i) square += x[i] * x[i];
    const FP32 scale = 1.0f / std::sqrt(square + EPS);
    for (int i = 0; i < n; ++i) x[i] *= scale;
}

// Residual：input[H] + branch[H] -> output[H]。
void residual_add(const FP32* input, const FP32* branch, FP32* output) {
    // 两个 residual 都是 [H]+[H] -> [H]；允许 output 与 input 指向同一数组。
    for (int i = 0; i < H; ++i) output[i] = input[i] + branch[i];
}

// RoPE：一个 head[AD] + position -> 同 shape head[AD]。前 RD=64 通道做 half-rotation，其余
// 通道保持原样；Q/K 都做相同 position 的旋转，V 永远不旋转。
void rope(FP32* x, int position) {
    FP32 old[RD];
    std::memcpy(old, x, sizeof(old));
    for (int i = 0; i < RD / 2; ++i) {
        const FP32 angle = position / std::pow(THETA, 2.0f * i / RD);
        const FP32 c = std::cos(angle), s = std::sin(angle);
        x[i] = old[i] * c - old[i + RD / 2] * s;
        x[i + RD / 2] = old[i + RD / 2] * c + old[i] * s;
    }
}

struct State {
    int position = 0;
    int capacity = 0;
    // 对 Delta layer：conv_history 是 [DQKV,CK-1]，delta_memory 是 [VH,KD,VD]；两者固定大小。
    // 对 attention layer：key/value cache 每 token append 一次，逻辑 shape 是
    // [tokens,KVH,AD]，故随 context 线性增长。非对应的 layer vector 保持为空。
    std::array<std::vector<FP32>, N> conv_history, delta_memory, key_cache, value_cache;
    explicit State(int context_size) : capacity(context_size) {
        for (int layer = 0; layer < N; ++layer) {
            if (layer % 4 != 3) {
                conv_history[layer].assign(static_cast<size_t>(DQKV) * (CK - 1), 0.0f);
                delta_memory[layer].assign(static_cast<size_t>(VH) * KD * VD, 0.0f);
            } else {
                key_cache[layer].reserve(static_cast<size_t>(capacity) * KV_WIDTH);
                value_cache[layer].reserve(static_cast<size_t>(capacity) * KV_WIDTH);
            }
        }
    }

    void reset() {
        position = 0;
        for (int layer = 0; layer < N; ++layer) {
            if (layer % 4 != 3) {
                std::fill(conv_history[layer].begin(), conv_history[layer].end(), 0.0f);
                std::fill(delta_memory[layer].begin(), delta_memory[layer].end(), 0.0f);
            } else {
                // clear 改变 logical length，但保留 reserve 出来的 Session Slot 内存。
                key_cache[layer].clear();
                value_cache[layer].clear();
            }
        }
    }
};

struct Work {
    // 公共临时量。
    std::vector<FP32> hidden = std::vector<FP32>(H);       // [H]
    std::vector<FP32> normalized = std::vector<FP32>(H);   // [H]
    std::vector<FP32> branch = std::vector<FP32>(H);       // [H]
    std::vector<FP32> logits = std::vector<FP32>(V);       // [V]

    // FFN：hidden[H] -> gate/up[I] -> branch[H]。
    std::vector<FP32> ffn_gate = std::vector<FP32>(I);     // [I]
    std::vector<FP32> ffn_up = std::vector<FP32>(I);       // [I]

    // DeltaNet。
    std::vector<FP32> delta_qkv = std::vector<FP32>(DQKV); // [Q | K | V]
    std::vector<FP32> delta_z = std::vector<FP32>(DO);     // [VH,VD]
    std::vector<FP32> delta_a = std::vector<FP32>(VH);     // [VH]
    std::vector<FP32> delta_b = std::vector<FP32>(VH);     // [VH]
    std::vector<FP32> delta_q = std::vector<FP32>(DO);     // [VH,KD]
    std::vector<FP32> delta_k = std::vector<FP32>(DO);     // [VH,KD]
    std::vector<FP32> delta_output = std::vector<FP32>(DO);// [VH,VD]

    // Attention。
    std::vector<FP32> query_and_gate = std::vector<FP32>(2 * AS); // [AH,2,AD]
    std::vector<FP32> query = std::vector<FP32>(AS);              // [AH,AD]
    std::vector<FP32> attention_gate = std::vector<FP32>(AS);     // [AH,AD]
    std::vector<FP32> key = std::vector<FP32>(KV_WIDTH);          // [KVH,AD]
    std::vector<FP32> value = std::vector<FP32>(KV_WIDTH);        // [KVH,AD]
    std::vector<FP32> attention_output = std::vector<FP32>(AS);   // [AH,AD]
    std::vector<FP32> attention_score; // [context_size]

    explicit Work(int context_size) : attention_score(context_size) {}
};

// DeltaNet 卷积是逐通道的 causal depthwise convolution；state 只保存 CK-1 个过去输入。
// weight 的布局是 [channel, CK]，当前输入乘最后一个权重；history 从最旧到最新。
void conv_step(FP32* x, const BF16* weight, std::vector<FP32>& history) {
    for (int channel = 0; channel < DQKV; ++channel) {
        FP32* past = history.data() + static_cast<size_t>(channel) * (CK - 1);
        FP32 sum = x[channel] * f32(weight[channel * CK + CK - 1]);
        for (int i = 0; i < CK - 1; ++i) sum += past[i] * f32(weight[channel * CK + i]);
        for (int i = 0; i < CK - 2; ++i) past[i] = past[i + 1];
        past[CK - 2] = x[channel];
        x[channel] = silu(sum);
    }
}

// 一个 value head 的四步 Delta rule。调用者已经切到某个 head 的 S=[KD,VD]。
// 这份 state 是 DeltaNet 不需要 KV cache 的原因：context 再长，S 的元素数不变。
void delta_rule(const FP32* q, const FP32* k, const FP32* v, FP32 log_decay, FP32 beta,
                FP32* state, FP32* out) {
    // (1) S <- exp(log_decay) S：遗忘旧记忆。log_decay 在模型中被构造成负数。
    const FP32 decay = std::exp(log_decay);
    for (int i = 0; i < KD * VD; ++i) state[i] *= decay;
    // (2) memory <- k^T S：当前 key 查询“这个位置已记住的 value”。
    FP32 memory[VD] = {};
    for (int value = 0; value < VD; ++value) {
        for (int key = 0; key < KD; ++key) memory[value] += k[key] * state[key * VD + value];
    }
    // (3) S <- S + k outer [beta * (v-memory)]：只写入预测误差，而非盲目累加 v。
    for (int key = 0; key < KD; ++key) {
        for (int value = 0; value < VD; ++value) {
            state[key * VD + value] += k[key] * beta * (v[value] - memory[value]);
        }
    }
    // (4) out <- q^T S：更新后立即读取，因此当前 token 能影响自己的输出。
    for (int value = 0; value < VD; ++value) {
        out[value] = 0.0f;
        for (int key = 0; key < KD; ++key) out[value] += q[key] * state[key * VD + value];
    }
}

// 完整 DeltaNet mixer：input[H] -> out[H]。
// input -> qkv[DQKV],z[DO],a/b[VH]；各 head 更新 S[KD,VD]，拼接 [DO] 后投影回 [H]。
void deltanet(const DeltaWeights& weights, std::vector<FP32>& conv_history,
              std::vector<FP32>& memory, const FP32* input, Work& work, FP32* out) {
    // 输入已经过 layer input RMSNorm。四个投影都从同一个 normalized hidden 得到。
    mv(weights.qkv, input, work.delta_qkv.data());
    mv(weights.z, input, work.delta_z.data());
    mv(weights.a, input, work.delta_a.data());
    mv(weights.b, input, work.delta_b.data());
    // 只有拼接 Q/K/V 经过 causal depthwise conv；z、a、b 不经过它。
    conv_step(work.delta_qkv.data(), weights.conv, conv_history);

    // 固定布局拆包，不创建 view 类：small_q[KH,KD]、small_k[KH,KD]、value[VH,VD]。
    const FP32* small_q = work.delta_qkv.data();
    const FP32* small_k = small_q + DQK;
    const FP32* value = small_k + DQK;
    for (int head = 0; head < VH; ++head) {
        // 27B 的 48 个 value head 会复用 16 个 small Q/K head；0.8B 此处是 1:1。
        const int qk_head = head / (VH / KH);
        FP32* q = work.delta_q.data() + head * KD;
        FP32* k = work.delta_k.data() + head * KD;
        std::memcpy(q, small_q + qk_head * KD, KD * sizeof(FP32));
        std::memcpy(k, small_k + qk_head * KD, KD * sizeof(FP32));
        l2(q, KD);
        l2(k, KD);
        // Q 的额外 1/sqrt(KD) 是此实现的 Delta rule 缩放约定。
        for (int i = 0; i < KD; ++i) q[i] /= std::sqrt(static_cast<FP32>(KD));
        // b 经过 sigmoid 得到写入比例 beta in (0,1)；a/A_log/dt_bias 决定衰减速度。
        const FP32 beta = sigmoid(work.delta_b[head]);
        const FP32 log_decay = -std::exp(weights.alog[head])
                               * softplus(work.delta_a[head] + f32(weights.dt[head]));
        delta_rule(q, k, value + head * VD, log_decay, beta,
                   memory.data() + static_cast<size_t>(head) * KD * VD,
                   work.delta_output.data() + head * VD);
        // 每个 value head 的 readout 在各自 VD 段内 norm/gate，之后再拼接为 [DO]。
        gated_rms(work.delta_output.data() + head * VD, weights.norm,
                  work.delta_z.data() + head * VD, work.delta_output.data() + head * VD);
    }
    mv(weights.out, work.delta_output.data(), out);
}

// 完整 GQA mixer：input[H] -> out[H]，并 append 当前 K/V。
// Q[AH,AD] @ K_cache[T,KVH,AD] -> score[AH,T]，加权 V 后拼接 [AS] 并投影回 [H]。
void attention(const AttentionWeights& weights, std::vector<FP32>& key_cache,
               std::vector<FP32>& value_cache, int position, const FP32* input,
               Work& work, FP32* out) {
    // q_proj 输出 [Q, gate] 两个 [AS] 向量；k/v 分别是 GQA 的 [KVH,AD] 向量。
    mv(weights.q, input, work.query_and_gate.data());
    mv(weights.k, input, work.key.data());
    mv(weights.v, input, work.value.data());
    for (int head = 0; head < AH; ++head) {
        // QNorm 和 RoPE 是逐 query head 操作；gate 不 norm、不旋转，留到 attention
        // readout 后才做 sigmoid 门控。
        std::memcpy(work.query.data() + head * AD,
                    work.query_and_gate.data() + head * 2 * AD, AD * sizeof(FP32));
        std::memcpy(work.attention_gate.data() + head * AD,
                    work.query_and_gate.data() + head * 2 * AD + AD, AD * sizeof(FP32));
        rms(work.query.data() + head * AD, weights.qnorm, AD, work.query.data() + head * AD);
        rope(work.query.data() + head * AD, position);
    }
    for (int head = 0; head < KVH; ++head) {
        // K 与 Q 使用同一 RoPE position；V 不依赖位置，因此不做 RoPE。
        rms(work.key.data() + head * AD, weights.knorm, AD, work.key.data() + head * AD);
        rope(work.key.data() + head * AD, position);
    }
    // 先 append 再读，等价于 causal attention 允许 token t 读取 0..t（包括自己）。
    key_cache.insert(key_cache.end(), work.key.begin(), work.key.end());
    value_cache.insert(value_cache.end(), work.value.begin(), work.value.end());
    const int tokens = static_cast<int>(key_cache.size() / KV_WIDTH);
    const FP32 scale = 1.0f / std::sqrt(static_cast<FP32>(AD));
    FP32* score = work.attention_score.data();  // [T]，逐 head 复用，不在 forward 中分配。

    for (int head = 0; head < AH; ++head) {
        // GQA：8 个 Q head 按 4:1 分为两组，每组共享一个 K/V head。
        const int kv_head = head / (AH / KVH);
        FP32 maximum = -std::numeric_limits<FP32>::infinity();
        for (int token = 0; token < tokens; ++token) {
            const FP32* key = key_cache.data() + (static_cast<size_t>(token) * KVH + kv_head) * AD;
            score[token] = dot_f32(work.query.data() + head * AD, key, AD) * scale;
            maximum = std::max(maximum, score[token]);
        }
        // stable softmax：同时复用 score 数组存 exp(score-maximum)。
        FP32 total = 0.0f;
        for (int token = 0; token < tokens; ++token) {
            score[token] = std::exp(score[token] - maximum);
            total += score[token];
        }
        FP32* result = work.attention_output.data() + head * AD;
        std::fill(result, result + AD, 0.0f);
        for (int token = 0; token < tokens; ++token) {
            const FP32 probability = score[token] / total;
            const FP32* value = value_cache.data() + (static_cast<size_t>(token) * KVH + kv_head) * AD;
            for (int i = 0; i < AD; ++i) result[i] += probability * value[i];
        }
    }
    // Qwen Gated Attention：attention readout 与 q_proj 的另一半 gate 逐元素相乘。
    for (int i = 0; i < AS; ++i) {
        work.attention_output[i] *= sigmoid(work.attention_gate[i]);
    }
    mv(weights.out, work.attention_output.data(), out);
}

// SwiGLU FFN：input[H] -> gate/up[I] -> mixed[I] -> out[H]。
void ffn(const Layer& layer, const FP32* input, Work& work, FP32* out) {
    // SwiGLU：gate/up 各投影到 [I]，逐元素 SiLU(gate)*up，再 down_proj 回 [H]。
    mv(layer.gate, input, work.ffn_gate.data());
    mv(layer.up, input, work.ffn_up.data());
    for (int i = 0; i < I; ++i) {
        work.ffn_gate[i] = silu(work.ffn_gate[i]) * work.ffn_up[i];
    }
    mv(layer.down, work.ffn_gate.data(), out);
}

// 这是完整的单 token forward。调用一次只处理一个 token；
// prompt 的 prefill 只是对 prompt ids 连续调用它，decode 则在每次 argmax 后再调用。
void forward(const Model& model, State& state, int token, Work& work,
             bool compute_logits = true) {
    if (state.position >= state.capacity) die("session context is full");
    LOG_DEBUG("forward started token=%d position=%d", token, state.position);
    embed(model.embedding, token, work.hidden.data());  // token id -> hidden[H]。
    for (int index = 0; index < N; ++index) {
        const Layer& layer = model.layer[index];
        // hidden[H] -> RMSNorm -> mixer -> branch[H] -> residual -> hidden[H]。
        rms(work.hidden.data(), layer.input_norm, H, work.normalized.data());
        if (layer.uses_deltanet) {
            deltanet(layer.delta, state.conv_history[index], state.delta_memory[index],
                     work.normalized.data(), work, work.branch.data());
        } else {
            attention(layer.attention, state.key_cache[index], state.value_cache[index],
                      state.position, work.normalized.data(), work, work.branch.data());
        }
        residual_add(work.hidden.data(), work.branch.data(), work.hidden.data());

        // hidden[H] -> RMSNorm -> FFN -> branch[H] -> residual -> hidden[H]。
        rms(work.hidden.data(), layer.post_norm, H, work.normalized.data());
        ffn(layer, work.normalized.data(), work, work.branch.data());
        residual_add(work.hidden.data(), work.branch.data(), work.hidden.data());
    }
    if (compute_logits) {
        rms(work.hidden.data(), model.final_norm, H, work.normalized.data());
        LOG_DEBUG("final norm completed");
        // lm_head：为词表的每一个候选 token 各算一个 logit。这里 W 直接重用 model.embedding：
        // logits[v] = dot(final_hidden, embedding[v])。这叫 tied embedding，避免再存一份 [V,H]
        // 输出矩阵；它与开头 embed() 的“按 token id 取同一张表的一行”正好相对。
        mv({model.embedding, V, H}, work.normalized.data(), work.logits.data());
        LOG_DEBUG("lm head completed logits=%d", V);
    }
    ++state.position;
    LOG_DEBUG("forward completed token=%d position=%d", token, state.position);
}

// 到这里模型数学已经结束。以下 File/Reader 只把 packed 文件绑定成上面的 Model 指针。
// packed 格式是 [magic 8][version u32][reserved u32]，随后每个 tensor 从 64-byte 边界开始。
struct File {
    int fd = -1;
    size_t size = 0;
    const uint8_t* data = nullptr;

    explicit File(const char* path) {
        try {
            fd = open(path, O_RDONLY);
            if (fd < 0) die("cannot open model.bin");
            struct stat info {};
            if (fstat(fd, &info) || info.st_size < 16) die("bad model.bin");
            size = static_cast<size_t>(info.st_size);
            data = static_cast<const uint8_t*>(mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
            if (data == MAP_FAILED) die("mmap model.bin failed");
            if (std::memcmp(data, "Q35COUR\0", 8) != 0) {
                die("wrong model.bin magic; run pack_weights.py");
            }
            uint32_t version = 0, reserved = 0;
            std::memcpy(&version, data + 8, sizeof(version));
            std::memcpy(&reserved, data + 12, sizeof(reserved));
            if (version != MODEL_FORMAT_VERSION || reserved != 0) {
                die("unsupported model.bin version");
            }
        } catch (...) {
            release();
            throw;
        }
    }
    ~File() {
        release();
    }
    void release() noexcept {
        if (data && data != MAP_FAILED) munmap(const_cast<uint8_t*>(data), size);
        if (fd >= 0) close(fd);
        data = nullptr;
        fd = -1;
        size = 0;
    }
    File(const File&) = delete;
};

struct Reader {
    const uint8_t* begin;
    const uint8_t* cursor;
    const uint8_t* end;

    explicit Reader(const File& file)
        : begin(file.data), cursor(file.data + 16), end(file.data + file.size) {}

    template <typename T> const T* take(size_t count) {
        const size_t offset = static_cast<size_t>(cursor - begin);
        cursor += (64 - offset % 64) % 64;
        const size_t bytes = count * sizeof(T);
        if (cursor > end || bytes > static_cast<size_t>(end - cursor)) die("truncated model.bin");
        const T* result = reinterpret_cast<const T*>(cursor);
        cursor += bytes;
        return result;
    }
    void finish() const {
        if (cursor != end) die("model.bin size does not match Qwen3.5-0.8B schema");
    }
};

// LoadedModel 只负责生命周期：File 拥有 mmap，Model 中的权重指针指向这块 mmap。
struct LoadedModel {
    File file;
    Reader reader;
    Model model;

    explicit LoadedModel(const char* path) : file(path), reader(file) {
        model.embedding = reader.take<BF16>(static_cast<size_t>(V) * H);
        model.final_norm = reader.take<BF16>(H);
        for (int index = 0; index < N; ++index) {
            Layer& layer = model.layer[index];
            layer.uses_deltanet = index % 4 != 3;
            layer.input_norm = reader.take<BF16>(H);
            if (layer.uses_deltanet) {
                layer.delta.qkv = {reader.take<BF16>(static_cast<size_t>(DQKV) * H), DQKV, H};
                layer.delta.z = {reader.take<BF16>(static_cast<size_t>(DO) * H), DO, H};
                layer.delta.a = {reader.take<BF16>(static_cast<size_t>(VH) * H), VH, H};
                layer.delta.b = {reader.take<BF16>(static_cast<size_t>(VH) * H), VH, H};
                layer.delta.conv = reader.take<BF16>(static_cast<size_t>(DQKV) * CK);
                layer.delta.alog = reader.take<FP32>(VH);
                layer.delta.dt = reader.take<BF16>(VH);
                layer.delta.norm = reader.take<FP32>(VD);
                layer.delta.out = {reader.take<BF16>(static_cast<size_t>(H) * DO), H, DO};
            } else {
                layer.attention.q = {
                    reader.take<BF16>(static_cast<size_t>(2 * AS) * H), 2 * AS, H};
                layer.attention.k = {
                    reader.take<BF16>(static_cast<size_t>(KV_WIDTH) * H), KV_WIDTH, H};
                layer.attention.v = {
                    reader.take<BF16>(static_cast<size_t>(KV_WIDTH) * H), KV_WIDTH, H};
                layer.attention.qnorm = reader.take<BF16>(AD);
                layer.attention.knorm = reader.take<BF16>(AD);
                layer.attention.out = {
                    reader.take<BF16>(static_cast<size_t>(H) * AS), H, AS};
            }
            layer.post_norm = reader.take<BF16>(H);
            layer.gate = {reader.take<BF16>(static_cast<size_t>(I) * H), I, H};
            layer.up = {reader.take<BF16>(static_cast<size_t>(I) * H), I, H};
            layer.down = {reader.take<BF16>(static_cast<size_t>(H) * I), H, I};
        }
        // 320 个 tensor 必须恰好吃完文件，拒绝截断和其他 schema 的额外 payload。
        reader.finish();
    }
};

}  // namespace qwen35

// 以下只是 runtime.cpp 与 plain CPU 实现之间的窄连接。Session 和 prefix
// 在 runtime；这里的 checkpoint 只描述 CPU State 实际怎样复制和恢复。
namespace q35_backend {

struct Checkpoint {
    int position = 0;
    std::array<std::vector<qwen35::FP32>, qwen35::N> conv_history;
    std::array<std::vector<qwen35::FP32>, qwen35::N> delta_memory;
    std::vector<qwen35::FP32> logits;
};

struct Model {
    qwen35::LoadedModel loaded;

    explicit Model(const char* path) : loaded(path) {}
};

struct State {
    qwen35::State live;
    qwen35::Work work;
    Checkpoint checkpoint;

    explicit State(int context_size) : live(context_size), work(context_size) {}
};

Model* model_create(const char* path) {
    return new Model(path);
}

void model_destroy(Model* model) {
    delete model;
}

State* state_create(Model* model, int context_size) {
    if (!model) throw std::runtime_error("backend model is null");
    return new State(context_size);
}

void state_destroy(State* state) {
    delete state;
}

void state_reset(State* state) {
    if (!state) throw std::runtime_error("backend state is null");
    state->live.reset();
}

void state_forward(Model* model, State* state, int token, bool compute_logits) {
    if (!model || !state) throw std::runtime_error("backend model/state is null");
    qwen35::forward(model->loaded.model, state->live, token, state->work,
                    compute_logits);
}

void state_checkpoint_save(State* state) {
    if (!state) throw std::runtime_error("backend state is null");
    Checkpoint& checkpoint = state->checkpoint;
    checkpoint.position = state->live.position;
    for (int layer = 0; layer < qwen35::N; ++layer) {
        if (layer % 4 == 3) continue;
        checkpoint.conv_history[layer] = state->live.conv_history[layer];
        checkpoint.delta_memory[layer] = state->live.delta_memory[layer];
    }
    checkpoint.logits = state->work.logits;
}

void state_checkpoint_restore(State* state) {
    if (!state) throw std::runtime_error("backend state is null");
    Checkpoint& checkpoint = state->checkpoint;
    const size_t kv_elements =
        static_cast<size_t>(checkpoint.position) * qwen35::KV_WIDTH;
    for (int layer = 0; layer < qwen35::N; ++layer) {
        if (layer % 4 != 3) {
            std::copy(checkpoint.conv_history[layer].begin(),
                      checkpoint.conv_history[layer].end(),
                      state->live.conv_history[layer].begin());
            std::copy(checkpoint.delta_memory[layer].begin(),
                      checkpoint.delta_memory[layer].end(),
                      state->live.delta_memory[layer].begin());
            continue;
        }
        if (state->live.key_cache[layer].size() < kv_elements ||
            state->live.value_cache[layer].size() < kv_elements) {
            throw std::runtime_error("attention cache is shorter than checkpoint");
        }
        state->live.key_cache[layer].resize(kv_elements);
        state->live.value_cache[layer].resize(kv_elements);
    }
    state->live.position = checkpoint.position;
    std::copy(checkpoint.logits.begin(), checkpoint.logits.end(),
              state->work.logits.begin());
}

int state_argmax(const State* state) {
    if (!state) return -1;
    return static_cast<int>(
        std::max_element(state->work.logits.begin(), state->work.logits.end()) -
        state->work.logits.begin());
}

void state_copy_logits(const State* state, float* output) {
    if (!state || !output) throw std::runtime_error("backend state/output is null");
    std::memcpy(output, state->work.logits.data(), sizeof(float) * qwen35::V);
}

int vocab_size() {
    return qwen35::V;
}

int max_context() {
    return qwen35::MAX_CONTEXT;
}

bool token_is_stop(int token) {
    return token == qwen35::END_OF_TEXT_TOKEN || token == qwen35::IM_END_TOKEN;
}

}  // namespace q35_backend

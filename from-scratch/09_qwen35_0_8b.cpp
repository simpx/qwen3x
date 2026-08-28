// 09_qwen35_0_8b.cpp -- 用官方权重运行 Qwen3.5-0.8B 文本主干。
//
// 第一遍不要从头逐行读。只找下面四个带“★”的入口：
//   ★1 Model：模型有哪些权重；
//   ★2 State / Work：哪些数据跨 token 保留，哪些只是临时结果；
//   ★3 deltanet / attention / ffn：三个已经学过的 branch；
//   ★4 forward：把一切拼起来的主干，建议最先读它。
// File、Reader、命令行和 self-test 只是让程序能运行，第一次阅读可以整段跳过。
//
// 一次 token 的完整数据流只有下面这些：
//
//   token id
//     -> embedding                              -> hidden[H]
//     -> 24 * {RMSNorm -> mixer -> residual
//              RMSNorm -> FFN   -> residual}    -> hidden[H]
//     -> final RMSNorm                          -> hidden[H]
//     -> tied lm_head                           -> logits[V]
//
// 每四层的 mixer 固定是：DeltaNet、DeltaNet、DeltaNet、Attention。
// FFN 和 MLP 在这里是同一个东西；本文统一叫 FFN，checkpoint 名字仍然叫 mlp.*。
//
// 真实 shape 总表（先看“进来多大、出去多大”，暂时不用背缩写）：
//
//   公共：      hidden[1024]，FFN intermediate[3584]，logits[248320]
//   Attention： Q[8,256]，K/V[2,256]；4 个 Q head 共享 1 个 KV head
//   DeltaNet：  Q/K/V[16,128]；每个 head 独立保存 S[128,128]
//
// 多 head 没有引入新的核心公式：attention() 只是把你学过的单 head attention 做 8 次，
// deltanet() 只是把你学过的单个 S 做 16 次，最后分别拼接再 linear 回 hidden[1024]。
// GQA 唯一多出来的规则是：Q head 0..3 共用 KV head 0，Q head 4..7 共用 KV head 1。
//
// 09_pack_weights.py 已按 Model 的读取顺序排好真实权重。本文件不做通用框架、不做
// GPU 优化，只保留能够算出官方模型正确 logits 的直接 CPU 实现。

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace qwen35 {

// 这些常数直接来自 Qwen/Qwen3.5-0.8B 的 text_config。不是运行时 Config；
// 换模型意味着另写一份课程，而不是把本文件演化成通用推理框架。
// V=词表大小，H=每个 token 的 hidden width，I=FFN intermediate width，N=层数。
constexpr int V = 248320, H = 1024, I = 3584, N = 24;
// full attention：AH 个 query head、KVH 个共享 key/value head、每 head AD 通道；
// 只旋转前 RD 个通道。AH/KVH 是 GQA 的压缩比来源。
constexpr int AH = 8, KVH = 2, AD = 256, RD = 64;
// DeltaNet：KH 个小 Q/K head，VH 个 value head；S 的单 head shape 是 [KD,VD]；
// CK 是 causal depthwise convolution 的 kernel 宽度。
constexpr int KH = 16, VH = 16, KD = 128, VD = 128, CK = 4;
// 以下是经常一起出现的平铺宽度。DQKV 的布局固定为 [small_Q | small_K | V]。
constexpr int AS = AH * AD, KVS = KVH * AD, DQK = KH * KD, DO = VH * VD, DQKV = 2 * DQK + DO;
// checkpoint 实际支持 262,144 positions；课程只为避免误用而给 cache 一个明确上限。
// 4096 也恰好能容纳 MMLU-Pro 官方 runner 的 context window，仍远小于模型的能力边界。
constexpr int MAX_TOKENS = 4096;
// C++ 的内置 float 在本程序支持的平台上就是 IEEE FP32。起别名是为了让它和
// 只保存 BF16 原始 bit 的 uint16_t 在代码里一眼可分。
using FP32 = float;
using BF16 = uint16_t;
static_assert(sizeof(FP32) == 4 && std::numeric_limits<FP32>::is_iec559);
static_assert(sizeof(BF16) == 2);
constexpr FP32 EPS = 1e-6f, THETA = 10000000.0f;

// 报告运行时错误并终止程序；不涉及 tensor 和 shape。
[[noreturn]] void die(const char* text) {
    std::fprintf(stderr, "qwen35: %s\n", text);
    std::exit(1);
}

// BF16 -> FP32：把 BF16 的 16 bit 放到 FP32 高16位，低16位补0；数值 shape 不变。
FP32 f32(BF16 value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    FP32 out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// FP32 -> BF16：丢弃低16位，并使用 round-to-nearest-even；数值 shape 不变。
// bias=0x7fff+(高16位最后一bit)，低16位加 bias 后是否进位决定高16位是否加1。
BF16 bf16(FP32 value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    // 0x7fff 是“低 16 bit 的一半 0x8000”减 1；末尾 U 表示 unsigned。
    // 再加上将被保留部分的最后一 bit，实现 round-to-nearest-even。
    const uint32_t bias = 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<BF16>((bits + bias) >> 16);
}

// Sigmoid 标量函数：sigmoid(x)=1/(1+exp(-x))，输出范围 (0,1)。
FP32 sigmoid(FP32 x) {
    if (x >= 0.0f) return 1.0f / (1.0f + std::exp(-x));
    const FP32 e = std::exp(x);
    return e / (1.0f + e);
}

// SiLU 标量函数：SiLU(x)=x*sigmoid(x)。
FP32 silu(FP32 x) { return x * sigmoid(x); }

// Softplus 标量函数：softplus(x)=log(1+exp(x))>0；x>20 时用 x 近似。
FP32 softplus(FP32 x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

// ─────────────────────────────────────────────────────────────────────────────
// ★1 Model：这里只描述 forward 会用到哪些权重
// ─────────────────────────────────────────────────────────────────────────────
// Linear 始终表示没有 bias 的 row-major W[rows,cols]。指针直接指向 mmap 权重页。
struct Linear { const BF16* w = nullptr; int rows = 0, cols = 0; };
struct DeltaWeights {
    Linear qkv;  // W[6144,1024]：hidden[1024] -> Q/K/V[16,128]
    Linear z;    // W[2048,1024]：hidden[1024] -> output gate[16,128]
    Linear a;    // W[16,1024]：每个 head 一个 decay 输入
    Linear b;    // W[16,1024]：每个 head 一个 beta 输入
    Linear out;  // W[1024,2048]：拼接 16 个 head -> hidden[1024]
    const BF16* conv = nullptr;    // [6144,4]
    const FP32* alog = nullptr;   // [16]
    const BF16* dt = nullptr;      // [16]
    const FP32* norm = nullptr;   // [128]，16 个 head 共用
};
struct AttentionWeights {
    Linear q;    // W[4096,1024]：hidden -> Q[8,256] + gate[8,256]
    Linear k;    // W[512,1024]：hidden -> K[2,256]
    Linear v;    // W[512,1024]：hidden -> V[2,256]
    Linear out;  // W[1024,2048]：拼接 8 个 head -> hidden[1024]
    const BF16* qnorm = nullptr;  // [256]，8 个 Q head 共用
    const BF16* knorm = nullptr;  // [256]，2 个 K head 共用
};
struct Layer {
    // 每层只有一种 mixer：DeltaNet 或 Attention；之后一定有同一个 SwiGLU FFN。
    bool uses_deltanet = false;
    const BF16* input_norm = nullptr;  // [1024]
    const BF16* post_norm = nullptr;   // [1024]
    Linear gate;  // W[3584,1024]
    Linear up;    // W[3584,1024]
    Linear down;  // W[1024,3584]
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

// 两个 FP32 向量的点积：[count] dot [count] -> 标量。
// dot(left,right)=sum_i(left[i]*right[i])。
FP32 dot_f32(const FP32* left, const FP32* right, int count) {
    FP32 sum = 0.0f;
    for (int i = 0; i < count; ++i) sum += left[i] * right[i];
    return sum;
}

// FP32 输入与 BF16 权重行的点积：[count] dot [count] -> FP32 标量。
// dot(weight,input)=sum_i(f32(weight[i])*input[i])。
FP32 dot_bf16(const BF16* weight, const FP32* input, int count) {
    FP32 sum = 0.0f;
    for (int i = 0; i < count; ++i) sum += f32(weight[i]) * input[i];
    return sum;
}

// 无 bias 的 matrix-vector multiplication：input[cols] -> output[rows]。
// output[row]=dot(weight[row],input)，即 W[rows,cols]@input[cols]。
void mv(const Linear& weight, const FP32* input, FP32* output) {
    for (int row = 0; row < weight.rows; ++row) {
        const BF16* weight_row = weight.w + static_cast<size_t>(row) * weight.cols;
        output[row] = dot_bf16(weight_row, input, weight.cols);
    }
}

// Embedding lookup：token 标量 -> out[H]，公式 out=table[token,:]，table[V,H]。
void embed(const BF16* table, int token, FP32* out) {
    if (token < 0 || token >= V) die("token outside vocabulary");
    // embedding table 的 shape 是 [V,H]；token id 唯一决定应复制的那一行。
    const BF16* row = table + static_cast<size_t>(token) * H;
    for (int i = 0; i < H; ++i) out[i] = f32(row[i]);
}

// Qwen RMSNorm：x[n] -> out[n]，weight[n]。
// m=mean(x^2)，out[i]=x[i]/sqrt(m+eps)*(1+weight[i])。
void rms(const FP32* x, const BF16* weight, int n, FP32* out) {
    // RMSNorm(x)_i = x_i / RMS(x) * (1 + weight_i)。它逐 token、逐向量工作，
    // 不跨 position，也不减均值。
    FP32 square = 0.0f;
    for (int i = 0; i < n; ++i) square += x[i] * x[i];
    const FP32 scale = 1.0f / std::sqrt(square / n + EPS);
    for (int i = 0; i < n; ++i) out[i] = x[i] * scale * (1.0f + f32(weight[i]));
}

// DeltaNet 单个 value head 的 gated RMSNorm：x[VD],weight[VD],z[VD] -> out[VD]。
// out[i]=x[i]/sqrt(mean(x^2)+eps)*weight[i]*SiLU(z[i])。
void gated_rms(const FP32* x, const FP32* weight, const FP32* z, FP32* out) {
    // DeltaNet head 内部专用：readout [VD] 先 RMSNorm，再乘 learned norm.weight 和
    // SiLU(z)。这里的 weight 是 checkpoint 中的 F32，和 ordinary RMSNorm 的 1+w
    // 约定不同；不要把两个 norm 函数合并后丢掉这一区别。
    FP32 square = 0.0f;
    for (int i = 0; i < VD; ++i) square += x[i] * x[i];
    const FP32 scale = 1.0f / std::sqrt(square / VD + EPS);
    for (int i = 0; i < VD; ++i) out[i] = x[i] * scale * weight[i] * silu(z[i]);
}

// L2 normalize：input[n] -> output[n]。
// output=input/sqrt(sum_i(input[i]^2)+eps)，所以 ||output||_2 约等于1。
void l2(const FP32* input, int n, FP32* output) {
    // L2Norm 令 q/k 的 Euclidean norm 接近 1；与 RMSNorm 的分母定义不同。
    FP32 square = 0.0f;
    for (int i = 0; i < n; ++i) square += input[i] * input[i];
    const FP32 scale = 1.0f / std::sqrt(square + EPS);
    for (int i = 0; i < n; ++i) output[i] = input[i] * scale;
}

// Residual add：input[H]+branch[H] -> output[H]。
// output[i]=input[i]+branch[i]；允许 output 与 input 指向同一数组。
void residual_add(const FP32* input, const FP32* branch, FP32* output) {
    // 两个 residual 都是 [H]+[H] 的原地逐元素加法。
    for (int i = 0; i < H; ++i) output[i] = input[i] + branch[i];
}

// RoPE：input[AD] -> output[AD]，只旋转前 RD 个通道，其余通道原样复制。
// angle_i=position/theta^(2i/RD)，对 (i,i+RD/2) 做二维旋转。
void rope(const FP32* input, int position, FP32* output) {
    FP32 old[RD];
    std::memcpy(old, input, sizeof(old));
    if (output != input) std::memcpy(output, input, AD * sizeof(FP32));
    for (int i = 0; i < RD / 2; ++i) {
        const FP32 angle = position / std::pow(THETA, 2.0f * i / RD);
        const FP32 c = std::cos(angle), s = std::sin(angle);
        output[i] = old[i] * c - old[i + RD / 2] * s;
        output[i + RD / 2] = old[i + RD / 2] * c + old[i] * s;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ★2 State / Work：State 跨 token 保留；Work 每次 forward 都可以覆盖
// ─────────────────────────────────────────────────────────────────────────────
struct State {
    int position = 0;
    // 对 Delta layer：conv_history 是 [DQKV,CK-1]，delta_memory 是 [VH,KD,VD]；两者固定大小。
    // 对 attention layer：key/value cache 每 token append 一次，逻辑 shape 是
    // [tokens,KVH,AD]，故随 context 线性增长。非对应的 layer vector 保持为空。
    std::array<std::vector<FP32>, N> conv_history, delta_memory, key_cache, value_cache;
    // 初始化每个 Delta layer 的 conv_history[DQKV,CK-1] 和 S[VH,KD,VD] 为0。
    // Attention 的 KV cache 初始 shape=[0,KVH,AD]，所以保留为空 vector。
    State() {
        for (int layer = 0; layer < N; ++layer) {
            if (layer % 4 != 3) {
                conv_history[layer].assign(static_cast<size_t>(DQKV) * (CK - 1), 0.0f);
                delta_memory[layer].assign(static_cast<size_t>(VH) * KD * VD, 0.0f);
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

    // FFN 临时量：hidden[H] -> gate/up[I] -> branch[H]。
    std::vector<FP32> ffn_gate = std::vector<FP32>(I);     // [I]
    std::vector<FP32> ffn_up = std::vector<FP32>(I);       // [I]

    // DeltaNet 临时量。
    std::vector<FP32> delta_qkv = std::vector<FP32>(DQKV); // [Q | K | V]
    std::vector<FP32> delta_z = std::vector<FP32>(DO);     // output gate [VH,VD]
    std::vector<FP32> delta_a = std::vector<FP32>(VH);     // decay input [VH]
    std::vector<FP32> delta_b = std::vector<FP32>(VH);     // beta input [VH]
    std::vector<FP32> delta_q = std::vector<FP32>(DO);     // [VH,KD]
    std::vector<FP32> delta_k = std::vector<FP32>(DO);     // [VH,KD]
    std::vector<FP32> delta_output = std::vector<FP32>(DO);// [VH,VD]

    // Attention 临时量。
    std::vector<FP32> query_and_gate = std::vector<FP32>(2 * AS); // [AH,2,AD]
    std::vector<FP32> query = std::vector<FP32>(AS);              // [AH,AD]
    std::vector<FP32> attention_gate = std::vector<FP32>(AS);     // [AH,AD]
    std::vector<FP32> key = std::vector<FP32>(KVS);               // [KVH,AD]
    std::vector<FP32> value = std::vector<FP32>(KVS);             // [KVH,AD]
    std::vector<FP32> attention_output = std::vector<FP32>(AS);   // [AH,AD]
};

// Causal depthwise conv：input[DQKV] -> output[DQKV]，并更新 history[DQKV,CK-1]。
// output[c]=SiLU(sum_lag(weight[c,lag]*x[t-CK+1+lag,c]))；不同 c 不互相混合。
void conv_step(const FP32* input, const BF16* weight, std::vector<FP32>& history,
               FP32* output) {
    for (int channel = 0; channel < DQKV; ++channel) {
        FP32* past = history.data() + static_cast<size_t>(channel) * (CK - 1);
        const FP32 current = input[channel];
        FP32 sum = current * f32(weight[channel * CK + CK - 1]);
        for (int i = 0; i < CK - 1; ++i) sum += past[i] * f32(weight[channel * CK + i]);
        for (int i = 0; i < CK - 2; ++i) past[i] = past[i + 1];
        past[CK - 2] = current;
        output[channel] = silu(sum);
    }
}

// 单个 DeltaNet head：q[KD],k[KD],v[VD],S[KD,VD] -> out[VD]，并原地更新 S。
// S=exp(log_decay)S；memory=k@S；S+=outer(k,beta*(v-memory))；out=q@S。
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

// ─────────────────────────────────────────────────────────────────────────────
// ★3 三个 branch：DeltaNet、Attention、FFN
// ─────────────────────────────────────────────────────────────────────────────
// 完整 DeltaNet mixer：input[H] -> out[H]。
// input -> qkv[DQKV],z[DO],a[VH],b[VH]；16 个 head 各自更新 S[KD,VD]，
// 产生 output[VH,VD]=[DO]，最后 Wout[H,DO]@[DO] -> out[H]。
void deltanet(const DeltaWeights& weights, std::vector<FP32>& conv_history,
              std::vector<FP32>& memory, const FP32* input, Work& work, FP32* out) {
    // 输入已经过 layer input RMSNorm。四个投影都从同一个 normalized hidden 得到。
    mv(weights.qkv, input, work.delta_qkv.data());
    mv(weights.z, input, work.delta_z.data());
    mv(weights.a, input, work.delta_a.data());
    mv(weights.b, input, work.delta_b.data());
    // 只有拼接 Q/K/V 经过 causal depthwise conv；z、a、b 不经过它。
    conv_step(work.delta_qkv.data(), weights.conv, conv_history, work.delta_qkv.data());

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
        l2(q, KD, q);
        l2(k, KD, k);
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

// 完整 GQA mixer：input[H] -> out[H]，并向 KV cache append 当前 K/V。
// input -> Q/gate[AH,AD],K/V[KVH,AD]；每个 Q head 计算
// softmax(Q@K_cache^T/sqrt(AD))@V_cache -> [AD]，拼接 [AS] 后 Wout[H,AS] -> out[H]。
void attention(const AttentionWeights& weights, std::vector<FP32>& key_cache,
               std::vector<FP32>& value_cache, int position, const FP32* input,
               Work& work, FP32* out) {
    // q_proj 输出 [Q, gate] 两个 [AS] 向量；k/v 分别是 GQA 的 [KVS] 向量。
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
        rope(work.query.data() + head * AD, position, work.query.data() + head * AD);
    }
    for (int head = 0; head < KVH; ++head) {
        // K 与 Q 使用同一 RoPE position；V 不依赖位置，因此不做 RoPE。
        rms(work.key.data() + head * AD, weights.knorm, AD, work.key.data() + head * AD);
        rope(work.key.data() + head * AD, position, work.key.data() + head * AD);
    }
    // 先 append 再读，等价于 causal attention 允许 token t 读取 0..t（包括自己）。
    key_cache.insert(key_cache.end(), work.key.begin(), work.key.end());
    value_cache.insert(value_cache.end(), work.value.begin(), work.value.end());
    const int tokens = static_cast<int>(key_cache.size() / KVS);
    const FP32 scale = 1.0f / std::sqrt(static_cast<FP32>(AD));
    std::vector<FP32> score(tokens);  // 单 head 的 score buffer，逐 head 复用。

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
        for (FP32& value : score) { value = std::exp(value - maximum); total += value; }
        FP32* result = work.attention_output.data() + head * AD;
        std::fill(result, result + AD, 0.0f);
        for (int token = 0; token < tokens; ++token) {
            const FP32 probability = score[token] / total;
            const FP32* value = value_cache.data() + (static_cast<size_t>(token) * KVH + kv_head) * AD;
            for (int i = 0; i < AD; ++i) result[i] += probability * value[i];
        }
    }
    // Qwen Gated Attention：attention readout 与 q_proj 的另一半 gate 逐元素相乘。
    for (int i = 0; i < AS; ++i) work.attention_output[i] *= sigmoid(work.attention_gate[i]);
    mv(weights.out, work.attention_output.data(), out);
}

// SwiGLU FFN：input[H] -> gate[I],up[I] -> mixed[I] -> out[H]。
// gate=Wg@input，up=Wu@input，mixed=SiLU(gate)*up，out=Wd@mixed。
void ffn(const Layer& layer, const FP32* input, Work& work, FP32* out) {
    // SwiGLU：gate/up 各投影到 [I]，逐元素 SiLU(gate)*up，再 down_proj 回 [H]。
    mv(layer.gate, input, work.ffn_gate.data());
    mv(layer.up, input, work.ffn_up.data());
    for (int i = 0; i < I; ++i) {
        work.ffn_gate[i] = silu(work.ffn_gate[i]) * work.ffn_up[i];
    }
    mv(layer.down, work.ffn_gate.data(), out);
}

// ─────────────────────────────────────────────────────────────────────────────
// ★4 主干：第一次阅读从这里开始
// ─────────────────────────────────────────────────────────────────────────────
// 单 token forward：token 标量 -> work.logits[V]，并更新每层的 Delta state / KV cache。
// h=embedding[token]；24层重复 h+=Mixer(RMSNorm(h))、h+=FFN(RMSNorm(h))；
// logits=embedding[V,H]@FinalRMSNorm(h)[H]。
void forward(const Model& model, State& state, int token, Work& work) {
    if (state.position >= MAX_TOKENS) die("teaching capstone supports at most 4096 tokens");
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
    rms(work.hidden.data(), model.final_norm, H, work.normalized.data());
    // lm_head：为词表的每一个候选 token 各算一个 logit。这里 W 直接重用 model.embedding：
    // logits[v] = dot(final_hidden, embedding[v])。这叫 tied embedding，避免再存一份 [V,H]
    // 输出矩阵；它与开头 embed() 的“按 token id 取同一张表的一行”正好相对。
    mv({model.embedding, V, H}, work.normalized.data(), work.logits.data());
    ++state.position;
}

// ─────────────────────────────────────────────────────────────────────────────
// 到这里，模型计算已经全部结束。以下只是加载文件、生成循环、测试和命令行。
// 第一次学习架构时可以停在这里。
// ─────────────────────────────────────────────────────────────────────────────

// File/Reader 只负责把 packer 生成的文件依次变成权重指针，不参与任何模型数学。
struct File {
    int fd = -1;
    size_t size = 0;
    const uint8_t* data = nullptr;

    // path -> data[size]：只读 mmap 整个 packed 文件，并检查16字节文件头。
    explicit File(const char* path) {
        fd = open(path, O_RDONLY);
        if (fd < 0) die("cannot open model.bin");
        struct stat info {};
        if (fstat(fd, &info) || info.st_size < 16) die("bad model.bin");
        size = static_cast<size_t>(info.st_size);
        data = static_cast<const uint8_t*>(mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
        if (data == MAP_FAILED) die("mmap model.bin failed");
        if (std::memcmp(data, "Q35COUR\0", 8) != 0) die("wrong model.bin magic; run 09_pack_weights.py");
    }
    // 释放 data[size] 的 mmap 和文件描述符。
    ~File() {
        if (data && data != MAP_FAILED) munmap(const_cast<uint8_t*>(data), size);
        if (fd >= 0) close(fd);
    }
    File(const File&) = delete;
};

struct Reader {
    const uint8_t* begin;
    const uint8_t* cursor;
    const uint8_t* end;

    // file.data[size] -> cursor=file.data+16，先跳过固定文件头。
    explicit Reader(const File& file) : begin(file.data), cursor(file.data + 16), end(file.data + file.size) {}

    // 从当前64字节对齐位置取得 T[count]，返回其首地址，并令 cursor+=count*sizeof(T)。
    template <typename T> const T* take(size_t count) {
        const size_t offset = static_cast<size_t>(cursor - begin);
        cursor += (64 - offset % 64) % 64;
        const size_t bytes = count * sizeof(T);
        if (cursor > end || bytes > static_cast<size_t>(end - cursor)) die("truncated model.bin");
        const T* result = reinterpret_cast<const T*>(cursor);
        cursor += bytes;
        return result;
    }
};

// LoadedModel 拥有 mmap 的生命周期；里面的 model 只保存指向 mmap 的权重指针。
struct LoadedModel {
    File file;
    Reader reader;
    Model model;

    // packed file -> Model：按 packer 的固定顺序把每段 tensor 地址绑定到对应权重字段。
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
                layer.attention.q = {reader.take<BF16>(static_cast<size_t>(2 * AS) * H), 2 * AS, H};
                layer.attention.k = {reader.take<BF16>(static_cast<size_t>(KVS) * H), KVS, H};
                layer.attention.v = {reader.take<BF16>(static_cast<size_t>(KVS) * H), KVS, H};
                layer.attention.qnorm = reader.take<BF16>(AD);
                layer.attention.knorm = reader.take<BF16>(AD);
                layer.attention.out = {reader.take<BF16>(static_cast<size_t>(H) * AS), H, AS};
            }
            layer.post_norm = reader.take<BF16>(H);
            layer.gate = {reader.take<BF16>(static_cast<size_t>(I) * H), I, H};
            layer.up = {reader.take<BF16>(static_cast<size_t>(I) * H), I, H};
            layer.down = {reader.take<BF16>(static_cast<size_t>(H) * I), H, I};
        }
    }
};

// Greedy 选 token：logits[V] -> token 标量。
// token=argmax_i(logits[i])；softmax 不改变大小顺序，所以这里不需要计算。
int argmax(const std::vector<FP32>& values) {
    return static_cast<int>(std::max_element(values.begin(), values.end()) - values.begin());
}

// 将 logits[V] 逐行打印，供 official oracle 做完整词表数值比较。
void dump_logits(const std::vector<FP32>& values) {
    // 只给开发期 official-oracle.py 使用：一行一个 FP32 logit，便于它比较完整词表，
    // 而不是只看 argmax 后“文字看起来正常”。正常的 --forward / --generate 不会走这里。
    for (FP32 value : values) std::printf("%.9g\n", value);
}

// 将命令行的 "a,b,c" 解析成 token ids[T]，并检查每个 id 满足 0<=id<V。
std::vector<int> parse_ids(const char* text) {
    // C++ core 刻意只接受 token ids。tokenizer 是可独立替换的文字外围工具，不应
    // 遮住本文件的模型计算；格式如 "248044,198,198"。
    std::vector<int> ids;
    const char* cursor = text;
    while (*cursor) {
        char* end = nullptr;
        const long value = std::strtol(cursor, &end, 10);
        if (end == cursor || value < 0 || value >= V) die("bad comma-separated token id");
        ids.push_back(static_cast<int>(value));
        if (*end == '\0') break;
        if (*end != ',') die("token ids must be comma-separated");
        cursor = end + 1;
    }
    if (ids.empty()) die("prompt is empty");
    return ids;
}

// Greedy generation：prompt[T] -> result[<=count]。
// 先逐 token prefill；随后 token_{t+1}=argmax(Forward(token_t,state_t))，遇到 EOS 停止。
void generate(const char* path, const std::vector<int>& prompt, int count,
              std::vector<int>* result) {
    LoadedModel loaded(path);
    const Model& model = loaded.model;
    State state;
    Work work;
    // prefill 完成后，work.logits 已是“最后一个 prompt token 后的下一个 token”分数。
    for (int token : prompt) forward(model, state, token, work);
    for (int step = 0; step < count; ++step) {
        const int next = argmax(work.logits);
        // 普通 text EOS 与 chat assistant 回合结束符都不应返回给用户。
        if (next == 248044 || next == 248046) break;
        result->push_back(next);  // 将输出 token 也作为下一次 decode 的输入。
        if (step + 1 < count) forward(model, state, next, work);  // decode
    }
}

// 无 checkpoint 自测：BF16(1.25)->1.25；argmax([1,3,2])=1；
// RMSNorm([3,4]) 使用 mean(x^2)=12.5，输入输出 shape 都是 [2]。
void self_test() {
    // 这是无需下载模型的微型单元测试；真实权重的端到端回归见 make oracle-test。
    assert(std::fabs(f32(bf16(1.25f)) - 1.25f) < 1e-6f);
    const std::vector<FP32> values = {1.0f, 3.0f, 2.0f};
    assert(argmax(values) == 1);
    FP32 x[] = {3.0f, 4.0f};
    const BF16 scale[] = {bf16(0.0f), bf16(0.0f)};
    FP32 y[2] = {};
    rms(x, scale, 2, y);
    assert(std::fabs(y[0] - 3.0f / std::sqrt(12.5f + EPS)) < 1e-5f);
    std::puts("self-test: passed (BF16, argmax, RMSNorm)");
}

// 打印四种命令行模式；不涉及 tensor 和 shape。
void usage(const char* program) {
    std::printf("usage: %s --self-test\n", program);
    std::printf("       %s --forward <qwen35-0.8b.bin> <id,id,...>\n", program);
    std::printf("       %s --logits <qwen35-0.8b.bin> <id,id,...>\n", program);
    std::printf("       %s --generate <qwen35-0.8b.bin> <id,id,...> <new-tokens>\n", program);
}

}  // namespace qwen35

// 命令行入口：把参数路由到 self-test、forward、logits 或 generate。
int main(int argc, char** argv) {
    using namespace qwen35;
    if (argc == 1 || std::strcmp(argv[1], "--self-test") == 0) { self_test(); return 0; }
    if (std::strcmp(argv[1], "--forward") == 0 && argc == 4) {
        // --forward 只做 prefill 并报告 next-token；它便于与官方 reference 比数值。
        LoadedModel loaded(argv[2]); State state; Work work;
        for (int token : parse_ids(argv[3])) forward(loaded.model, state, token, work);
        const int next = argmax(work.logits);
        std::printf("next token: %d, logit: %.6f\n", next, work.logits[next]);
        return 0;
    }
    if (std::strcmp(argv[1], "--logits") == 0 && argc == 4) {
        // 开发期数值 oracle：对一个完整 prefill prompt 输出 V=248320 个 logits。
        LoadedModel loaded(argv[2]); State state; Work work;
        for (int token : parse_ids(argv[3])) forward(loaded.model, state, token, work);
        dump_logits(work.logits);
        return 0;
    }
    if (std::strcmp(argv[1], "--generate") == 0 && argc == 5) {
        // --generate 输出 token ids。用官方 tokenizer decode 后才会得到 UTF-8 文本。
        std::vector<int> output;
        generate(argv[2], parse_ids(argv[3]), std::atoi(argv[4]), &output);
        std::printf("generated:");
        for (int token : output) std::printf(" %d", token);
        std::putchar('\n');
        return 0;
    }
    usage(argv[0]);
    return 1;
}

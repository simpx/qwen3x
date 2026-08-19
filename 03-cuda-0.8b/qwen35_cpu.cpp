// qwen35.cpp -- Qwen3.5-0.8B 的 plain C++ CPU reference。
//
// 此文件从 lessons/09 毕业而来：保留同样直接的模型数学，但把 token 生命周期明写为
// Model（mmap 权重）、State（跨 token 的 KV/GDN/conv）和 Work（本次 forward scratch）。
// 没有 Tensor、Backend、Operator 或 class hierarchy。真正的执行路径仍是一眼可见的：
//
//   prefill(tokens) / decode(token) -> forward(token)
//       -> embedding -> 24 x [DeltaNet 或 attention, SwiGLU] -> logits
//
// Stage 1 的官方 HF vectors 是本文件的数值裁判。--trace-logits 将每一步完整 vocabulary
// logits 写成原始 float32，供 test_cpu.py 比较；--state-check 验证两种同义写法留下相同 state。

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
constexpr float EPS = 1e-6f, THETA = 10000000.0f;
using B = uint16_t;  // 一个 BF16 权重元素的原始 bit；运算时立即转为 FP32。

[[noreturn]] void die(const char* text) {
    std::fprintf(stderr, "qwen35: %s\n", text);
    std::exit(1);
}

float f32(B value) {
    // BF16 与 FP32 共用最高 16 bit；左移后按位复制即可得到精确的 FP32 表示。
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

B bf16(float value) {
    // 只用于 self-test。真实模型权重由 09_pack_weights.py 原样复制，不会走这里。
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t bias = 0x7fffu + ((bits >> 16) & 1u);  // round-to-nearest-even
    return static_cast<B>((bits + bias) >> 16);
}

float sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + std::exp(-x));
    const float e = std::exp(x);
    return e / (1.0f + e);
}

float silu(float x) { return x * sigmoid(x); }  // SwiGLU 与 DeltaNet gate 共用的 SiLU。
float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

// pack_weights.py 的文件开头是 [8-byte magic, u32 version, u32 reserved]，随后每个
// tensor 都按 64 字节对齐。这里没有 tensor name/shape 元数据：本程序和转换器
// 都固定为同一个模型，Reader 的 take() 顺序就是简化后的格式 schema。
// mmap 只建立虚拟映射；操作系统按真正访问到的权重页加载。
struct File {
    int fd = -1;
    size_t size = 0;
    const uint8_t* data = nullptr;

    explicit File(const char* path) {
        fd = open(path, O_RDONLY);
        if (fd < 0) die("cannot open model.bin");
        struct stat info {};
        if (fstat(fd, &info) || info.st_size < 16) die("bad model.bin");
        size = static_cast<size_t>(info.st_size);
        data = static_cast<const uint8_t*>(mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
        if (data == MAP_FAILED) die("mmap model.bin failed");
        if (std::memcmp(data, "Q35COUR\0", 8) != 0) die("wrong model.bin magic; run pack_weights.py");
    }
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
    explicit Reader(const File& file) : begin(file.data), cursor(file.data + 16), end(file.data + file.size) {}

    void align() {
        // 必须在每个 tensor 之前与转换器做相同的向上取整，否则后续所有指针都会错位。
        const size_t offset = static_cast<size_t>(cursor - begin);
        cursor += (64 - offset % 64) % 64;
    }
    template <typename T> const T* take(size_t count) {
        align();
        const size_t bytes = count * sizeof(T);
        if (cursor > end || bytes > static_cast<size_t>(end - cursor)) die("truncated model.bin");
        // model.bin 是本机 little-endian、按自然对齐读取的教学格式；不是可移植交换格式。
        const T* result = reinterpret_cast<const T*>(cursor);
        cursor += bytes;
        return result;
    }
};

// Linear 始终表示没有 bias 的 row-major W[rows,cols]。指针直接指向 mmap 权重页。
struct Linear { const B* w = nullptr; int rows = 0, cols = 0; };
struct Delta {
    // DeltaNet 的四个输入投影、depthwise conv 参数、衰减参数、head norm 和输出投影。
    Linear qkv, z, a, b, out;
    const B* conv = nullptr;
    const float* alog = nullptr;
    const B* dt = nullptr;
    const float* norm = nullptr;
};
struct Attention {
    // q 同时产出 query 和 attention gate，所以有 2*AS 行；k/v 是 GQA 的小宽度。
    Linear q, k, v, out;
    const B* qnorm = nullptr;
    const B* knorm = nullptr;
};
struct Layer {
    // 每一层共有输入 RMSNorm、mixer 分支、post-mixer RMSNorm 和 MLP；delta/a 只有
    // 其中一个被填充。bool 的目的只是保持 forward 的 if 一眼可读。
    bool delta = false;
    const B* input_norm = nullptr;
    const B* post_norm = nullptr;
    Linear gate, up, down;
    Delta d;
    Attention a;
};

struct Model {
    File file;
    Reader reader;
    // 同一张 [V,H] 表既用于开头的 token -> hidden 查表，也在结尾作为 tied lm_head：
    // final hidden 与表的每一行点积，得到 V 个“下一个 token”分数。
    const B* embedding = nullptr;
    const B* final_norm = nullptr;
    std::array<Layer, N> layer {};

    explicit Model(const char* path) : file(path), reader(file) {
        // 以下读取顺序必须逐项匹配 pack_weights.py::expected_tensors()。先读所有层共享的
        // embedding/final norm，再顺序读第 0..23 层，故不需要 map<string,tensor>。
        embedding = reader.take<B>(static_cast<size_t>(V) * H);
        final_norm = reader.take<B>(H);
        for (int index = 0; index < N; ++index) {
            Layer& l = layer[index];
            // Qwen3.5 hybrid 的固定 4 层周期：0,1,2 是 DeltaNet，3 是 full attention。
            l.delta = index % 4 != 3;
            l.input_norm = reader.take<B>(H);
            if (l.delta) {
                l.d.qkv = {reader.take<B>(static_cast<size_t>(DQKV) * H), DQKV, H};
                l.d.z = {reader.take<B>(static_cast<size_t>(DO) * H), DO, H};
                l.d.a = {reader.take<B>(static_cast<size_t>(VH) * H), VH, H};
                l.d.b = {reader.take<B>(static_cast<size_t>(VH) * H), VH, H};
                l.d.conv = reader.take<B>(static_cast<size_t>(DQKV) * CK);
                l.d.alog = reader.take<float>(VH);
                l.d.dt = reader.take<B>(VH);
                l.d.norm = reader.take<float>(VD);
                l.d.out = {reader.take<B>(static_cast<size_t>(H) * DO), H, DO};
            } else {
                l.a.q = {reader.take<B>(static_cast<size_t>(2 * AS) * H), 2 * AS, H};
                l.a.k = {reader.take<B>(static_cast<size_t>(KVS) * H), KVS, H};
                l.a.v = {reader.take<B>(static_cast<size_t>(KVS) * H), KVS, H};
                l.a.qnorm = reader.take<B>(AD);
                l.a.knorm = reader.take<B>(AD);
                l.a.out = {reader.take<B>(static_cast<size_t>(H) * AS), H, AS};
            }
            l.post_norm = reader.take<B>(H);
            // 每种 mixer 后面都是同一套 SwiGLU MLP。
            l.gate = {reader.take<B>(static_cast<size_t>(I) * H), I, H};
            l.up = {reader.take<B>(static_cast<size_t>(I) * H), I, H};
            l.down = {reader.take<B>(static_cast<size_t>(H) * I), H, I};
        }
    }
};

// 线性层是 row-major W[rows, cols] 与向量 x[cols] 的最直白 GEMV。
// 这是 CPU correctness reference，不是快速 GEMM：batch=1 时每次只算一个向量。
void mv(const Linear& w, const float* x, float* y) {
    for (int row = 0; row < w.rows; ++row) {
        float sum = 0.0f;
        const B* weight = w.w + static_cast<size_t>(row) * w.cols;
        for (int col = 0; col < w.cols; ++col) sum += f32(weight[col]) * x[col];
        y[row] = sum;
    }
}

void embed(const B* table, int token, float* out) {
    if (token < 0 || token >= V) die("token outside vocabulary");
    // embedding table 的 shape 是 [V,H]；token id 唯一决定应复制的那一行。
    const B* row = table + static_cast<size_t>(token) * H;
    for (int i = 0; i < H; ++i) out[i] = f32(row[i]);
}

// Qwen ordinary RMSNorm 使用 (1 + weight)，而不是常见的直接 weight。
void rms(const float* x, const B* weight, int n, float* out) {
    // RMSNorm(x)_i = x_i / RMS(x) * (1 + weight_i)。它逐 token、逐向量工作，
    // 不跨 position，也不减均值。
    float square = 0.0f;
    for (int i = 0; i < n; ++i) square += x[i] * x[i];
    const float scale = 1.0f / std::sqrt(square / n + EPS);
    for (int i = 0; i < n; ++i) out[i] = x[i] * scale * (1.0f + f32(weight[i]));
}

void gated_rms(const float* x, const float* weight, const float* z, float* out) {
    // DeltaNet head 内部专用：readout [VD] 先 RMSNorm，再乘 learned norm.weight 和
    // SiLU(z)。这里的 weight 是 checkpoint 中的 F32，和 ordinary RMSNorm 的 1+w
    // 约定不同；不要把两个 norm 函数合并后丢掉这一区别。
    float square = 0.0f;
    for (int i = 0; i < VD; ++i) square += x[i] * x[i];
    const float scale = 1.0f / std::sqrt(square / VD + EPS);
    for (int i = 0; i < VD; ++i) out[i] = x[i] * scale * weight[i] * silu(z[i]);
}

void l2(float* x, int n) {
    // L2Norm 令 q/k 的 Euclidean norm 接近 1；与 RMSNorm 的分母定义不同。
    float square = 0.0f;
    for (int i = 0; i < n; ++i) square += x[i] * x[i];
    const float scale = 1.0f / std::sqrt(square + EPS);
    for (int i = 0; i < n; ++i) x[i] *= scale;
}

void add(float* x, const float* branch) {
    // 两个 residual 都是 [H]+[H] 的原地逐元素加法。
    for (int i = 0; i < H; ++i) x[i] += branch[i];
}

// 对一个 head 的前 RD=64 通道执行 Qwen half-rotation RoPE。head 的其余 AD-RD
// 通道保持原样；Q/K 都做相同 position 的旋转，V 永远不旋转。
void rope(float* x, int position) {
    float old[RD];
    std::memcpy(old, x, sizeof(old));
    for (int i = 0; i < RD / 2; ++i) {
        const float angle = position / std::pow(THETA, 2.0f * i / RD);
        const float c = std::cos(angle), s = std::sin(angle);
        x[i] = old[i] * c - old[i + RD / 2] * s;
        x[i + RD / 2] = old[i + RD / 2] * c + old[i] * s;
    }
}

struct State {
    int position = 0;
    // 对 Delta layer：conv 是 [DQKV,CK-1]，recurrent 是 [VH,KD,VD]；两者固定大小。
    // 对 attention layer：keys/values 每 token append 一次，逻辑 shape 是
    // [tokens,KVH,AD]，故随 context 线性增长。非对应的 layer vector 保持为空。
    std::array<std::vector<float>, N> conv, recurrent, keys, values;
    State() {
        for (int layer = 0; layer < N; ++layer) {
            if (layer % 4 != 3) {
                conv[layer].assign(static_cast<size_t>(DQKV) * (CK - 1), 0.0f);
                recurrent[layer].assign(static_cast<size_t>(VH) * KD * VD, 0.0f);
            }
        }
    }
};

struct Work {
    // 一次 forward 重复使用的临时 FP32 buffer。它们不是跨 token state：下一 token
    // 可以覆盖；真正需要保留的是上面的 State。logits 是唯一长度 V 的工作区。
    std::vector<float> h = std::vector<float>(H), n = std::vector<float>(H), mix = std::vector<float>(H);
    std::vector<float> gate = std::vector<float>(I), up = std::vector<float>(I), logits = std::vector<float>(V);
    std::vector<float> qkv = std::vector<float>(DQKV), z = std::vector<float>(DO);
    std::vector<float> da = std::vector<float>(VH), db = std::vector<float>(VH);
    std::vector<float> dq = std::vector<float>(DO), dk = std::vector<float>(DO), dout = std::vector<float>(DO);
    std::vector<float> aqp = std::vector<float>(2 * AS), aq = std::vector<float>(AS), ag = std::vector<float>(AS);
    std::vector<float> ak = std::vector<float>(KVS), av = std::vector<float>(KVS), ao = std::vector<float>(AS);
};

// DeltaNet 卷积是逐通道的 causal depthwise convolution；state 只保存 CK-1 个过去输入。
// weight 的布局是 [channel, CK]，当前输入乘最后一个权重；history 从最旧到最新。
void conv_step(float* x, const B* weight, std::vector<float>& history) {
    for (int channel = 0; channel < DQKV; ++channel) {
        float* past = history.data() + static_cast<size_t>(channel) * (CK - 1);
        float sum = x[channel] * f32(weight[channel * CK + CK - 1]);
        for (int i = 0; i < CK - 1; ++i) sum += past[i] * f32(weight[channel * CK + i]);
        for (int i = 0; i < CK - 2; ++i) past[i] = past[i + 1];
        past[CK - 2] = x[channel];
        x[channel] = silu(sum);
    }
}

// 一个 value head 的四步 Delta rule。调用者已经切到某个 head 的 S=[KD,VD]。
// 这份 state 是 DeltaNet 不需要 KV cache 的原因：context 再长，S 的元素数不变。
void delta_rule(const float* q, const float* k, const float* v, float log_decay, float beta,
                float* state, float* out) {
    // (1) S <- exp(log_decay) S：遗忘旧记忆。log_decay 在模型中被构造成负数。
    const float decay = std::exp(log_decay);
    for (int i = 0; i < KD * VD; ++i) state[i] *= decay;
    // (2) memory <- k^T S：当前 key 查询“这个位置已记住的 value”。
    float memory[VD] = {};
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

void deltanet(const Delta& d, std::vector<float>& conv, std::vector<float>& recurrent,
              const float* input, Work& w, float* out) {
    // 输入已经过 layer input RMSNorm。四个投影都从同一个 normalized hidden 得到。
    mv(d.qkv, input, w.qkv.data());
    mv(d.z, input, w.z.data());
    mv(d.a, input, w.da.data());
    mv(d.b, input, w.db.data());
    // 只有拼接 Q/K/V 经过 causal depthwise conv；z、a、b 不经过它。
    conv_step(w.qkv.data(), d.conv, conv);

    // 固定布局拆包，不创建 view 类：small_q[KH,KD]、small_k[KH,KD]、value[VH,VD]。
    const float* small_q = w.qkv.data();
    const float* small_k = small_q + DQK;
    const float* value = small_k + DQK;
    for (int head = 0; head < VH; ++head) {
        // 27B 的 48 个 value head 会复用 16 个 small Q/K head；0.8B 此处是 1:1。
        const int qk_head = head / (VH / KH);
        float* q = w.dq.data() + head * KD;
        float* k = w.dk.data() + head * KD;
        std::memcpy(q, small_q + qk_head * KD, KD * sizeof(float));
        std::memcpy(k, small_k + qk_head * KD, KD * sizeof(float));
        l2(q, KD);
        l2(k, KD);
        // Q 的额外 1/sqrt(KD) 是此实现的 Delta rule 缩放约定。
        for (int i = 0; i < KD; ++i) q[i] /= std::sqrt(static_cast<float>(KD));
        // b 经过 sigmoid 得到写入比例 beta in (0,1)；a/A_log/dt_bias 决定衰减速度。
        const float beta = sigmoid(w.db[head]);
        const float log_decay = -std::exp(d.alog[head]) * softplus(w.da[head] + f32(d.dt[head]));
        delta_rule(q, k, value + head * VD, log_decay, beta,
                   recurrent.data() + static_cast<size_t>(head) * KD * VD,
                   w.dout.data() + head * VD);
        // 每个 value head 的 readout 在各自 VD 段内 norm/gate，之后再拼接为 [DO]。
        gated_rms(w.dout.data() + head * VD, d.norm, w.z.data() + head * VD, w.dout.data() + head * VD);
    }
    mv(d.out, w.dout.data(), out);
}

void attention(const Attention& a, std::vector<float>& keys, std::vector<float>& values,
               int position, const float* input, Work& w, float* out) {
    // q_proj 输出 [Q, gate] 两个 [AS] 向量；k/v 分别是 GQA 的 [KVS] 向量。
    mv(a.q, input, w.aqp.data());
    mv(a.k, input, w.ak.data());
    mv(a.v, input, w.av.data());
    for (int head = 0; head < AH; ++head) {
        // QNorm 和 RoPE 是逐 query head 操作；gate 不 norm、不旋转，留到 attention
        // readout 后才做 sigmoid 门控。
        std::memcpy(w.aq.data() + head * AD, w.aqp.data() + head * 2 * AD, AD * sizeof(float));
        std::memcpy(w.ag.data() + head * AD, w.aqp.data() + head * 2 * AD + AD, AD * sizeof(float));
        rms(w.aq.data() + head * AD, a.qnorm, AD, w.aq.data() + head * AD);
        rope(w.aq.data() + head * AD, position);
    }
    for (int head = 0; head < KVH; ++head) {
        // K 与 Q 使用同一 RoPE position；V 不依赖位置，因此不做 RoPE。
        rms(w.ak.data() + head * AD, a.knorm, AD, w.ak.data() + head * AD);
        rope(w.ak.data() + head * AD, position);
    }
    // 先 append 再读，等价于 causal attention 允许 token t 读取 0..t（包括自己）。
    keys.insert(keys.end(), w.ak.begin(), w.ak.end());
    values.insert(values.end(), w.av.begin(), w.av.end());
    const int tokens = static_cast<int>(keys.size() / KVS);
    const float scale = 1.0f / std::sqrt(static_cast<float>(AD));
    std::vector<float> score(tokens);  // 单 head 的 score buffer，逐 head 复用。

    for (int head = 0; head < AH; ++head) {
        // GQA：8 个 Q head 按 4:1 分为两组，每组共享一个 K/V head。
        const int kv_head = head / (AH / KVH);
        float maximum = -std::numeric_limits<float>::infinity();
        for (int token = 0; token < tokens; ++token) {
            float dot = 0.0f;
            const float* key = keys.data() + (static_cast<size_t>(token) * KVH + kv_head) * AD;
            for (int i = 0; i < AD; ++i) dot += w.aq[head * AD + i] * key[i];
            score[token] = dot * scale;
            maximum = std::max(maximum, score[token]);
        }
        // stable softmax：同时复用 score 数组存 exp(score-maximum)。
        float total = 0.0f;
        for (float& value : score) { value = std::exp(value - maximum); total += value; }
        float* result = w.ao.data() + head * AD;
        std::fill(result, result + AD, 0.0f);
        for (int token = 0; token < tokens; ++token) {
            const float probability = score[token] / total;
            const float* value = values.data() + (static_cast<size_t>(token) * KVH + kv_head) * AD;
            for (int i = 0; i < AD; ++i) result[i] += probability * value[i];
        }
    }
    // Qwen Gated Attention：attention readout 与 q_proj 的另一半 gate 逐元素相乘。
    for (int i = 0; i < AS; ++i) w.ao[i] *= sigmoid(w.ag[i]);
    mv(a.out, w.ao.data(), out);
}

void mlp(const Layer& l, const float* input, Work& w, float* out) {
    // SwiGLU：gate/up 各投影到 [I]，逐元素 SiLU(gate)*up，再 down_proj 回 [H]。
    mv(l.gate, input, w.gate.data());
    mv(l.up, input, w.up.data());
    for (int i = 0; i < I; ++i) w.gate[i] = silu(w.gate[i]) * w.up[i];
    mv(l.down, w.gate.data(), out);
}

// 这就是课程最终应从上读到下的完整 token forward。调用一次只处理一个 token；
// prompt 的 prefill 只是对 prompt ids 连续调用它，decode 则在每次 argmax 后再调用。
void forward(const Model& m, State& s, int token, Work& w) {
    if (s.position >= MAX_TOKENS) die("CPU reference supports at most 4096 tokens");
    embed(m.embedding, token, w.h.data());  // hidden[H]，本 token 的 layer-0 输入。
    for (int index = 0; index < N; ++index) {
        const Layer& l = m.layer[index];
        rms(w.h.data(), l.input_norm, H, w.n.data());  // pre-mixer RMSNorm。
        if (l.delta) deltanet(l.d, s.conv[index], s.recurrent[index], w.n.data(), w, w.mix.data());
        else attention(l.a, s.keys[index], s.values[index], s.position, w.n.data(), w, w.mix.data());
        add(w.h.data(), w.mix.data());                 // 第一个 residual。
        rms(w.h.data(), l.post_norm, H, w.n.data());   // pre-MLP RMSNorm。
        mlp(l, w.n.data(), w, w.mix.data());
        add(w.h.data(), w.mix.data());                 // 第二个 residual，进入下一 layer。
    }
    rms(w.h.data(), m.final_norm, H, w.n.data());      // final hidden：它已编码到目前为止的上下文。
    // lm_head：为词表的每一个候选 token 各算一个 logit。这里 W 直接重用 m.embedding：
    // logits[v] = dot(final_hidden, embedding[v])。这叫 tied embedding，避免再存一份 [V,H]
    // 输出矩阵；它与开头 embed() 的“按 token id 取同一张表的一行”正好相对。
    mv({m.embedding, V, H}, w.n.data(), w.logits.data());
    ++s.position;
}

// prefill 不是另一套模型算法：它只是把 prompt 的每一个 id 依次喂入同一个 forward。
// 每次 forward 都会 append attention KV、更新 DeltaNet recurrent/conv state；结束时
// w.logits 预测最后一个 prompt token 后的下一个 token。
void prefill(const Model& m, State& s, const std::vector<int>& tokens, Work& w) {
    if (tokens.empty()) die("prefill prompt is empty");
    for (int token : tokens) forward(m, s, token, w);
}

// decode 同样只是一个有语义的名字：token 是上一步采样得到、重新喂给模型的 id。
// 它保留传入的 State，因此绝不会重新计算 prompt。
void decode(const Model& m, State& s, int token, Work& w) {
    forward(m, s, token, w);
}

int argmax(const std::vector<float>& values) {
    // deterministic greedy sampling；课程以它避免 temperature/random seed 的额外变量。
    return static_cast<int>(std::max_element(values.begin(), values.end()) - values.begin());
}

void dump_logits(const std::vector<float>& values) {
    // 只给开发期 Stage 1 vector test 使用：一行一个 FP32 logit，便于它比较完整词表，
    // 而不是只看 argmax 后“文字看起来正常”。正常的 --forward / --generate 不会走这里。
    for (float value : values) std::printf("%.9g\n", value);
}

void trace_logits(const Model& m, State& s, const std::vector<int>& tokens, Work& w, const char* path) {
    // 测试交换格式故意朴素：连续写 N 个 [V] float32 rows；N 来自输入 token 数，因而无需
    // 再包一层 Tensor/archive 格式。这里只用于 Stage 1 Python vectors 与 Stage 2 CPU 对照。
    FILE* output = std::fopen(path, "wb");
    if (!output) die("cannot open trace output");
    for (int token : tokens) {
        forward(m, s, token, w);
        if (std::fwrite(w.logits.data(), sizeof(float), V, output) != static_cast<size_t>(V)) {
            std::fclose(output);
            die("cannot write trace output");
        }
    }
    if (std::fclose(output) != 0) die("cannot close trace output");
}

bool same_vector(const std::vector<float>& left, const std::vector<float>& right) {
    return left.size() == right.size() &&
           (left.empty() || std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0);
}

bool same_state(const State& left, const State& right) {
    if (left.position != right.position) return false;
    for (int layer = 0; layer < N; ++layer) {
        if (!same_vector(left.conv[layer], right.conv[layer]) ||
            !same_vector(left.recurrent[layer], right.recurrent[layer]) ||
            !same_vector(left.keys[layer], right.keys[layer]) ||
            !same_vector(left.values[layer], right.values[layer])) return false;
    }
    return true;
}

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

void generate(const Model& model, State& state, Work& work, const std::vector<int>& prompt, int count,
              std::vector<int>* result) {
    // prefill 完成后，work.logits 已是“最后一个 prompt token 后的下一个 token”分数。
    prefill(model, state, prompt, work);
    for (int step = 0; step < count; ++step) {
        const int next = argmax(work.logits);
        // 普通 text EOS 与 chat assistant 回合结束符都不应返回给用户。
        if (next == 248044 || next == 248046) break;
        result->push_back(next);  // 将输出 token 也作为下一次 decode 的输入。
        if (step + 1 < count) decode(model, state, next, work);  // decode
    }
}

void self_test() {
    // 这是无需下载模型的微型单元测试；真实权重的回归见本目录的 make test。
    assert(std::fabs(f32(bf16(1.25f)) - 1.25f) < 1e-6f);
    const std::vector<float> values = {1.0f, 3.0f, 2.0f};
    assert(argmax(values) == 1);
    float x[] = {3.0f, 4.0f};
    const B scale[] = {bf16(0.0f), bf16(0.0f)};
    float y[2] = {};
    rms(x, scale, 2, y);
    assert(std::fabs(y[0] - 3.0f / std::sqrt(12.5f + EPS)) < 1e-5f);
    std::puts("self-test: passed (BF16, argmax, RMSNorm)");
}

void usage(const char* program) {
    std::printf("usage: %s --self-test\n", program);
    std::printf("       %s --forward <qwen35-0.8b.bin> <id,id,...>\n", program);
    std::printf("       %s --logits <qwen35-0.8b.bin> <id,id,...>\n", program);
    std::printf("       %s --trace-logits <qwen35-0.8b.bin> <id,id,...> <out.f32>\n", program);
    std::printf("       %s --state-check <qwen35-0.8b.bin> <prefill-ids> <decode-ids>\n", program);
    std::printf("       %s --generate <qwen35-0.8b.bin> <id,id,...> <new-tokens>\n", program);
}

}  // namespace qwen35

int main(int argc, char** argv) {
    using namespace qwen35;
    if (argc == 1 || std::strcmp(argv[1], "--self-test") == 0) { self_test(); return 0; }
    if (std::strcmp(argv[1], "--forward") == 0 && argc == 4) {
        // --forward 只做 prefill 并报告 next-token；它便于与官方 reference 比数值。
        Model model(argv[2]); State state; Work work;
        prefill(model, state, parse_ids(argv[3]), work);
        const int next = argmax(work.logits);
        std::printf("next token: %d, logit: %.6f\n", next, work.logits[next]);
        return 0;
    }
    if (std::strcmp(argv[1], "--logits") == 0 && argc == 4) {
        // 开发期数值 oracle：对一个完整 prefill prompt 输出 V=248320 个 logits。
        Model model(argv[2]); State state; Work work;
        prefill(model, state, parse_ids(argv[3]), work);
        dump_logits(work.logits);
        return 0;
    }
    if (std::strcmp(argv[1], "--trace-logits") == 0 && argc == 5) {
        Model model(argv[2]); State state; Work work;
        trace_logits(model, state, parse_ids(argv[3]), work, argv[4]);
        return 0;
    }
    if (std::strcmp(argv[1], "--state-check") == 0 && argc == 5) {
        Model model(argv[2]);
        const std::vector<int> prompt = parse_ids(argv[3]);
        const std::vector<int> suffix = parse_ids(argv[4]);
        State via_api, direct;
        Work api_work, direct_work;
        prefill(model, via_api, prompt, api_work);
        for (int token : suffix) decode(model, via_api, token, api_work);
        for (int token : prompt) forward(model, direct, token, direct_work);
        for (int token : suffix) forward(model, direct, token, direct_work);
        if (!same_state(via_api, direct) || !same_vector(api_work.logits, direct_work.logits)) {
            die("prefill/decode state differs from direct forward");
        }
        std::printf("state-check: passed (%d prompt + %d decode tokens)\n",
                    static_cast<int>(prompt.size()), static_cast<int>(suffix.size()));
        return 0;
    }
    if (std::strcmp(argv[1], "--generate") == 0 && argc == 5) {
        // --generate 输出 token ids。用官方 tokenizer decode 后才会得到 UTF-8 文本。
        Model model(argv[2]); State state; Work work;
        std::vector<int> output;
        generate(model, state, work, parse_ids(argv[3]), std::atoi(argv[4]), &output);
        std::printf("generated:");
        for (int token : output) std::printf(" %d", token);
        std::putchar('\n');
        return 0;
    }
    usage(argv[0]);
    return 1;
}

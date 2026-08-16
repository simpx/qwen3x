// qwen38.cpp capstone -- 一个固定 Qwen3.5-0.8B text backbone 的 CPU 推理器。
//
// 这是课程的最后一课：没有 Tensor 类、没有算子注册表、没有通用模型兼容层。
// convert.py 已按本文件读取的顺序排好权重，因此这里从上到下就是一次 token
// 的真实 Qwen hybrid forward。Qwen3.8-27B 与它有同样的 3 DeltaNet : 1
// full-attention 栈；差别主要是层数和矩阵尺寸。

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

#ifdef QWEN38_WITH_TOKENIZER
#include "qwen38_tokenizer.h"
#endif

namespace qwen38_course {

// 这些常数直接来自 Qwen/Qwen3.5-0.8B 的 text_config。不是运行时 Config；
// 换模型意味着另写一份课程，而不是把本文件演化成通用推理框架。
constexpr int V = 248320, H = 1024, I = 3584, N = 24;
constexpr int AH = 8, KVH = 2, AD = 256, RD = 64;
constexpr int KH = 16, VH = 16, KD = 128, VD = 128, CK = 4;
constexpr int AS = AH * AD, KVS = KVH * AD, DQK = KH * KD, DO = VH * VD, DQKV = 2 * DQK + DO;
constexpr float EPS = 1e-6f, THETA = 10000000.0f;
using B = uint16_t;  // 一个 BF16 权重元素的原始 bit。

[[noreturn]] void die(const char* text) {
    std::fprintf(stderr, "qwen38: %s\n", text);
    std::exit(1);
}

float f32(B value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

B bf16(float value) {
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

float silu(float x) { return x * sigmoid(x); }
float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

// convert.py 的文件开头是固定 16 字节，随后每个 tensor 都按 64 字节对齐。
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
        if (std::memcmp(data, "Q38COUR\0", 8) != 0) die("wrong model.bin magic; run convert.py");
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
        const size_t offset = static_cast<size_t>(cursor - begin);
        cursor += (64 - offset % 64) % 64;
    }
    template <typename T> const T* take(size_t count) {
        align();
        const size_t bytes = count * sizeof(T);
        if (cursor > end || bytes > static_cast<size_t>(end - cursor)) die("truncated model.bin");
        const T* result = reinterpret_cast<const T*>(cursor);
        cursor += bytes;
        return result;
    }
};

struct Linear { const B* w = nullptr; int rows = 0, cols = 0; };
struct Delta {
    Linear qkv, z, a, b, out;
    const B* conv = nullptr;
    const float* alog = nullptr;
    const B* dt = nullptr;
    const float* norm = nullptr;
};
struct Attention {
    Linear q, k, v, out;
    const B* qnorm = nullptr;
    const B* knorm = nullptr;
};
struct Layer {
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
    const B* embedding = nullptr;
    const B* final_norm = nullptr;
    std::array<Layer, N> layer {};

    explicit Model(const char* path) : file(path), reader(file) {
        embedding = reader.take<B>(static_cast<size_t>(V) * H);
        final_norm = reader.take<B>(H);
        for (int index = 0; index < N; ++index) {
            Layer& l = layer[index];
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
            l.gate = {reader.take<B>(static_cast<size_t>(I) * H), I, H};
            l.up = {reader.take<B>(static_cast<size_t>(I) * H), I, H};
            l.down = {reader.take<B>(static_cast<size_t>(H) * I), H, I};
        }
    }
};

// 线性层是 row-major W[rows, cols] 与向量 x[cols] 的最直白 GEMV。
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
    const B* row = table + static_cast<size_t>(token) * H;
    for (int i = 0; i < H; ++i) out[i] = f32(row[i]);
}

// Qwen ordinary RMSNorm 使用 (1 + weight)，而不是常见的直接 weight。
void rms(const float* x, const B* weight, int n, float* out) {
    float square = 0.0f;
    for (int i = 0; i < n; ++i) square += x[i] * x[i];
    const float scale = 1.0f / std::sqrt(square / n + EPS);
    for (int i = 0; i < n; ++i) out[i] = x[i] * scale * (1.0f + f32(weight[i]));
}

void gated_rms(const float* x, const float* weight, const float* z, float* out) {
    float square = 0.0f;
    for (int i = 0; i < VD; ++i) square += x[i] * x[i];
    const float scale = 1.0f / std::sqrt(square / VD + EPS);
    for (int i = 0; i < VD; ++i) out[i] = x[i] * scale * weight[i] * silu(z[i]);
}

void l2(float* x, int n) {
    float square = 0.0f;
    for (int i = 0; i < n; ++i) square += x[i] * x[i];
    const float scale = 1.0f / std::sqrt(square + EPS);
    for (int i = 0; i < n; ++i) x[i] *= scale;
}

void add(float* x, const float* branch) {
    for (int i = 0; i < H; ++i) x[i] += branch[i];
}

// 对一个 head 的前 64 通道执行 Qwen half-rotation RoPE。
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
    std::vector<float> h = std::vector<float>(H), n = std::vector<float>(H), mix = std::vector<float>(H);
    std::vector<float> gate = std::vector<float>(I), up = std::vector<float>(I), logits = std::vector<float>(V);
    std::vector<float> qkv = std::vector<float>(DQKV), z = std::vector<float>(DO);
    std::vector<float> da = std::vector<float>(VH), db = std::vector<float>(VH);
    std::vector<float> dq = std::vector<float>(DO), dk = std::vector<float>(DO), dout = std::vector<float>(DO);
    std::vector<float> aqp = std::vector<float>(2 * AS), aq = std::vector<float>(AS), ag = std::vector<float>(AS);
    std::vector<float> ak = std::vector<float>(KVS), av = std::vector<float>(KVS), ao = std::vector<float>(AS);
};

// DeltaNet 卷积是逐通道的 causal depthwise convolution；state 只保存 CK-1 个过去输入。
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

// 一个 value head 的四步 Delta rule。S 的布局是 [head][key_dim][value_dim]。
void delta_rule(const float* q, const float* k, const float* v, float log_decay, float beta,
                float* state, float* out) {
    const float decay = std::exp(log_decay);
    for (int i = 0; i < KD * VD; ++i) state[i] *= decay;
    float memory[VD] = {};
    for (int value = 0; value < VD; ++value) {
        for (int key = 0; key < KD; ++key) memory[value] += k[key] * state[key * VD + value];
    }
    for (int key = 0; key < KD; ++key) {
        for (int value = 0; value < VD; ++value) {
            state[key * VD + value] += k[key] * beta * (v[value] - memory[value]);
        }
    }
    for (int value = 0; value < VD; ++value) {
        out[value] = 0.0f;
        for (int key = 0; key < KD; ++key) out[value] += q[key] * state[key * VD + value];
    }
}

void deltanet(const Delta& d, std::vector<float>& conv, std::vector<float>& recurrent,
              const float* input, Work& w, float* out) {
    mv(d.qkv, input, w.qkv.data());
    mv(d.z, input, w.z.data());
    mv(d.a, input, w.da.data());
    mv(d.b, input, w.db.data());
    conv_step(w.qkv.data(), d.conv, conv);

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
        for (int i = 0; i < KD; ++i) q[i] /= std::sqrt(static_cast<float>(KD));
        const float beta = sigmoid(w.db[head]);
        const float log_decay = -std::exp(d.alog[head]) * softplus(w.da[head] + f32(d.dt[head]));
        delta_rule(q, k, value + head * VD, log_decay, beta,
                   recurrent.data() + static_cast<size_t>(head) * KD * VD,
                   w.dout.data() + head * VD);
        gated_rms(w.dout.data() + head * VD, d.norm, w.z.data() + head * VD, w.dout.data() + head * VD);
    }
    mv(d.out, w.dout.data(), out);
}

void attention(const Attention& a, std::vector<float>& keys, std::vector<float>& values,
               int position, const float* input, Work& w, float* out) {
    mv(a.q, input, w.aqp.data());
    mv(a.k, input, w.ak.data());
    mv(a.v, input, w.av.data());
    for (int head = 0; head < AH; ++head) {
        std::memcpy(w.aq.data() + head * AD, w.aqp.data() + head * 2 * AD, AD * sizeof(float));
        std::memcpy(w.ag.data() + head * AD, w.aqp.data() + head * 2 * AD + AD, AD * sizeof(float));
        rms(w.aq.data() + head * AD, a.qnorm, AD, w.aq.data() + head * AD);
        rope(w.aq.data() + head * AD, position);
    }
    for (int head = 0; head < KVH; ++head) {
        rms(w.ak.data() + head * AD, a.knorm, AD, w.ak.data() + head * AD);
        rope(w.ak.data() + head * AD, position);
    }
    keys.insert(keys.end(), w.ak.begin(), w.ak.end());
    values.insert(values.end(), w.av.begin(), w.av.end());
    const int tokens = static_cast<int>(keys.size() / KVS);
    const float scale = 1.0f / std::sqrt(static_cast<float>(AD));
    std::vector<float> score(tokens);

    for (int head = 0; head < AH; ++head) {
        const int kv_head = head / (AH / KVH);
        float maximum = -std::numeric_limits<float>::infinity();
        for (int token = 0; token < tokens; ++token) {
            float dot = 0.0f;
            const float* key = keys.data() + (static_cast<size_t>(token) * KVH + kv_head) * AD;
            for (int i = 0; i < AD; ++i) dot += w.aq[head * AD + i] * key[i];
            score[token] = dot * scale;
            maximum = std::max(maximum, score[token]);
        }
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
    for (int i = 0; i < AS; ++i) w.ao[i] *= sigmoid(w.ag[i]);
    mv(a.out, w.ao.data(), out);
}

void mlp(const Layer& l, const float* input, Work& w, float* out) {
    mv(l.gate, input, w.gate.data());
    mv(l.up, input, w.up.data());
    for (int i = 0; i < I; ++i) w.gate[i] = silu(w.gate[i]) * w.up[i];
    mv(l.down, w.gate.data(), out);
}

// 这就是课程最终应从上读到下的完整 token forward。
void forward(const Model& m, State& s, int token, Work& w) {
    if (s.position >= 2048) die("teaching capstone supports at most 2048 tokens");
    embed(m.embedding, token, w.h.data());
    for (int index = 0; index < N; ++index) {
        const Layer& l = m.layer[index];
        rms(w.h.data(), l.input_norm, H, w.n.data());
        if (l.delta) deltanet(l.d, s.conv[index], s.recurrent[index], w.n.data(), w, w.mix.data());
        else attention(l.a, s.keys[index], s.values[index], s.position, w.n.data(), w, w.mix.data());
        add(w.h.data(), w.mix.data());
        rms(w.h.data(), l.post_norm, H, w.n.data());
        mlp(l, w.n.data(), w, w.mix.data());
        add(w.h.data(), w.mix.data());
    }
    rms(w.h.data(), m.final_norm, H, w.n.data());
    mv({m.embedding, V, H}, w.n.data(), w.logits.data());  // 0.8B 的 lm_head 与 embedding tied。
    ++s.position;
}

int argmax(const std::vector<float>& values) {
    return static_cast<int>(std::max_element(values.begin(), values.end()) - values.begin());
}

std::vector<int> parse_ids(const char* text) {
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

void generate(const char* path, const std::vector<int>& prompt, int count, std::vector<int>* result) {
    Model model(path);
    State state;
    Work work;
    for (int token : prompt) forward(model, state, token, work);  // prefill
    for (int step = 0; step < count; ++step) {
        const int next = argmax(work.logits);
        if (next == 248044) break;  // Qwen3.5 text EOS
        result->push_back(next);
        if (step + 1 < count) forward(model, state, next, work);  // decode
    }
}

void self_test() {
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
    std::printf("       %s --forward <qwen38-0.8b.bin> <id,id,...>\n", program);
    std::printf("       %s --generate <qwen38-0.8b.bin> <id,id,...> <new-tokens>\n", program);
#ifdef QWEN38_WITH_TOKENIZER
    std::printf("       %s --generate-text <qwen38-0.8b.bin> <tokenizer-dir> <text> <new-tokens>\n", program);
#endif
}

}  // namespace qwen38_course

int main(int argc, char** argv) {
    using namespace qwen38_course;
    if (argc == 1 || std::strcmp(argv[1], "--self-test") == 0) { self_test(); return 0; }
    if (std::strcmp(argv[1], "--forward") == 0 && argc == 4) {
        Model model(argv[2]); State state; Work work;
        for (int token : parse_ids(argv[3])) forward(model, state, token, work);
        const int next = argmax(work.logits);
        std::printf("next token: %d, logit: %.6f\n", next, work.logits[next]);
        return 0;
    }
    if (std::strcmp(argv[1], "--generate") == 0 && argc == 5) {
        std::vector<int> output;
        generate(argv[2], parse_ids(argv[3]), std::atoi(argv[4]), &output);
        std::printf("generated:");
        for (int token : output) std::printf(" %d", token);
        std::putchar('\n');
        return 0;
    }
#ifdef QWEN38_WITH_TOKENIZER
    if (std::strcmp(argv[1], "--generate-text") == 0 && argc == 6) {
        qwen38::QwenTokenizer tokenizer(argv[3]);
        std::vector<int> output;
        generate(argv[2], tokenizer.encode(argv[4]), std::atoi(argv[5]), &output);
        std::printf("%s\n", tokenizer.decode(output).c_str());
        return 0;
    }
#endif
    usage(argv[0]);
    return 1;
}

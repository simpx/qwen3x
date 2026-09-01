// engine.cpp -- plain C++ Qwen3.5 CPU correctness engine.
// Model owns mapped weights, State owns history/cache, Work owns scratch memory.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <new>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "internal.h"
#include "model_config.h"
#include "qwen35.h"

namespace q35_backend {

using FP32 = float;
using BF16 = uint16_t;
constexpr FP32 EPS = 1e-6f, THETA = 10000000.0f;
constexpr int END_OF_TEXT_TOKEN = 248044, IM_END_TOKEN = 248046;
static_assert(sizeof(FP32) == 4 && std::numeric_limits<FP32>::is_iec559);

struct Linear { const BF16* w = nullptr; int rows = 0, cols = 0; };
struct DeltaWeights {
    Linear qkv, z, a, b, out;
    const BF16* conv = nullptr;
    const FP32* alog = nullptr;
    const BF16* dt = nullptr;
    const FP32* norm = nullptr;
};
struct AttentionWeights {
    Linear q, k, v, out;
    const BF16 *qnorm = nullptr, *knorm = nullptr;
};
struct Layer {
    const BF16 *input_norm = nullptr, *post_norm = nullptr;
    Linear gate, up, down;
    DeltaWeights delta;
    AttentionWeights attention;
};

// State/Work 各自只拥有少量连续内存；创建时切好指针，forward 中不再分配。
struct Storage {
    std::unique_ptr<FP32[]> data;
    size_t count = 0, used = 0;
    void allocate(size_t n) {
        data.reset(new (std::nothrow) FP32[n]);
        Q35_ASSERT(data, "FP32 allocation failed count=%zu", n);
        count = n;
    }
    FP32* take(size_t n) {
        Q35_ASSERT(used <= count && n <= count - used,
                   "storage used=%zu take=%zu count=%zu", used, n, count);
        FP32* result = data.get() + used;
        used += n;
        return result;
    }
};

struct Work {
    Storage storage;
    FP32 *hidden, *normalized, *branch, *logits, *ffn_gate, *ffn_up;
    FP32 *delta_qkv, *delta_z, *delta_a, *delta_b, *delta_q, *delta_k, *delta_output;
    FP32 *query_and_gate, *query, *attention_gate, *key, *value, *attention_output;
    explicit Work(const q35_model::ModelConfig& c) {
        const int DQKV = 2 * c.KH * c.KD + c.VH * c.VD;
        const int DO = c.VH * c.VD, AS = c.AH * c.AD, KVW = c.KVH * c.AD;
        storage.allocate(3ull * c.H + c.V + 2ull * c.I + DQKV + 4ull * DO +
                         2ull * c.VH + 5ull * AS + 2ull * KVW);
        hidden = storage.take(c.H);
        normalized = storage.take(c.H);
        branch = storage.take(c.H);
        logits = storage.take(c.V);
        ffn_gate = storage.take(c.I);
        ffn_up = storage.take(c.I);
        delta_qkv = storage.take(DQKV);
        delta_z = storage.take(DO);
        delta_a = storage.take(c.VH);
        delta_b = storage.take(c.VH);
        delta_q = storage.take(DO);
        delta_k = storage.take(DO);
        delta_output = storage.take(DO);
        query_and_gate = storage.take(2 * AS);
        query = storage.take(AS);
        attention_gate = storage.take(AS);
        key = storage.take(KVW);
        value = storage.take(KVW);
        attention_output = storage.take(AS);
        Q35_ASSERT(storage.used == storage.count, "Work storage layout mismatch");
    }
};

struct LayerState { FP32 *conv = nullptr, *memory = nullptr, *key = nullptr, *value = nullptr; };
struct State {
    const q35_model::ModelConfig* config;
    int position = 0, capacity, checkpoint_position = 0;
    std::unique_ptr<LayerState[]> layer;
    std::unique_ptr<FP32[]> score, checkpoint_recurrent, checkpoint_logits;
    Storage recurrent, kv;
    Work work;
    State(const q35_model::ModelConfig& c, int context) : config(&c), capacity(context), work(c) {
        // Delta layer 保存固定 recurrent state；Attention layer 保存随 context 增长的 KV。
        const int delta_layers = c.N - c.N / c.AI, attention_layers = c.N / c.AI;
        const size_t conv = static_cast<size_t>(2 * c.KH * c.KD + c.VH * c.VD) * (c.CK - 1);
        const size_t memory = static_cast<size_t>(c.VH) * c.KD * c.VD;
        const size_t cache = static_cast<size_t>(context) * c.KVH * c.AD;
        layer.reset(new (std::nothrow) LayerState[c.N]);
        score.reset(new (std::nothrow) FP32[context]);
        recurrent.allocate(static_cast<size_t>(delta_layers) * (conv + memory));
        kv.allocate(static_cast<size_t>(attention_layers) * 2 * cache);
        checkpoint_recurrent.reset(new (std::nothrow) FP32[recurrent.count]);
        checkpoint_logits.reset(new (std::nothrow) FP32[c.V]);
        Q35_ASSERT(layer && score && checkpoint_recurrent && checkpoint_logits,
                   "State allocation failed model=%s context=%d", c.name, context);
        for (int i = 0; i < c.N; ++i) {
            if (i % c.AI != c.AI - 1) {
                layer[i].conv = recurrent.take(conv);
                layer[i].memory = recurrent.take(memory);
            } else {
                layer[i].key = kv.take(cache);
                layer[i].value = kv.take(cache);
            }
        }
        Q35_ASSERT(recurrent.used == recurrent.count && kv.used == kv.count, "State layout mismatch");
        reset();
    }
    void reset() {
        position = 0;
        std::fill(recurrent.data.get(), recurrent.data.get() + recurrent.count, 0.0f);
    }
};

struct Model {
    int fd = -1;
    size_t file_size = 0;
    const uint8_t* file = nullptr;
    const q35_model::ModelConfig* config = nullptr;
    const BF16 *embedding = nullptr, *final_norm = nullptr;
    std::unique_ptr<Layer[]> layer;
    ~Model() {
        if (file) munmap(const_cast<uint8_t*>(file), file_size);
        if (fd >= 0) close(fd);
    }
    bool load(const char* path, const char** error);
};

// 小算子直接保留公式和显式 shape，CPU 是数值正确性的基线实现。
FP32 f32(BF16 value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    FP32 result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
FP32 sigmoid(FP32 x) {
    if (x >= 0.0f) return 1.0f / (1.0f + std::exp(-x));
    const FP32 e = std::exp(x); return e / (1.0f + e);
}
FP32 silu(FP32 x) { return x * sigmoid(x); }
FP32 softplus(FP32 x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

void embed(const BF16* table, int token, int H, FP32* out) {
    const BF16* row = table + static_cast<size_t>(token) * H;
    for (int i = 0; i < H; ++i) out[i] = f32(row[i]);
}
FP32 dot(const FP32* a, const FP32* b, int n) {
    FP32 sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}
FP32 dot(const BF16* a, const FP32* b, int n) {
    FP32 sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += f32(a[i]) * b[i];
    return sum;
}
void mv(const Linear& w, const FP32* x, FP32* y) {
    for (int row = 0; row < w.rows; ++row)
        y[row] = dot(w.w + static_cast<size_t>(row) * w.cols, x, w.cols);
}
void rms(const FP32* x, const BF16* w, int n, FP32* out) {
    FP32 square = 0.0f;
    for (int i = 0; i < n; ++i) square += x[i] * x[i];
    const FP32 scale = 1.0f / std::sqrt(square / n + EPS);
    for (int i = 0; i < n; ++i) out[i] = x[i] * scale * (1.0f + f32(w[i]));
}
void gated_rms(FP32* x, const FP32* weight, const FP32* gate, int n) {
    FP32 square = 0.0f;
    for (int i = 0; i < n; ++i) square += x[i] * x[i];
    const FP32 scale = 1.0f / std::sqrt(square / n + EPS);
    for (int i = 0; i < n; ++i) x[i] *= scale * weight[i] * silu(gate[i]);
}
void residual_add(FP32* hidden, const FP32* branch, int H) {
    for (int i = 0; i < H; ++i) hidden[i] += branch[i];
}
void l2(FP32* x, int n) {
    FP32 square = 0.0f;
    for (int i = 0; i < n; ++i) square += x[i] * x[i];
    const FP32 scale = 1.0f / std::sqrt(square + EPS);
    for (int i = 0; i < n; ++i) x[i] *= scale;
}
void rope(FP32* x, int position, int RD) {
    FP32 old[64];
    Q35_ASSERT(RD <= 64, "RoPE dimension=%d", RD);
    std::memcpy(old, x, static_cast<size_t>(RD) * sizeof(FP32));
    for (int i = 0; i < RD / 2; ++i) {
        const FP32 angle = position / std::pow(THETA, 2.0f * i / RD);
        const FP32 c = std::cos(angle), s = std::sin(angle);
        x[i] = old[i] * c - old[i + RD / 2] * s;
        x[i + RD / 2] = old[i + RD / 2] * c + old[i] * s;
    }
}

void conv_step(FP32* x, const BF16* w, FP32* history, int DQKV, int CK) {
    for (int channel = 0; channel < DQKV; ++channel) {
        FP32* past = history + static_cast<size_t>(channel) * (CK - 1);
        FP32 sum = x[channel] * f32(w[channel * CK + CK - 1]);
        for (int i = 0; i < CK - 1; ++i) sum += past[i] * f32(w[channel * CK + i]);
        for (int i = 0; i < CK - 2; ++i) past[i] = past[i + 1];
        past[CK - 2] = x[channel];
        x[channel] = silu(sum);
    }
}
void delta_rule(const FP32* q, const FP32* k, const FP32* v, FP32 decay, FP32 beta,
                FP32* state, FP32* out, int KD, int VD) {
    FP32 predicted[128] = {};
    Q35_ASSERT(VD <= 128, "Delta value dimension=%d", VD);
    // S <- decay*S; predicted <- k*S; S <- S+k*(v-predicted); out <- q*S.
    const FP32 scale = std::exp(decay);
    for (int i = 0; i < KD * VD; ++i) state[i] *= scale;
    for (int value = 0; value < VD; ++value)
        for (int key = 0; key < KD; ++key) predicted[value] += k[key] * state[key * VD + value];
    for (int key = 0; key < KD; ++key)
        for (int value = 0; value < VD; ++value)
            state[key * VD + value] += k[key] * beta * (v[value] - predicted[value]);
    for (int value = 0; value < VD; ++value) {
        out[value] = 0.0f;
        for (int key = 0; key < KD; ++key) out[value] += q[key] * state[key * VD + value];
    }
}

void deltanet(const DeltaWeights& w, LayerState& state, const FP32* input,
              Work& work, FP32* out, const q35_model::ModelConfig& c) {
    const int DQK = c.KH * c.KD, DQKV = 2 * DQK + c.VH * c.VD;
    mv(w.qkv, input, work.delta_qkv);
    mv(w.z, input, work.delta_z);
    mv(w.a, input, work.delta_a);
    mv(w.b, input, work.delta_b);
    conv_step(work.delta_qkv, w.conv, state.conv, DQKV, c.CK);
    // qkv 固定拆成 small_q[KH,KD]、small_k[KH,KD]、value[VH,VD]。
    const FP32* small_q = work.delta_qkv;
    const FP32* small_k = small_q + DQK;
    const FP32* value = small_k + DQK;
    for (int head = 0; head < c.VH; ++head) {
        const int qk_head = head / (c.VH / c.KH);
        FP32* q = work.delta_q + head * c.KD;
        FP32* k = work.delta_k + head * c.KD;
        std::memcpy(q, small_q + qk_head * c.KD, c.KD * sizeof(FP32));
        std::memcpy(k, small_k + qk_head * c.KD, c.KD * sizeof(FP32));
        l2(q, c.KD);
        l2(k, c.KD);
        for (int i = 0; i < c.KD; ++i) q[i] /= std::sqrt(static_cast<FP32>(c.KD));
        const FP32 beta = sigmoid(work.delta_b[head]);
        const FP32 decay = -std::exp(w.alog[head]) * softplus(work.delta_a[head] + f32(w.dt[head]));
        FP32* result = work.delta_output + head * c.VD;
        delta_rule(q, k, value + head * c.VD, decay, beta,
                   state.memory + static_cast<size_t>(head) * c.KD * c.VD,
                   result, c.KD, c.VD);
        gated_rms(result, w.norm, work.delta_z + head * c.VD, c.VD);
    }
    mv(w.out, work.delta_output, out);
}

void attention(const AttentionWeights& w, LayerState& state, FP32* score, int position,
               const FP32* input, Work& work, FP32* out, const q35_model::ModelConfig& c) {
    const int AS = c.AH * c.AD, KVW = c.KVH * c.AD;
    mv(w.q, input, work.query_and_gate);
    mv(w.k, input, work.key);
    mv(w.v, input, work.value);
    for (int head = 0; head < c.AH; ++head) {
        FP32* q = work.query + head * c.AD;
        std::memcpy(q, work.query_and_gate + head * 2 * c.AD, c.AD * sizeof(FP32));
        std::memcpy(work.attention_gate + head * c.AD,
                    work.query_and_gate + head * 2 * c.AD + c.AD, c.AD * sizeof(FP32));
        rms(q, w.qnorm, c.AD, q);
        rope(q, position, c.RD);
    }
    for (int head = 0; head < c.KVH; ++head) {
        FP32* key = work.key + head * c.AD;
        rms(key, w.knorm, c.AD, key);
        rope(key, position, c.RD);
    }
    // 当前 K/V 先写入 cache，因此 token 可以读取过去和自己。
    const size_t offset = static_cast<size_t>(position) * KVW;
    std::memcpy(state.key + offset, work.key, KVW * sizeof(FP32));
    std::memcpy(state.value + offset, work.value, KVW * sizeof(FP32));
    const FP32 divisor = std::sqrt(static_cast<FP32>(c.AD));
    for (int head = 0; head < c.AH; ++head) {
        // GQA query head 共享 KV head；score 数组原地完成 stable softmax。
        const int kv_head = head / (c.AH / c.KVH);
        FP32 maximum = -std::numeric_limits<FP32>::infinity();
        for (int token = 0; token <= position; ++token) {
            const FP32* key = state.key + (static_cast<size_t>(token) * c.KVH + kv_head) * c.AD;
            score[token] = dot(work.query + head * c.AD, key, c.AD) / divisor;
            maximum = std::max(maximum, score[token]);
        }
        FP32 total = 0.0f;
        for (int token = 0; token <= position; ++token)
            total += score[token] = std::exp(score[token] - maximum);
        FP32* result = work.attention_output + head * c.AD;
        std::fill(result, result + c.AD, 0.0f);
        for (int token = 0; token <= position; ++token) {
            const FP32* value = state.value + (static_cast<size_t>(token) * c.KVH + kv_head) * c.AD;
            for (int i = 0; i < c.AD; ++i) result[i] += score[token] / total * value[i];
        }
    }
    for (int i = 0; i < AS; ++i) work.attention_output[i] *= sigmoid(work.attention_gate[i]);
    mv(w.out, work.attention_output, out);
}

void ffn(const Layer& layer, const FP32* input, Work& work, FP32* out, int I) {
    mv(layer.gate, input, work.ffn_gate);
    mv(layer.up, input, work.ffn_up);
    for (int i = 0; i < I; ++i) work.ffn_gate[i] = silu(work.ffn_gate[i]) * work.ffn_up[i];
    mv(layer.down, work.ffn_gate, out);
}
void forward(const Model& model, State& state, int token, bool compute_logits) {
    const q35_model::ModelConfig& c = *model.config;
    Work& work = state.work;
    Q35_ASSERT(token >= 0 && token < c.V && state.position < state.capacity,
               "forward token=%d position=%d model=%s", token, state.position, c.name);
    embed(model.embedding, token, c.H, work.hidden);
    // embedding -> N * {norm, mixer, residual, norm, FFN, residual} -> logits。
    for (int i = 0; i < c.N; ++i) {
        const Layer& layer = model.layer[i];
        rms(work.hidden, layer.input_norm, c.H, work.normalized);
        if (i % c.AI != c.AI - 1)
            deltanet(layer.delta, state.layer[i], work.normalized, work, work.branch, c);
        else
            attention(layer.attention, state.layer[i], state.score.get(), state.position,
                      work.normalized, work, work.branch, c);
        residual_add(work.hidden, work.branch, c.H);
        rms(work.hidden, layer.post_norm, c.H, work.normalized);
        ffn(layer, work.normalized, work, work.branch, c.I);
        residual_add(work.hidden, work.branch, c.H);
    }
    if (compute_logits) {
        rms(work.hidden, model.final_norm, c.H, work.normalized);
        mv({model.embedding, c.V, c.H}, work.normalized, work.logits);
    }
    ++state.position;
}

struct Reader {
    const uint8_t *begin, *cursor, *end;
    const char* error = nullptr;
    template <typename T> const T* take(size_t count) {
        if (error) return nullptr;
        const size_t padding = (64 - static_cast<size_t>(cursor - begin) % 64) % 64;
        if (padding > static_cast<size_t>(end - cursor)) { error = "truncated model.bin"; return nullptr; }
        cursor += padding;
        const size_t bytes = count * sizeof(T);
        if (bytes > static_cast<size_t>(end - cursor)) { error = "truncated model.bin"; return nullptr; }
        const T* result = reinterpret_cast<const T*>(cursor); cursor += bytes; return result;
    }
};

// model.bin 是唯一固定 tensor stream；header 选择 config，Reader 按 64-byte alignment 绑定指针。
bool Model::load(const char* path, const char** error) {
    auto fail = [&](const char* message) { *error = message; return false; };
    fd = open(path, O_RDONLY);
    struct stat info {};
    if (fd < 0) return fail("cannot open model.bin");
    if (fstat(fd, &info) || info.st_size < static_cast<off_t>(q35_model::HEADER_SIZE))
        return fail("bad model.bin");
    file_size = static_cast<size_t>(info.st_size);
    file = static_cast<const uint8_t*>(mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (file == MAP_FAILED) { file = nullptr; return fail("mmap model.bin failed"); }
    if (std::memcmp(file, "Q35MODL\0", 8) != 0) return fail("wrong model.bin magic");
    uint32_t version = 0, reserved = 0;
    std::memcpy(&version, file + 8, 4); std::memcpy(&reserved, file + 12, 4);
    if (reserved || version != q35_model::FORMAT_VERSION)
        return fail("unsupported model.bin version");
    const uint32_t id = q35_model::header_field(file, q35_model::MODEL_ID);
    config = q35_model::config_for_id(id);
    if (!config) return fail("unsupported Qwen3.5 model ID");
    if (!q35_model::header_matches(file, file_size, *config))
        return fail("Qwen3.5 model.bin header mismatch");
    Reader reader {file, file + q35_model::HEADER_SIZE, file + file_size};
    const auto& c = *config;
    const int AS = c.AH * c.AD, KVW = c.KVH * c.AD;
    const int DO = c.VH * c.VD, DQKV = 2 * c.KH * c.KD + DO;
    auto linear = [&](int rows, int cols) {
        return Linear {reader.take<BF16>(static_cast<size_t>(rows) * cols), rows, cols};
    };
    layer.reset(new (std::nothrow) Layer[c.N]);
    if (!layer) return fail("cannot allocate model layer table");
    embedding = reader.take<BF16>(static_cast<size_t>(c.V) * c.H);
    final_norm = reader.take<BF16>(c.H);
    for (int i = 0; i < c.N; ++i) {
        Layer& l = layer[i]; l.input_norm = reader.take<BF16>(c.H);
        if (i % c.AI != c.AI - 1) {
            l.delta.qkv = linear(DQKV, c.H); l.delta.z = linear(DO, c.H);
            l.delta.a = linear(c.VH, c.H); l.delta.b = linear(c.VH, c.H);
            l.delta.conv = reader.take<BF16>(static_cast<size_t>(DQKV) * c.CK);
            l.delta.alog = reader.take<FP32>(c.VH); l.delta.dt = reader.take<BF16>(c.VH);
            l.delta.norm = reader.take<FP32>(c.VD); l.delta.out = linear(c.H, DO);
        } else {
            l.attention.q = linear(2 * AS, c.H); l.attention.k = linear(KVW, c.H);
            l.attention.v = linear(KVW, c.H); l.attention.qnorm = reader.take<BF16>(c.AD);
            l.attention.knorm = reader.take<BF16>(c.AD); l.attention.out = linear(c.H, AS);
        }
        l.post_norm = reader.take<BF16>(c.H); l.gate = linear(c.I, c.H);
        l.up = linear(c.I, c.H); l.down = linear(c.H, c.I);
    }
    if (reader.error) return fail(reader.error);
    if (reader.cursor != reader.end) return fail("model.bin size does not match schema");
    return true;
}

Model* model_create(const char* path, char* err, size_t errlen) {
    std::unique_ptr<Model> model(new (std::nothrow) Model());
    const char* message = nullptr;
    if (!model || !model->load(path, &message)) {
        if (err && errlen) std::snprintf(err, errlen, "%s", message ? message : "allocation failed");
        return nullptr;
    }
    if (err && errlen) err[0] = '\0';
    return model.release();
}
void model_destroy(Model* model) { delete model; }
State* state_create(Model* model, int context_size) {
    Q35_ASSERT(model && model->config && context_size > 0 && context_size <= q35_model::MAX_CONTEXT,
               "state_create model=%p context=%d", static_cast<void*>(model), context_size);
    return new State(*model->config, context_size);
}
void state_destroy(State* state) { delete state; }
void state_reset(State* state) { Q35_ASSERT(state, "state_reset null"); state->reset(); }
void state_forward(Model* model, State* state, const int* tokens, int count, bool logits) {
    Q35_ASSERT(model && state && tokens && count > 0, "state_forward count=%d", count);
    for (int i = 0; i < count; ++i) forward(*model, *state, tokens[i], logits && i + 1 == count);
}
void state_checkpoint_save(State* state) {
    Q35_ASSERT(state, "checkpoint save null"); state->checkpoint_position = state->position;
    std::memcpy(state->checkpoint_recurrent.get(), state->recurrent.data.get(),
                state->recurrent.count * sizeof(FP32));
    std::memcpy(state->checkpoint_logits.get(), state->work.logits,
                static_cast<size_t>(state->config->V) * sizeof(FP32));
}
void state_checkpoint_restore(State* state) {
    Q35_ASSERT(state, "checkpoint restore null");
    Q35_ASSERT(state->checkpoint_position >= 0 && state->checkpoint_position <= state->capacity,
               "checkpoint position=%d capacity=%d", state->checkpoint_position, state->capacity);
    std::memcpy(state->recurrent.data.get(), state->checkpoint_recurrent.get(),
                state->recurrent.count * sizeof(FP32));
    std::memcpy(state->work.logits, state->checkpoint_logits.get(),
                static_cast<size_t>(state->config->V) * sizeof(FP32));
    state->position = state->checkpoint_position;
}
int state_argmax(const State* state) {
    if (!state) return -1;
    return static_cast<int>(std::max_element(state->work.logits,
                            state->work.logits + state->config->V) - state->work.logits);
}
void state_copy_logits(const State* state, float* output) {
    Q35_ASSERT(state && output, "state_copy_logits null");
    std::memcpy(output, state->work.logits, static_cast<size_t>(state->config->V) * sizeof(FP32));
}
int vocab_size() { return q35_model::QWEN35_08B.V; }
int max_context() { return q35_model::MAX_CONTEXT; }
bool token_is_stop(int token) { return token == END_OF_TEXT_TOKEN || token == IM_END_TOKEN; }
uint32_t model_id(const Model* model) { return model->config->id; }

}  // namespace q35_backend

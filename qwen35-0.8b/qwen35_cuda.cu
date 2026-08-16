// qwen35_cuda.cu -- qwen35.cpp 的可选 CUDA backend。
//
// CPU 文件仍是唯一的数学 reference。本文件故意不设计 Backend class：它直接复用
// qwen35.cpp 的 Model loader/weight structs，然后把同一批固定权重和 state 放到 GPU。
// 这版以直接 GEMV kernel 保持 BF16 weight / FP32 activation 的 reference 语义；其余
// 小而模型特定的算子也在下面以直接 kernel 写出。

#include <cuda_runtime.h>

// 复用 CPU 的文件格式、权重 structs、parse_ids/argmax 和数学 self-test。把原 main
// 改名即可避免两个入口；CUDA 的真正 forward 在本文件下半部分。
#define main qwen35_cpu_reference_main
#include "qwen35.cpp"
#undef main

namespace qwen35 {

constexpr int MAX_TOKENS = 2048;
constexpr int THREADS = 256;

[[noreturn]] void cuda_die(const char* where, const char* detail) {
    std::fprintf(stderr, "qwen35_cuda: %s: %s\n", where, detail);
    std::exit(1);
}

void cuda_ok(cudaError_t status, const char* where) {
    if (status != cudaSuccess) cuda_die(where, cudaGetErrorString(status));
}

// device 端 BF16->FP32。权重保持 BF16；activation/state 一律 FP32，和 CPU reference
// 一样，因而数值比较的含义非常清楚。
__device__ float dbf(B value) { return __uint_as_float(static_cast<unsigned int>(value) << 16); }
__device__ float dsigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}
__device__ float dsilu(float x) { return x * dsigmoid(x); }
__device__ float dsoftplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

// 所有 reduction kernel 都固定用 256 thread；这段 textbook reduction 比引入 CUB 更
// 容易与 RMSNorm/attention/DeltaNet 的数学逐项对应。
__device__ float block_sum(float value) {
    __shared__ float shared[THREADS];
    const int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride /= 2) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    return shared[0];
}

__device__ float block_max(float value) {
    __shared__ float shared[THREADS];
    const int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride /= 2) {
        if (tid < stride) shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        __syncthreads();
    }
    return shared[0];
}

__global__ void k_embed(const B* table, int token, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < H) out[i] = dbf(table[static_cast<size_t>(token) * H + i]);
}

// 一 block 计算 W 的一行。它与 CPU mv() 的循环完全同构：只把每一行的 col 求和
// 并行化。batch=1 的 correctness backend 优先保留这份直接的 GEMV，而非改变 activation
// 精度去迁就某个 cuBLAS GEMV API 组合。
__global__ void k_gemv(const B* weight, int rows, int cols, const float* input, float* output) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    const B* weights = weight + static_cast<size_t>(row) * cols;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) sum += dbf(weights[col]) * input[col];
    const float total = block_sum(sum);
    if (threadIdx.x == 0) output[row] = total;
}

// Qwen ordinary RMSNorm: x / RMS(x) * (1 + weight)。一个 block 处理一个向量。
__global__ void k_rms_bf16(const float* x, const B* weight, int n, float* out) {
    float square = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) square += x[i] * x[i];
    const float total = block_sum(square);
    const float scale = rsqrtf(total / n + EPS);
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = x[i] * scale * (1.0f + dbf(weight[i]));
}

__global__ void k_add(float* x, const float* branch, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += branch[i];
}

__global__ void k_swiglu(float* gate, const float* up) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < I) gate[i] = dsilu(gate[i]) * up[i];
}

// [channel, CK-1] history 由最旧到最新排列；每个 channel 独立，所以一 thread 足够。
__global__ void k_conv(float* x, const B* weight, float* history) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= DQKV) return;
    float* past = history + static_cast<size_t>(channel) * (CK - 1);
    const float input = x[channel];
    float sum = input * dbf(weight[channel * CK + CK - 1]);
    for (int i = 0; i < CK - 1; ++i) sum += past[i] * dbf(weight[channel * CK + i]);
    for (int i = 0; i < CK - 2; ++i) past[i] = past[i + 1];
    past[CK - 2] = input;
    x[channel] = dsilu(sum);
}

// 一 block 准备一个 value head 所需的 q/k：从 small q/k 复制、L2Norm，并给 q 加
// 1/sqrt(KD)。0.8B 是 1:1；写法仍保留更大模型的 qk-head reuse 映射。
__global__ void k_delta_prepare_qk(const float* small_q, const float* small_k, float* q, float* k) {
    const int head = blockIdx.x;
    const int i = threadIdx.x;
    const int qk_head = head / (VH / KH);
    const float q_value = i < KD ? small_q[qk_head * KD + i] : 0.0f;
    const float k_value = i < KD ? small_k[qk_head * KD + i] : 0.0f;
    const float q_total = block_sum(q_value * q_value);
    __shared__ float q_scale;
    if (i == 0) q_scale = rsqrtf(q_total + EPS) / sqrtf(static_cast<float>(KD));
    __syncthreads();
    if (i < KD) q[head * KD + i] = q_value * q_scale;

    const float k_total = block_sum(k_value * k_value);
    __shared__ float k_scale;
    if (i == 0) k_scale = rsqrtf(k_total + EPS);
    __syncthreads();
    if (i < KD) k[head * KD + i] = k_value * k_scale;
}

__global__ void k_delta_params(const float* a, const float* b, const float* alog, const B* dt,
                                float* beta, float* decay) {
    const int head = threadIdx.x;
    if (head < VH) {
        beta[head] = dsigmoid(b[head]);
        decay[head] = expf(-expf(alog[head]) * dsoftplus(a[head] + dbf(dt[head])));
    }
}

__global__ void k_delta_decay(float* state, const float* decay) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(VH) * KD * VD;
    if (index < count) state[index] *= decay[index / (KD * VD)];
}

// 每个 block 写出一个 memory[head,value] = sum_key k * S。它发生在 decay 后、update 前。
__global__ void k_delta_memory(const float* key, const float* state, float* memory) {
    const int index = blockIdx.x;
    const int head = index / VD;
    const int value = index % VD;
    float sum = 0.0f;
    for (int key_dim = threadIdx.x; key_dim < KD; key_dim += blockDim.x) {
        sum += key[head * KD + key_dim] * state[(static_cast<size_t>(head) * KD + key_dim) * VD + value];
    }
    const float total = block_sum(sum);
    if (threadIdx.x == 0) memory[index] = total;
}

__global__ void k_delta_update(const float* key, const float* value, const float* beta,
                               const float* memory, float* state) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(VH) * KD * VD;
    if (index >= count) return;
    const int head = index / (KD * VD);
    const int rest = index % (KD * VD);
    const int key_dim = rest / VD;
    const int value_dim = rest % VD;
    state[index] += key[head * KD + key_dim] * beta[head] *
                    (value[head * VD + value_dim] - memory[head * VD + value_dim]);
}

__global__ void k_delta_read(const float* query, const float* state, float* out) {
    const int index = blockIdx.x;
    const int head = index / VD;
    const int value = index % VD;
    float sum = 0.0f;
    for (int key_dim = threadIdx.x; key_dim < KD; key_dim += blockDim.x) {
        sum += query[head * KD + key_dim] * state[(static_cast<size_t>(head) * KD + key_dim) * VD + value];
    }
    const float total = block_sum(sum);
    if (threadIdx.x == 0) out[index] = total;
}

// DeltaNet 的 head-local norm/gate。norm.weight 是 F32 直接乘，不是 ordinary RMSNorm 的 1+w。
__global__ void k_gated_rms(float* x, const float* weight, const float* z) {
    const int head = blockIdx.x;
    const int i = threadIdx.x;
    const int offset = head * VD;
    const float value = i < VD ? x[offset + i] : 0.0f;
    const float total = block_sum(value * value);
    const float scale = rsqrtf(total / VD + EPS);
    if (i < VD) x[offset + i] = value * scale * weight[i] * dsilu(z[offset + i]);
}

// q_proj 的输出是 [query, gate]。这里对每个 query head 做 QNorm、复制 gate、并在
// 前 RD 通道做 half-rotation RoPE。
__global__ void k_attention_q_prepare(const float* q_proj, const B* norm, int position,
                                      float* query, float* gate) {
    const int head = blockIdx.x;
    const int i = threadIdx.x;
    const int input = head * 2 * AD;
    const int output = head * AD;
    const float raw = i < AD ? q_proj[input + i] : 0.0f;
    const float total = block_sum(raw * raw);
    const float scale = rsqrtf(total / AD + EPS);
    if (i < AD) {
        query[output + i] = raw * scale * (1.0f + dbf(norm[i]));
        gate[output + i] = q_proj[input + AD + i];
    }
    __syncthreads();
    if (i < RD / 2) {
        const float angle = position / powf(THETA, 2.0f * i / RD);
        const float left = query[output + i];
        const float right = query[output + i + RD / 2];
        query[output + i] = left * cosf(angle) - right * sinf(angle);
        query[output + i + RD / 2] = right * cosf(angle) + left * sinf(angle);
    }
}

__global__ void k_attention_k_prepare(float* key, const B* norm, int position) {
    const int head = blockIdx.x;
    const int i = threadIdx.x;
    const int offset = head * AD;
    const float raw = i < AD ? key[offset + i] : 0.0f;
    const float total = block_sum(raw * raw);
    const float scale = rsqrtf(total / AD + EPS);
    if (i < AD) key[offset + i] = raw * scale * (1.0f + dbf(norm[i]));
    __syncthreads();
    if (i < RD / 2) {
        const float angle = position / powf(THETA, 2.0f * i / RD);
        const float left = key[offset + i];
        const float right = key[offset + i + RD / 2];
        key[offset + i] = left * cosf(angle) - right * sinf(angle);
        key[offset + i + RD / 2] = right * cosf(angle) + left * sinf(angle);
    }
}

__global__ void k_cache_write(const float* key, const float* value, int position,
                              float* keys, float* values) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < KVS) {
        keys[static_cast<size_t>(position) * KVS + i] = key[i];
        values[static_cast<size_t>(position) * KVS + i] = value[i];
    }
}

// 一个 block 计算一个 (query head, token) score；只读到目前已写入的 cache，因此天然 causal。
__global__ void k_attention_scores(const float* query, const float* keys, int tokens, float* scores) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    if (token >= tokens) return;
    const int kv_head = head / (AH / KVH);
    float dot = 0.0f;
    for (int i = threadIdx.x; i < AD; i += blockDim.x) {
        dot += query[head * AD + i] * keys[(static_cast<size_t>(token) * KVH + kv_head) * AD + i];
    }
    const float total = block_sum(dot);
    if (threadIdx.x == 0) scores[head * MAX_TOKENS + token] = total / sqrtf(static_cast<float>(AD));
}

__global__ void k_attention_softmax(float* scores, int tokens) {
    const int head = blockIdx.x;
    float maximum = -1.0e30f;
    for (int token = threadIdx.x; token < tokens; token += blockDim.x) {
        maximum = fmaxf(maximum, scores[head * MAX_TOKENS + token]);
    }
    maximum = block_max(maximum);
    float sum = 0.0f;
    for (int token = threadIdx.x; token < tokens; token += blockDim.x) {
        const int index = head * MAX_TOKENS + token;
        const float probability = expf(scores[index] - maximum);
        scores[index] = probability;
        sum += probability;
    }
    sum = block_sum(sum);
    for (int token = threadIdx.x; token < tokens; token += blockDim.x) {
        scores[head * MAX_TOKENS + token] /= sum;
    }
}

__global__ void k_attention_values(const float* scores, const float* values, int tokens,
                                   const float* gate, float* out) {
    const int head = blockIdx.x;
    const int i = threadIdx.x;
    if (i >= AD) return;
    const int kv_head = head / (AH / KVH);
    float sum = 0.0f;
    for (int token = 0; token < tokens; ++token) {
        sum += scores[head * MAX_TOKENS + token] *
               values[(static_cast<size_t>(token) * KVH + kv_head) * AD + i];
    }
    out[head * AD + i] = sum * dsigmoid(gate[head * AD + i]);
}

// C++ host side ---------------------------------------------------------------

struct CudaArena {
    std::vector<void*> pointers;
    ~CudaArena() {
        for (void* pointer : pointers) cudaFree(pointer);
    }
    float* floats(size_t count, bool zero = false) {
        float* pointer = nullptr;
        cuda_ok(cudaMalloc(&pointer, count * sizeof(float)), "cudaMalloc float");
        pointers.push_back(pointer);
        if (zero) cuda_ok(cudaMemset(pointer, 0, count * sizeof(float)), "cudaMemset float");
        return pointer;
    }
    B* bf16s(size_t count) {
        B* pointer = nullptr;
        cuda_ok(cudaMalloc(&pointer, count * sizeof(B)), "cudaMalloc BF16");
        pointers.push_back(pointer);
        return pointer;
    }
};

// DeviceModel 保留 host Model 仅用来持有 mmap 文件和进行一次 upload；forward 从不解引用 host 权重。
struct DeviceModel {
    Model host;
    CudaArena arena;
    const B* embedding = nullptr;
    const B* final_norm = nullptr;
    std::array<Layer, N> layer {};

    explicit DeviceModel(const char* path) : host(path) {
        embedding = upload_bf16(host.embedding, static_cast<size_t>(V) * H);
        final_norm = upload_bf16(host.final_norm, H);
        for (int index = 0; index < N; ++index) {
            const Layer& source = host.layer[index];
            Layer& target = layer[index];
            target.delta = source.delta;
            target.input_norm = upload_bf16(source.input_norm, H);
            if (target.delta) {
                target.d.qkv = upload_linear(source.d.qkv);
                target.d.z = upload_linear(source.d.z);
                target.d.a = upload_linear(source.d.a);
                target.d.b = upload_linear(source.d.b);
                target.d.conv = upload_bf16(source.d.conv, static_cast<size_t>(DQKV) * CK);
                target.d.alog = upload_f32(source.d.alog, VH);
                target.d.dt = upload_bf16(source.d.dt, VH);
                target.d.norm = upload_f32(source.d.norm, VD);
                target.d.out = upload_linear(source.d.out);
            } else {
                target.a.q = upload_linear(source.a.q);
                target.a.k = upload_linear(source.a.k);
                target.a.v = upload_linear(source.a.v);
                target.a.qnorm = upload_bf16(source.a.qnorm, AD);
                target.a.knorm = upload_bf16(source.a.knorm, AD);
                target.a.out = upload_linear(source.a.out);
            }
            target.post_norm = upload_bf16(source.post_norm, H);
            target.gate = upload_linear(source.gate);
            target.up = upload_linear(source.up);
            target.down = upload_linear(source.down);
        }
    }

    const B* upload_bf16(const B* source, size_t count) {
        B* target = arena.bf16s(count);
        cuda_ok(cudaMemcpy(target, source, count * sizeof(B), cudaMemcpyHostToDevice), "upload BF16");
        return target;
    }
    const float* upload_f32(const float* source, size_t count) {
        float* target = arena.floats(count);
        cuda_ok(cudaMemcpy(target, source, count * sizeof(float), cudaMemcpyHostToDevice), "upload F32");
        return target;
    }
    Linear upload_linear(const Linear& source) {
        return {upload_bf16(source.w, static_cast<size_t>(source.rows) * source.cols), source.rows, source.cols};
    }
};

struct CudaState {
    int position = 0;
    CudaArena arena;
    std::array<float*, N> conv {}, recurrent {}, keys {}, values {};
    CudaState() {
        for (int layer = 0; layer < N; ++layer) {
            if (layer % 4 != 3) {
                conv[layer] = arena.floats(static_cast<size_t>(DQKV) * (CK - 1), true);
                recurrent[layer] = arena.floats(static_cast<size_t>(VH) * KD * VD, true);
            } else {
                keys[layer] = arena.floats(static_cast<size_t>(MAX_TOKENS) * KVS, true);
                values[layer] = arena.floats(static_cast<size_t>(MAX_TOKENS) * KVS, true);
            }
        }
    }
};

struct CudaWork {
    CudaArena arena;
    float *h, *n, *mix, *gate, *up, *logits;
    float *qkv, *z, *da, *db, *dq, *dk, *dout, *beta, *decay, *memory;
    float *aqp, *aq, *ag, *ak, *av, *ao, *scores;
    std::vector<float> host_logits = std::vector<float>(V);
    CudaWork() {
        h = arena.floats(H); n = arena.floats(H); mix = arena.floats(H);
        gate = arena.floats(I); up = arena.floats(I); logits = arena.floats(V);
        qkv = arena.floats(DQKV); z = arena.floats(DO); da = arena.floats(VH); db = arena.floats(VH);
        dq = arena.floats(DO); dk = arena.floats(DO); dout = arena.floats(DO);
        beta = arena.floats(VH); decay = arena.floats(VH); memory = arena.floats(DO);
        aqp = arena.floats(2 * AS); aq = arena.floats(AS); ag = arena.floats(AS);
        ak = arena.floats(KVS); av = arena.floats(KVS); ao = arena.floats(AS);
        scores = arena.floats(static_cast<size_t>(AH) * MAX_TOKENS);
    }
};

void gemv(const Linear& weight, const float* input, float* output) {
    k_gemv<<<weight.rows, THREADS>>>(weight.w, weight.rows, weight.cols, input, output);
}

void deltanet_cuda(const Delta& d, float* conv, float* recurrent, const float* input,
                   CudaWork& w, float* out) {
    gemv(d.qkv, input, w.qkv);
    gemv(d.z, input, w.z);
    gemv(d.a, input, w.da);
    gemv(d.b, input, w.db);
    k_conv<<<(DQKV + THREADS - 1) / THREADS, THREADS>>>(w.qkv, d.conv, conv);
    k_delta_prepare_qk<<<VH, THREADS>>>(w.qkv, w.qkv + DQK, w.dq, w.dk);
    k_delta_params<<<1, THREADS>>>(w.da, w.db, d.alog, d.dt, w.beta, w.decay);
    const size_t state_count = static_cast<size_t>(VH) * KD * VD;
    k_delta_decay<<<(state_count + THREADS - 1) / THREADS, THREADS>>>(recurrent, w.decay);
    k_delta_memory<<<VH * VD, THREADS>>>(w.dk, recurrent, w.memory);
    k_delta_update<<<(state_count + THREADS - 1) / THREADS, THREADS>>>(w.dk, w.qkv + 2 * DQK,
                                                                        w.beta, w.memory, recurrent);
    k_delta_read<<<VH * VD, THREADS>>>(w.dq, recurrent, w.dout);
    k_gated_rms<<<VH, THREADS>>>(w.dout, d.norm, w.z);
    gemv(d.out, w.dout, out);
}

void attention_cuda(const Attention& a, float* keys, float* values, int position,
                    const float* input, CudaWork& w, float* out) {
    gemv(a.q, input, w.aqp);
    gemv(a.k, input, w.ak);
    gemv(a.v, input, w.av);
    k_attention_q_prepare<<<AH, THREADS>>>(w.aqp, a.qnorm, position, w.aq, w.ag);
    k_attention_k_prepare<<<KVH, THREADS>>>(w.ak, a.knorm, position);
    k_cache_write<<<(KVS + THREADS - 1) / THREADS, THREADS>>>(w.ak, w.av, position, keys, values);
    const int tokens = position + 1;
    k_attention_scores<<<dim3(AH, tokens), THREADS>>>(w.aq, keys, tokens, w.scores);
    k_attention_softmax<<<AH, THREADS>>>(w.scores, tokens);
    k_attention_values<<<AH, THREADS>>>(w.scores, values, tokens, w.ag, w.ao);
    gemv(a.out, w.ao, out);
}

void mlp_cuda(const Layer& layer, const float* input, CudaWork& w, float* out) {
    gemv(layer.gate, input, w.gate);
    gemv(layer.up, input, w.up);
    k_swiglu<<<(I + THREADS - 1) / THREADS, THREADS>>>(w.gate, w.up);
    gemv(layer.down, w.gate, out);
}

void forward_cuda(const DeviceModel& model, CudaState& state, int token, CudaWork& work) {
    if (state.position >= MAX_TOKENS) cuda_die("forward", "maximum context is 2048 tokens");
    k_embed<<<(H + THREADS - 1) / THREADS, THREADS>>>(model.embedding, token, work.h);
    for (int index = 0; index < N; ++index) {
        const Layer& layer = model.layer[index];
        k_rms_bf16<<<1, THREADS>>>(work.h, layer.input_norm, H, work.n);
        if (layer.delta) deltanet_cuda(layer.d, state.conv[index], state.recurrent[index], work.n, work, work.mix);
        else attention_cuda(layer.a, state.keys[index], state.values[index], state.position, work.n, work, work.mix);
        k_add<<<(H + THREADS - 1) / THREADS, THREADS>>>(work.h, work.mix, H);
        k_rms_bf16<<<1, THREADS>>>(work.h, layer.post_norm, H, work.n);
        mlp_cuda(layer, work.n, work, work.mix);
        k_add<<<(H + THREADS - 1) / THREADS, THREADS>>>(work.h, work.mix, H);
    }
    k_rms_bf16<<<1, THREADS>>>(work.h, model.final_norm, H, work.n);
    gemv({model.embedding, V, H}, work.n, work.logits);  // tied lm_head。
    cuda_ok(cudaGetLastError(), "CUDA forward kernel launch");
    cuda_ok(cudaMemcpy(work.host_logits.data(), work.logits, static_cast<size_t>(V) * sizeof(float),
                       cudaMemcpyDeviceToHost), "download logits");
    ++state.position;
}

void generate_cuda(const char* path, const std::vector<int>& prompt, int count, std::vector<int>* result) {
    DeviceModel model(path);  // 上传全部 BF16/F32 权重一次。
    CudaState state;          // generation 的 KV cache / DeltaNet state 也只分配一次。
    CudaWork work;
    for (int token : prompt) forward_cuda(model, state, token, work);
    for (int step = 0; step < count; ++step) {
        const int next = argmax(work.host_logits);
        if (next == 248044 || next == 248046) break;
        result->push_back(next);
        if (step + 1 < count) forward_cuda(model, state, next, work);
    }
}

void cuda_self_test() {
    self_test();
    int devices = 0;
    cuda_ok(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (devices == 0) cuda_die("cuda_self_test", "no CUDA device");
    std::printf("cuda-self-test: found %d device(s)\n", devices);
}

void cuda_usage(const char* program) {
    std::printf("usage: %s --self-test\n", program);
    std::printf("       %s --forward <qwen35-0.8b.bin> <id,id,...>\n", program);
    std::printf("       %s --generate <qwen35-0.8b.bin> <id,id,...> <new-tokens>\n", program);
}

}  // namespace qwen35

int main(int argc, char** argv) {
    using namespace qwen35;
    if (argc == 1 || std::strcmp(argv[1], "--self-test") == 0) { cuda_self_test(); return 0; }
    if (std::strcmp(argv[1], "--forward") == 0 && argc == 4) {
        DeviceModel model(argv[2]); CudaState state; CudaWork work;
        for (int token : parse_ids(argv[3])) forward_cuda(model, state, token, work);
        const int next = argmax(work.host_logits);
        std::printf("next token: %d, logit: %.6f\n", next, work.host_logits[next]);
        return 0;
    }
    if (std::strcmp(argv[1], "--generate") == 0 && argc == 5) {
        std::vector<int> output;
        generate_cuda(argv[2], parse_ids(argv[3]), std::atoi(argv[4]), &output);
        std::printf("generated:");
        for (int token : output) std::printf(" %d", token);
        std::putchar('\n');
        return 0;
    }
    cuda_usage(argv[0]);
    return 1;
}

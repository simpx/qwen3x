// qwen35_cuda.cu -- Qwen3.5-0.8B 的 CUDA performance backend。
//
// CPU 数学 reference 已是课程的 ../lessons/09_qwen35_0_8b.cpp。本文件直接复用它的
// 固定 loader/weight structs，却把权重和 state 留在 GPU。没有 Backend class：性能路径
// 也只服务这一种模型、这一种 batch=1 token forward。

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

// 复用 CPU 的文件格式、权重 structs、parse_ids/argmax 和数学 self-test。把原 main
// 改名即可避免两个入口；CUDA 的真正 forward 在本文件下半部分。
#define main qwen35_cpu_reference_main
#include "../lessons/09_qwen35_0_8b.cpp"
#undef main

namespace qwen35 {

constexpr int THREADS = 256;

[[noreturn]] void cuda_die(const char* where, const char* detail) {
    std::fprintf(stderr, "qwen35_cuda: %s: %s\n", where, detail);
    std::exit(1);
}

void cuda_ok(cudaError_t status, const char* where) {
    if (status != cudaSuccess) cuda_die(where, cudaGetErrorString(status));
}

void cublas_ok(cublasStatus_t status, const char* where) {
    if (status != CUBLAS_STATUS_SUCCESS) cuda_die(where, "cuBLAS call failed");
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

// cuBLAS 的 Tensor Core BF16 GEMV 要求两侧矩阵都是 BF16。每个 linear 前把短 activation
// 向量舍入一次；权重本来就是 BF16，累加和 branch/state 仍保留 FP32。
__global__ void k_f32_to_bf16(const float* input, B* output, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t bits = __float_as_uint(input[i]);
    const uint32_t bias = 0x7fffu + ((bits >> 16) & 1u);  // round-to-nearest-even
    output[i] = static_cast<B>((bits + bias) >> 16);
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

// 生成不需要把 V=248320 个 logits 都传回 CPU：最后只要一个 token id。这个单 block
// reduction 每个 thread 扫一部分 vocabulary，避免每个 decode token 都有 1 MiB D2H copy。
__global__ void k_argmax(const float* logits, int* index_out, float* value_out) {
    __shared__ float values[THREADS];
    __shared__ int indices[THREADS];
    float best = -1.0e30f;
    int best_index = 0;
    for (int index = threadIdx.x; index < V; index += blockDim.x) {
        const float value = logits[index];
        if (value > best || (value == best && index < best_index)) {
            best = value;
            best_index = index;
        }
    }
    values[threadIdx.x] = best;
    indices[threadIdx.x] = best_index;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            const float other_value = values[threadIdx.x + stride];
            const int other_index = indices[threadIdx.x + stride];
            if (other_value > values[threadIdx.x] ||
                (other_value == values[threadIdx.x] && other_index < indices[threadIdx.x])) {
                values[threadIdx.x] = other_value;
                indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *index_out = indices[0];
        *value_out = values[0];
    }
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
    int* ints(size_t count) {
        int* pointer = nullptr;
        cuda_ok(cudaMalloc(&pointer, count * sizeof(int)), "cudaMalloc int");
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
    void reset() {
        // 下一请求的 DeltaNet 不能带上前一请求的 recurrent/conv state。attention cache 则
        // 不必清零：position 从 0 重新开始，forward 会在读取之前覆写所有可见 token。
        position = 0;
        for (int layer = 0; layer < N; ++layer) {
            if (layer % 4 != 3) {
                cuda_ok(cudaMemset(conv[layer], 0, static_cast<size_t>(DQKV) * (CK - 1) * sizeof(float)),
                        "reset DeltaNet convolution");
                cuda_ok(cudaMemset(recurrent[layer], 0, static_cast<size_t>(VH) * KD * VD * sizeof(float)),
                        "reset DeltaNet recurrent state");
            }
        }
    }
    void copy_from(const CudaState& source) {
        // MMLU-Pro 的同一 category 共享完整 five-shot prefix。这里从一个已经 prefill 的
        // base state 克隆到工作 state：只复制 prefix 已写入的 KV rows，DeltaNet state 则固定
        // 大小。所有 copy 都是 GPU->GPU，因此不把 cache 拉回 host。
        position = source.position;
        const size_t kv_bytes = static_cast<size_t>(position) * KVS * sizeof(float);
        for (int layer = 0; layer < N; ++layer) {
            if (layer % 4 != 3) {
                // 同一 default stream 上的 async D2D copy 会排在随后 suffix prefill 的所有
                // kernels 之前；相比同步 cudaMemcpy，不必为每层的两小块 recurrent state
                // 往返 WSL driver 一次。正确性仍由 stream 顺序保证。
                cuda_ok(cudaMemcpyAsync(conv[layer], source.conv[layer],
                                        static_cast<size_t>(DQKV) * (CK - 1) * sizeof(float),
                                        cudaMemcpyDeviceToDevice), "clone DeltaNet convolution");
                cuda_ok(cudaMemcpyAsync(recurrent[layer], source.recurrent[layer],
                                        static_cast<size_t>(VH) * KD * VD * sizeof(float),
                                        cudaMemcpyDeviceToDevice), "clone DeltaNet recurrent state");
            } else if (kv_bytes) {
                cuda_ok(cudaMemcpyAsync(keys[layer], source.keys[layer], kv_bytes, cudaMemcpyDeviceToDevice),
                        "clone attention keys");
                cuda_ok(cudaMemcpyAsync(values[layer], source.values[layer], kv_bytes, cudaMemcpyDeviceToDevice),
                        "clone attention values");
            }
        }
    }
};

struct CudaWork {
    CudaArena arena;
    // cuBLAS 是唯一引入的性能库：每个 linear 仍明确写在 forward 中，只把最无聊、
    // 最重的 W*x 换成成熟且调优过的 GEMV 实现。
    cublasHandle_t blas = nullptr;
    float *h, *n, *mix, *gate, *up, *logits, *best_logit;
    B* gemv_input;
    int* best_token;
    float *qkv, *z, *da, *db, *dq, *dk, *dout, *beta, *decay, *memory;
    float *aqp, *aq, *ag, *ak, *av, *ao, *scores;
    std::vector<float> host_logits = std::vector<float>(V);
    CudaWork() {
        cublas_ok(cublasCreate(&blas), "cublasCreate");
        cublas_ok(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
        h = arena.floats(H); n = arena.floats(H); mix = arena.floats(H);
        gate = arena.floats(I); up = arena.floats(I); logits = arena.floats(V);
        gemv_input = arena.bf16s(I);  // 所有 W*x 的 input 宽度都不超过 SwiGLU 的 I=3584。
        best_logit = arena.floats(1); best_token = arena.ints(1);
        qkv = arena.floats(DQKV); z = arena.floats(DO); da = arena.floats(VH); db = arena.floats(VH);
        dq = arena.floats(DO); dk = arena.floats(DO); dout = arena.floats(DO);
        beta = arena.floats(VH); decay = arena.floats(VH); memory = arena.floats(DO);
        aqp = arena.floats(2 * AS); aq = arena.floats(AS); ag = arena.floats(AS);
        ak = arena.floats(KVS); av = arena.floats(KVS); ao = arena.floats(AS);
        scores = arena.floats(static_cast<size_t>(AH) * MAX_TOKENS);
    }
    ~CudaWork() { if (blas) cublasDestroy(blas); }
    CudaWork(const CudaWork&) = delete;
};

void gemv(CudaWork& work, const Linear& weight, const float* input, float* output) {
    // 权重实际是 row-major W[rows,cols]。cuBLAS 只看 column-major，故把同一段内存解释为
    // W^T[cols,rows]，再请求 transpose：output[rows,1] = W * input[cols,1]。
    // 为让 Tensor Cores 可用，短 activation 向量也临时转 BF16；output 和所有 recurrent
    // state 仍是 FP32。第 09 课保留无此舍入的 CPU correctness oracle。
    const float one = 1.0f, zero = 0.0f;
    k_f32_to_bf16<<<(weight.cols + THREADS - 1) / THREADS, THREADS>>>(input, work.gemv_input,
                                                                        weight.cols);
    cublas_ok(cublasGemmEx(work.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                            weight.rows, 1, weight.cols, &one,
                            weight.w, CUDA_R_16BF, weight.cols,
                            work.gemv_input, CUDA_R_16BF, weight.cols, &zero,
                            output, CUDA_R_32F, weight.rows,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
              "cublasGemmEx W*x");
}

void deltanet_cuda(const Delta& d, float* conv, float* recurrent, const float* input,
                   CudaWork& w, float* out) {
    gemv(w, d.qkv, input, w.qkv);
    gemv(w, d.z, input, w.z);
    gemv(w, d.a, input, w.da);
    gemv(w, d.b, input, w.db);
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
    gemv(w, d.out, w.dout, out);
}

void attention_cuda(const Attention& a, float* keys, float* values, int position,
                    const float* input, CudaWork& w, float* out) {
    gemv(w, a.q, input, w.aqp);
    gemv(w, a.k, input, w.ak);
    gemv(w, a.v, input, w.av);
    k_attention_q_prepare<<<AH, THREADS>>>(w.aqp, a.qnorm, position, w.aq, w.ag);
    k_attention_k_prepare<<<KVH, THREADS>>>(w.ak, a.knorm, position);
    k_cache_write<<<(KVS + THREADS - 1) / THREADS, THREADS>>>(w.ak, w.av, position, keys, values);
    const int tokens = position + 1;
    k_attention_scores<<<dim3(AH, tokens), THREADS>>>(w.aq, keys, tokens, w.scores);
    k_attention_softmax<<<AH, THREADS>>>(w.scores, tokens);
    k_attention_values<<<AH, THREADS>>>(w.scores, values, tokens, w.ag, w.ao);
    gemv(w, a.out, w.ao, out);
}

void mlp_cuda(const Layer& layer, const float* input, CudaWork& w, float* out) {
    gemv(w, layer.gate, input, w.gate);
    gemv(w, layer.up, input, w.up);
    k_swiglu<<<(I + THREADS - 1) / THREADS, THREADS>>>(w.gate, w.up);
    gemv(w, layer.down, w.gate, out);
}

struct DeviceTop { int token; float logit; };

DeviceTop argmax_cuda(CudaWork& work) {
    k_argmax<<<1, THREADS>>>(work.logits, work.best_token, work.best_logit);
    cuda_ok(cudaGetLastError(), "CUDA argmax kernel launch");
    DeviceTop result {};
    cuda_ok(cudaMemcpy(&result.token, work.best_token, sizeof(result.token), cudaMemcpyDeviceToHost),
            "download argmax token");
    cuda_ok(cudaMemcpy(&result.logit, work.best_logit, sizeof(result.logit), cudaMemcpyDeviceToHost),
            "download argmax logit");
    return result;
}

void forward_cuda(const DeviceModel& model, CudaState& state, int token, CudaWork& work,
                  bool download_all_logits = false) {
    if (state.position >= MAX_TOKENS) cuda_die("forward", "maximum context is 4096 tokens");
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
    gemv(work, {model.embedding, V, H}, work.n, work.logits);  // tied lm_head。
    cuda_ok(cudaGetLastError(), "CUDA forward kernel launch");
    if (download_all_logits) {
        cuda_ok(cudaMemcpy(work.host_logits.data(), work.logits, static_cast<size_t>(V) * sizeof(float),
                           cudaMemcpyDeviceToHost), "download logits");
    }
    ++state.position;
}

bool ends_with(const std::vector<int>& tokens, const std::vector<int>& suffix) {
    return tokens.size() >= suffix.size() &&
           std::equal(suffix.rbegin(), suffix.rend(), tokens.rbegin());
}

void prefill_cuda(const DeviceModel& model, CudaState& state, CudaWork& work,
                  const std::vector<int>& prompt) {
    for (int token : prompt) forward_cuda(model, state, token, work);
}

void decode_cuda(const DeviceModel& model, CudaState& state, CudaWork& work, int count,
                 const std::vector<std::vector<int>>& stop_sequences, std::vector<int>* result) {
    for (int step = 0; step < count; ++step) {
        const DeviceTop next = argmax_cuda(work);
        if (next.token == 248044 || next.token == 248046) break;
        result->push_back(next.token);
        for (const std::vector<int>& stop : stop_sequences) {
            if (ends_with(*result, stop)) {
                // vLLM 的 stop string 默认不计入 completion；移走对应 ids 后立即返回。
                result->resize(result->size() - stop.size());
                return;
            }
        }
        if (step + 1 < count) forward_cuda(model, state, next.token, work);
    }
}

void generate_cuda(const DeviceModel& model, CudaState& state, CudaWork& work,
                   const std::vector<int>& prompt, int count, std::vector<int>* result) {
    state.reset();
    prefill_cuda(model, state, work, prompt);
    decode_cuda(model, state, work, count, {}, result);
}

void generate_cuda(const char* path, const std::vector<int>& prompt, int count, std::vector<int>* result) {
    DeviceModel model(path);  // 普通 CLI 一次调用仍然简单、无隐式后台服务。
    CudaState state;
    CudaWork work;
    generate_cuda(model, state, work, prompt, count, result);
}

void serve_cuda(const char* path) {
    // 评测时唯一值得持久化的东西是模型、GPU work buffers 和 cache allocation。旧协议为
    // `new_tokens<TAB>id,id,...`；MMLU-Pro 还使用两条 model-specific 命令：
    // `cache<TAB>prefix_ids`，以及 `generate<TAB>new_tokens<TAB>stop;ids<TAB>suffix_ids`。
    // 后者从 cache clone state 后再 prefill suffix，因此同一学科的 five-shot prefix 只算一次。
    DeviceModel model(path);
    CudaState base;
    CudaState state;
    CudaWork work;
    bool base_ready = false;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "quit") return;
        if (line.rfind("cache\t", 0) == 0) {
            base.reset();
            prefill_cuda(model, base, work, parse_ids(line.c_str() + 6));
            base_ready = true;
            std::printf("ready\n");
            std::fflush(stdout);
            continue;
        }
        if (line.rfind("generate\t", 0) == 0) {
            const size_t first = line.find('\t');
            const size_t second = line.find('\t', first + 1);
            const size_t third = second == std::string::npos ? std::string::npos : line.find('\t', second + 1);
            if (!base_ready || second == std::string::npos || third == std::string::npos) {
                std::printf("error: cache first, then generate<TAB>new_tokens<TAB>stop;ids<TAB>suffix_ids\n");
                std::fflush(stdout);
                continue;
            }
            const int count = std::atoi(line.substr(first + 1, second - first - 1).c_str());
            if (count <= 0) {
                std::printf("error: new_tokens must be positive\n");
                std::fflush(stdout);
                continue;
            }
            std::vector<std::vector<int>> stops;
            const std::string stop_text = line.substr(second + 1, third - second - 1);
            if (stop_text != "-") {
                size_t begin = 0;
                while (begin < stop_text.size()) {
                    const size_t end = stop_text.find(';', begin);
                    const std::string one = stop_text.substr(begin, end - begin);
                    if (!one.empty()) stops.push_back(parse_ids(one.c_str()));
                    if (end == std::string::npos) break;
                    begin = end + 1;
                }
            }
            state.copy_from(base);
            prefill_cuda(model, state, work, parse_ids(line.c_str() + third + 1));
            std::vector<int> output;
            decode_cuda(model, state, work, count, stops, &output);
            std::printf("generated:");
            for (int token : output) std::printf(" %d", token);
            std::putchar('\n');
            std::fflush(stdout);
            continue;
        }
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) {
            std::printf("error: expected new_tokens<TAB>id,id,...\n");
            std::fflush(stdout);
            continue;
        }
        const int count = std::atoi(line.substr(0, tab).c_str());
        if (count <= 0) {
            std::printf("error: new_tokens must be positive\n");
            std::fflush(stdout);
            continue;
        }
        std::vector<int> output;
        generate_cuda(model, state, work, parse_ids(line.c_str() + tab + 1), count, &output);
        std::printf("generated:");
        for (int token : output) std::printf(" %d", token);
        std::putchar('\n');
        std::fflush(stdout);
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
    std::printf("       %s --logits <qwen35-0.8b.bin> <id,id,...>\n", program);
    std::printf("       %s --generate <qwen35-0.8b.bin> <id,id,...> <new-tokens>\n", program);
    std::printf("       %s --serve <qwen35-0.8b.bin>  # stdin: new-tokens<TAB>id,id,...\n", program);
}

}  // namespace qwen35

int main(int argc, char** argv) {
    using namespace qwen35;
    if (argc == 1 || std::strcmp(argv[1], "--self-test") == 0) { cuda_self_test(); return 0; }
    if (std::strcmp(argv[1], "--forward") == 0 && argc == 4) {
        DeviceModel model(argv[2]); CudaState state; CudaWork work;
        for (int token : parse_ids(argv[3])) forward_cuda(model, state, token, work);
        const DeviceTop next = argmax_cuda(work);
        std::printf("next token: %d, logit: %.6f\n", next.token, next.logit);
        return 0;
    }
    if (std::strcmp(argv[1], "--logits") == 0 && argc == 4) {
        // 复用 CPU 文件的 dump format；区别只是 logits 已由 forward_cuda 下载到 host。
        DeviceModel model(argv[2]); CudaState state; CudaWork work;
        for (int token : parse_ids(argv[3])) forward_cuda(model, state, token, work, true);
        dump_logits(work.host_logits);
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
    if (std::strcmp(argv[1], "--serve") == 0 && argc == 3) { serve_cuda(argv[2]); return 0; }
    cuda_usage(argv[0]);
    return 1;
}

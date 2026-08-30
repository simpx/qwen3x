// arch/cuda/engine.cu -- readable CUDA engine for Qwen3.5-0.8B.
//
// This is the same model data flow as engine.cpp with a different physical
// home for Model, State and Work:
//
//   Model: packed BF16 weights are copied once and kept as BF16 on the GPU
//   State: recurrent/KV state and its checkpoint stay on the GPU
//   Work:  one Session-owned GPU scratch area reused by every forward
//
// Prefill runs the same layers over explicit token chunks; decode captures the
// one-token forward as a CUDA Graph. Both stay behind q35_backend::state_forward()
// and do not enter runtime.cpp.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "internal.h"

namespace qwen35_cuda {

constexpr int V = 248320, H = 1024, I = 3584, N = 24;
constexpr int AH = 8, KVH = 2, AD = 256, RD = 64;
constexpr int KH = 16, VH = 16, KD = 128, VD = 128, CK = 4;
constexpr int AS = AH * AD, KV_WIDTH = KVH * AD;
constexpr int DQK = KH * KD, DO = VH * VD, DQKV = 2 * DQK + DO;
constexpr int MAX_CONTEXT = 262144;
constexpr uint32_t MODEL_FORMAT_VERSION = 1;
constexpr int END_OF_TEXT_TOKEN = 248044, IM_END_TOKEN = 248046;
constexpr float EPS = 1e-6f, THETA = 10000000.0f;
constexpr int BLOCK = 256;
constexpr int PREFILL_CHUNK = 128;
using BF16 = uint16_t;

void cuda_fatal(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    q35_internal::report_assertion(
        "CUDA operation succeeded", __FILE__, __LINE__, "%s: %s",
        operation, cudaGetErrorString(status));
    std::abort();
}

#define CUDA_OK(call) qwen35_cuda::cuda_fatal((call), #call)

struct Linear {
    const BF16* weight = nullptr;
    int rows = 0;
    int columns = 0;
};

struct DeltaWeights {
    Linear qkv, z, a, b, out;
    const BF16* conv = nullptr;
    const float* alog = nullptr;
    const BF16* dt = nullptr;
    const float* norm = nullptr;
};

struct AttentionWeights {
    Linear q, k, v, out;
    const BF16* qnorm = nullptr;
    const BF16* knorm = nullptr;
};

struct Layer {
    bool uses_deltanet = false;
    const BF16* input_norm = nullptr;
    const BF16* post_norm = nullptr;
    Linear gate, up, down;
    DeltaWeights delta;
    AttentionWeights attention;
};

struct ModelData {
    const BF16* embedding = nullptr;
    const BF16* final_norm = nullptr;
    std::array<Layer, N> layers {};
    std::vector<void*> allocations;

    ~ModelData() {
        for (void* pointer : allocations) cudaFree(pointer);
    }
};

struct Work {
    float* storage = nullptr;
    float* hidden = nullptr;
    float* normalized = nullptr;
    float* logits = nullptr;
    float* ffn_gate = nullptr;
    float* ffn_up = nullptr;
    float* delta_qkv = nullptr;
    float* delta_z = nullptr;
    float* delta_a = nullptr;
    float* delta_b = nullptr;
    float* delta_q = nullptr;
    float* delta_k = nullptr;
    float* delta_output = nullptr;
    float* query_and_gate = nullptr;
    float* query = nullptr;
    float* attention_gate = nullptr;
    float* key = nullptr;
    float* value = nullptr;
    float* attention_output = nullptr;

    void allocate() {
        const size_t count = 2 * H + V + 2 * I + DQKV + DO + 2 * VH +
                             2 * DO + DO + 2 * AS + AS + AS +
                             2 * KV_WIDTH + AS;
        CUDA_OK(cudaMalloc(&storage, count * sizeof(float)));
        float* cursor = storage;
#define TAKE(name, size) name = cursor; cursor += (size)
        TAKE(hidden, H);
        TAKE(normalized, H);
        TAKE(logits, V);
        TAKE(ffn_gate, I);
        TAKE(ffn_up, I);
        TAKE(delta_qkv, DQKV);
        TAKE(delta_z, DO);
        TAKE(delta_a, VH);
        TAKE(delta_b, VH);
        TAKE(delta_q, DO);
        TAKE(delta_k, DO);
        TAKE(delta_output, DO);
        TAKE(query_and_gate, 2 * AS);
        TAKE(query, AS);
        TAKE(attention_gate, AS);
        TAKE(key, KV_WIDTH);
        TAKE(value, KV_WIDTH);
        TAKE(attention_output, AS);
#undef TAKE
        Q35_ASSERT(cursor == storage + count,
                   "CUDA Work layout cursor=%p end=%p",
                   static_cast<void*>(cursor),
                   static_cast<void*>(storage + count));
    }

    void release() {
        if (storage) cudaFree(storage);
        storage = nullptr;
    }
};

struct BatchWork {
    float* storage = nullptr;
    float* hidden = nullptr;
    float* normalized = nullptr;
    float* delta_qkv = nullptr;
    float* delta_z = nullptr;
    float* delta_a = nullptr;
    float* delta_b = nullptr;
    float* delta_q = nullptr;
    float* delta_k = nullptr;
    float* delta_output = nullptr;
    float* query_and_gate = nullptr;
    float* query = nullptr;
    float* attention_gate = nullptr;
    float* key = nullptr;
    float* value = nullptr;
    float* attention_output = nullptr;
    float* ffn_gate = nullptr;
    float* ffn_up = nullptr;

    void allocate() {
        const size_t stride = 2 * H + DQKV + DO + 2 * VH + 3 * DO +
                              5 * AS + 2 * KV_WIDTH + 2 * I;
        CUDA_OK(cudaMalloc(&storage,
                           static_cast<size_t>(PREFILL_CHUNK) * stride *
                           sizeof(float)));
        float* cursor = storage;
#define TAKE_BATCH(name, width) \
        name = cursor; cursor += static_cast<size_t>(PREFILL_CHUNK) * (width)
        TAKE_BATCH(hidden, H);
        TAKE_BATCH(normalized, H);
        TAKE_BATCH(delta_qkv, DQKV);
        TAKE_BATCH(delta_z, DO);
        TAKE_BATCH(delta_a, VH);
        TAKE_BATCH(delta_b, VH);
        TAKE_BATCH(delta_q, DO);
        TAKE_BATCH(delta_k, DO);
        TAKE_BATCH(delta_output, DO);
        TAKE_BATCH(query_and_gate, 2 * AS);
        TAKE_BATCH(query, AS);
        TAKE_BATCH(attention_gate, AS);
        TAKE_BATCH(key, KV_WIDTH);
        TAKE_BATCH(value, KV_WIDTH);
        TAKE_BATCH(attention_output, AS);
        TAKE_BATCH(ffn_gate, I);
        TAKE_BATCH(ffn_up, I);
#undef TAKE_BATCH
        Q35_ASSERT(cursor == storage + PREFILL_CHUNK * stride,
                   "CUDA BatchWork layout cursor=%p end=%p",
                   static_cast<void*>(cursor),
                   static_cast<void*>(storage + PREFILL_CHUNK * stride));
    }

    void release() {
        if (storage) cudaFree(storage);
        storage = nullptr;
    }
};

struct StateData {
    int position = 0;
    int capacity = 0;
    int checkpoint_position = 0;
    std::array<float*, N> conv {};
    std::array<float*, N> memory {};
    std::array<float*, N> key_cache {};
    std::array<float*, N> value_cache {};
    std::array<float*, N> checkpoint_conv {};
    std::array<float*, N> checkpoint_memory {};
    float* checkpoint_logits = nullptr;
    int* device_tokens = nullptr;
    int* device_position = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraphExec_t forward_graph = nullptr;
    cudaGraphExec_t logits_graph = nullptr;
    Work work;
    BatchWork batch;
    std::vector<float> host_logits;

    explicit StateData(int context_size)
        : capacity(context_size), host_logits(V) {
        CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        work.allocate();
        batch.allocate();
        CUDA_OK(cudaMalloc(&checkpoint_logits, static_cast<size_t>(V) * sizeof(float)));
        CUDA_OK(cudaMalloc(&device_tokens,
                           static_cast<size_t>(capacity) * sizeof(int)));
        CUDA_OK(cudaMalloc(&device_position, sizeof(int)));
        CUDA_OK(cudaMemsetAsync(device_position, 0, sizeof(int), stream));
        for (int layer = 0; layer < N; ++layer) {
            if (layer % 4 != 3) {
                const size_t conv_bytes = static_cast<size_t>(DQKV) * (CK - 1) * sizeof(float);
                const size_t memory_bytes = static_cast<size_t>(VH) * KD * VD * sizeof(float);
                CUDA_OK(cudaMalloc(&conv[layer], conv_bytes));
                CUDA_OK(cudaMalloc(&memory[layer], memory_bytes));
                CUDA_OK(cudaMalloc(&checkpoint_conv[layer], conv_bytes));
                CUDA_OK(cudaMalloc(&checkpoint_memory[layer], memory_bytes));
                CUDA_OK(cudaMemset(conv[layer], 0, conv_bytes));
                CUDA_OK(cudaMemset(memory[layer], 0, memory_bytes));
            } else {
                const size_t cache_bytes = static_cast<size_t>(capacity) * KV_WIDTH * sizeof(float);
                CUDA_OK(cudaMalloc(&key_cache[layer], cache_bytes));
                CUDA_OK(cudaMalloc(&value_cache[layer], cache_bytes));
            }
        }
    }

    ~StateData() {
        if (forward_graph) cudaGraphExecDestroy(forward_graph);
        if (logits_graph) cudaGraphExecDestroy(logits_graph);
        work.release();
        batch.release();
        if (checkpoint_logits) cudaFree(checkpoint_logits);
        if (device_tokens) cudaFree(device_tokens);
        if (device_position) cudaFree(device_position);
        for (int layer = 0; layer < N; ++layer) {
            if (conv[layer]) cudaFree(conv[layer]);
            if (memory[layer]) cudaFree(memory[layer]);
            if (key_cache[layer]) cudaFree(key_cache[layer]);
            if (value_cache[layer]) cudaFree(value_cache[layer]);
            if (checkpoint_conv[layer]) cudaFree(checkpoint_conv[layer]);
            if (checkpoint_memory[layer]) cudaFree(checkpoint_memory[layer]);
        }
        if (stream) cudaStreamDestroy(stream);
    }
};

__device__ float block_sum(float value, float* shared) {
    const int thread = threadIdx.x;
    const int lane = thread % 32;
    const int warp = thread / 32;
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = lane < blockDim.x / 32 ? shared[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        if (lane == 0) shared[0] = value;
    }
    __syncthreads();
    return shared[0];
}

__device__ float f32(BF16 value) {
    return __uint_as_float(static_cast<uint32_t>(value) << 16);
}

__global__ void embed_kernel(const BF16* table, const int* tokens,
                             const int* position, float* output) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < H) {
        const int token = tokens[*position];
        output[index] = f32(table[static_cast<size_t>(token) * H + index]);
    }
}

__global__ void mv_kernel(const BF16* weight, const float* input,
                          float* output, int rows, int columns) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        sum += f32(weight[static_cast<size_t>(row) * columns + column]) * input[column];
    }
    __shared__ float shared[BLOCK];
    const float total = block_sum(sum, shared);
    if (threadIdx.x == 0) output[row] = total;
}

__global__ void mv_add_kernel(const BF16* weight, const float* input,
                              float* output, int rows, int columns) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        sum += f32(weight[static_cast<size_t>(row) * columns + column]) *
               input[column];
    }
    __shared__ float shared[BLOCK];
    const float total = block_sum(sum, shared);
    if (threadIdx.x == 0) output[row] += total;
}

__device__ float dot_row(const BF16* weight, int row, const float* input,
                         int columns, float* shared) {
    float sum = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        sum += f32(weight[static_cast<size_t>(row) * columns + column]) *
               input[column];
    }
    return block_sum(sum, shared);
}

__global__ void delta_projections_kernel(
    const BF16* qkv_weight, const BF16* z_weight,
    const BF16* a_weight, const BF16* b_weight,
    const float* input, float* qkv, float* z, float* a, float* b) {
    const int block = blockIdx.x;
    int row = block;
    const BF16* weight = qkv_weight;
    float* output = qkv;
    if (block >= DQKV + DO + VH) {
        row = block - DQKV - DO - VH;
        weight = b_weight;
        output = b;
    } else if (block >= DQKV + DO) {
        row = block - DQKV - DO;
        weight = a_weight;
        output = a;
    } else if (block >= DQKV) {
        row = block - DQKV;
        weight = z_weight;
        output = z;
    }
    __shared__ float shared[BLOCK];
    const float total = dot_row(weight, row, input, H, shared);
    if (threadIdx.x == 0) output[row] = total;
}

__global__ void attention_projections_kernel(
    const BF16* q_weight, const BF16* k_weight, const BF16* v_weight,
    const float* input, float* query_and_gate, float* key, float* value) {
    const int block = blockIdx.x;
    int row = block;
    const BF16* weight = q_weight;
    float* output = query_and_gate;
    if (block >= 2 * AS + KV_WIDTH) {
        row = block - 2 * AS - KV_WIDTH;
        weight = v_weight;
        output = value;
    } else if (block >= 2 * AS) {
        row = block - 2 * AS;
        weight = k_weight;
        output = key;
    }
    __shared__ float shared[BLOCK];
    const float total = dot_row(weight, row, input, H, shared);
    if (threadIdx.x == 0) output[row] = total;
}

__global__ void ffn_projections_kernel(
    const BF16* gate_weight, const BF16* up_weight,
    const float* input, float* gate, float* up) {
    int row = blockIdx.x;
    const BF16* weight = gate_weight;
    float* output = gate;
    if (row >= I) {
        row -= I;
        weight = up_weight;
        output = up;
    }
    __shared__ float shared[BLOCK];
    const float total = dot_row(weight, row, input, H, shared);
    if (threadIdx.x == 0) output[row] = total;
}

__global__ void batch_embed_kernel(const BF16* table, const int* tokens,
                                   int start, int count, float* output) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = count * H;
    if (index < total) {
        const int token_index = index / H;
        const int hidden_index = index % H;
        const int token = tokens[start + token_index];
        output[index] = f32(table[static_cast<size_t>(token) * H + hidden_index]);
    }
}

__global__ void batch_rms_kernel(const float* input, const BF16* weight,
                                 int width, int count, float* output) {
    const int token = blockIdx.x;
    if (token >= count) return;
    const float* source = input + static_cast<size_t>(token) * width;
    float* destination = output + static_cast<size_t>(token) * width;
    float square = 0.0f;
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        square += source[index] * source[index];
    }
    __shared__ float shared[BLOCK];
    const float total = block_sum(square, shared);
    const float scale = rsqrtf(total / width + EPS);
    for (int index = threadIdx.x; index < width; index += blockDim.x) {
        destination[index] = source[index] * scale * (1.0f + f32(weight[index]));
    }
}

__global__ void batch_mv_add_kernel(const BF16* weight, const float* input,
                                    float* output, int rows, int columns,
                                    int count) {
    const int row = blockIdx.x;
    const int token = blockIdx.y;
    if (row >= rows || token >= count) return;
    __shared__ float shared[BLOCK];
    const float total = dot_row(
        weight, row, input + static_cast<size_t>(token) * columns,
        columns, shared);
    if (threadIdx.x == 0) {
        output[static_cast<size_t>(token) * rows + row] += total;
    }
}

__global__ void batch_delta_projections_kernel(
    const BF16* qkv_weight, const BF16* z_weight,
    const BF16* a_weight, const BF16* b_weight,
    const float* input, int count,
    float* qkv, float* z, float* a, float* b) {
    const int token = blockIdx.y;
    if (token >= count) return;
    const int block = blockIdx.x;
    int row = block;
    int width = DQKV;
    const BF16* weight = qkv_weight;
    float* output = qkv;
    if (block >= DQKV + DO + VH) {
        row = block - DQKV - DO - VH;
        width = VH;
        weight = b_weight;
        output = b;
    } else if (block >= DQKV + DO) {
        row = block - DQKV - DO;
        width = VH;
        weight = a_weight;
        output = a;
    } else if (block >= DQKV) {
        row = block - DQKV;
        width = DO;
        weight = z_weight;
        output = z;
    }
    __shared__ float shared[BLOCK];
    const float total = dot_row(
        weight, row, input + static_cast<size_t>(token) * H, H, shared);
    if (threadIdx.x == 0) output[static_cast<size_t>(token) * width + row] = total;
}

__global__ void batch_attention_projections_kernel(
    const BF16* q_weight, const BF16* k_weight, const BF16* v_weight,
    const float* input, int count,
    float* query_and_gate, float* key, float* value) {
    const int token = blockIdx.y;
    if (token >= count) return;
    const int block = blockIdx.x;
    int row = block;
    int width = 2 * AS;
    const BF16* weight = q_weight;
    float* output = query_and_gate;
    if (block >= 2 * AS + KV_WIDTH) {
        row = block - 2 * AS - KV_WIDTH;
        width = KV_WIDTH;
        weight = v_weight;
        output = value;
    } else if (block >= 2 * AS) {
        row = block - 2 * AS;
        width = KV_WIDTH;
        weight = k_weight;
        output = key;
    }
    __shared__ float shared[BLOCK];
    const float total = dot_row(
        weight, row, input + static_cast<size_t>(token) * H, H, shared);
    if (threadIdx.x == 0) output[static_cast<size_t>(token) * width + row] = total;
}

__global__ void batch_ffn_projections_kernel(
    const BF16* gate_weight, const BF16* up_weight,
    const float* input, int count, float* gate, float* up) {
    const int token = blockIdx.y;
    if (token >= count) return;
    int row = blockIdx.x;
    const BF16* weight = gate_weight;
    float* output = gate;
    if (row >= I) {
        row -= I;
        weight = up_weight;
        output = up;
    }
    __shared__ float shared[BLOCK];
    const float total = dot_row(
        weight, row, input + static_cast<size_t>(token) * H, H, shared);
    if (threadIdx.x == 0) output[static_cast<size_t>(token) * I + row] = total;
}

__global__ void batch_swiglu_kernel(float* gate, const float* up, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count * I) {
        gate[index] = gate[index] / (1.0f + expf(-gate[index])) * up[index];
    }
}

__global__ void batch_conv_kernel(float* qkv, const BF16* weight,
                                  float* history, int count) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= DQKV) return;
    float* past = history + static_cast<size_t>(channel) * (CK - 1);
    for (int token = 0; token < count; ++token) {
        float* value = qkv + static_cast<size_t>(token) * DQKV + channel;
        float sum = *value * f32(weight[channel * CK + CK - 1]);
        for (int index = 0; index < CK - 1; ++index) {
            sum += past[index] * f32(weight[channel * CK + index]);
        }
        for (int index = 0; index < CK - 2; ++index) past[index] = past[index + 1];
        past[CK - 2] = *value;
        *value = sum / (1.0f + expf(-sum));
    }
}

__global__ void batch_prepare_delta_qk_kernel(
    const float* qkv, int count, float* query, float* key) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    if (token >= count) return;
    const int index = threadIdx.x;
    const float* source = qkv + static_cast<size_t>(token) * DQKV;
    float q = index < KD ? source[head * KD + index] : 0.0f;
    float k = index < KD ? source[DQK + head * KD + index] : 0.0f;
    __shared__ float shared[BLOCK];
    const float q_square = block_sum(q * q, shared);
    if (index < KD) {
        query[(static_cast<size_t>(token) * VH + head) * KD + index] =
            q * rsqrtf(q_square + EPS) / sqrtf(static_cast<float>(KD));
    }
    __syncthreads();
    const float k_square = block_sum(k * k, shared);
    if (index < KD) {
        key[(static_cast<size_t>(token) * VH + head) * KD + index] =
            k * rsqrtf(k_square + EPS);
    }
}

__global__ void batch_delta_rule_kernel(
    const float* query, const float* key, const float* qkv,
    const float* a, const float* b, const float* alog, const BF16* dt,
    int count, float* memory, float* output) {
    const int head = blockIdx.x;
    const int thread = threadIdx.x;
    float* state = memory + static_cast<size_t>(head) * KD * VD;
    __shared__ float predicted[VD];
    for (int token = 0; token < count; ++token) {
        const float* q = query + (static_cast<size_t>(token) * VH + head) * KD;
        const float* k = key + (static_cast<size_t>(token) * VH + head) * KD;
        const float* v = qkv + static_cast<size_t>(token) * DQKV + 2 * DQK + head * VD;
        const float beta = 1.0f / (1.0f + expf(-b[token * VH + head]));
        const float biased_a = a[token * VH + head] + f32(dt[head]);
        const float softplus = biased_a > 20.0f
            ? biased_a : log1pf(expf(biased_a));
        const float decay = expf(-expf(alog[head]) * softplus);
        if (thread < VD) {
            float sum = 0.0f;
            for (int index = 0; index < KD; ++index) {
                sum += k[index] * decay * state[index * VD + thread];
            }
            predicted[thread] = sum;
        }
        __syncthreads();
        for (int index = thread; index < KD * VD; index += blockDim.x) {
            const int key_index = index / VD;
            const int value_index = index % VD;
            state[index] = decay * state[index] + k[key_index] * beta *
                (v[value_index] - predicted[value_index]);
        }
        __syncthreads();
        if (thread < VD) {
            float sum = 0.0f;
            for (int index = 0; index < KD; ++index) {
                sum += q[index] * state[index * VD + thread];
            }
            output[(static_cast<size_t>(token) * VH + head) * VD + thread] = sum;
        }
        __syncthreads();
    }
}

__global__ void batch_gated_rms_kernel(float* values, const float* weight,
                                       const float* gate, int count) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    if (token >= count) return;
    const int index = threadIdx.x;
    float* segment = values + (static_cast<size_t>(token) * VH + head) * VD;
    const float square = index < VD ? segment[index] * segment[index] : 0.0f;
    __shared__ float shared[BLOCK];
    const float total = block_sum(square, shared);
    if (index < VD) {
        const float z = gate[(static_cast<size_t>(token) * VH + head) * VD + index];
        segment[index] *= rsqrtf(total / VD + EPS) * weight[index] *
                          z / (1.0f + expf(-z));
    }
}

__global__ void batch_prepare_query_kernel(
    const float* packed, const BF16* norm, int start, int count,
    float* query, float* gate) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    if (token >= count) return;
    const int index = threadIdx.x;
    const float* source = packed + static_cast<size_t>(token) * 2 * AS;
    float* q = query + (static_cast<size_t>(token) * AH + head) * AD;
    float* g = gate + (static_cast<size_t>(token) * AH + head) * AD;
    float value = source[head * 2 * AD + index];
    q[index] = value;
    g[index] = source[head * 2 * AD + AD + index];
    __shared__ float shared[BLOCK];
    const float total = block_sum(value * value, shared);
    q[index] = value * rsqrtf(total / AD + EPS) * (1.0f + f32(norm[index]));
    __syncthreads();
    if (index < RD / 2) {
        const float angle = (start + token) / powf(THETA, 2.0f * index / RD);
        const float left = q[index], right = q[index + RD / 2];
        const float cosine = cosf(angle), sine = sinf(angle);
        q[index] = left * cosine - right * sine;
        q[index + RD / 2] = right * cosine + left * sine;
    }
}

__global__ void batch_prepare_key_kernel(
    float* key, const BF16* norm, int start, int count) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    if (token >= count) return;
    const int index = threadIdx.x;
    float* k = key + (static_cast<size_t>(token) * KVH + head) * AD;
    const float value = k[index];
    __shared__ float shared[BLOCK];
    const float total = block_sum(value * value, shared);
    k[index] = value * rsqrtf(total / AD + EPS) * (1.0f + f32(norm[index]));
    __syncthreads();
    if (index < RD / 2) {
        const float angle = (start + token) / powf(THETA, 2.0f * index / RD);
        const float left = k[index], right = k[index + RD / 2];
        const float cosine = cosf(angle), sine = sinf(angle);
        k[index] = left * cosine - right * sine;
        k[index + RD / 2] = right * cosine + left * sine;
    }
}

__global__ void batch_store_kv_kernel(
    float* key_cache, float* value_cache, int start, int count,
    const float* key, const float* value) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count * KV_WIDTH) {
        const int token = index / KV_WIDTH;
        const int channel = index % KV_WIDTH;
        const size_t cache = static_cast<size_t>(start + token) * KV_WIDTH + channel;
        key_cache[cache] = key[index];
        value_cache[cache] = value[index];
    }
}

__global__ void batch_attention_kernel(
    const float* query, const float* gate,
    const float* key_cache, const float* value_cache,
    int start, int count, float* output) {
    const int head = blockIdx.x;
    const int query_token = blockIdx.y;
    if (query_token >= count) return;
    const int index = threadIdx.x;
    const int kv_head = head / (AH / KVH);
    const int position = start + query_token;
    const float q = query[(static_cast<size_t>(query_token) * AH + head) * AD + index];
    float accumulator = 0.0f;
    float maximum = -3.402823466e+38F;
    float denominator = 0.0f;
    __shared__ float shared[BLOCK];
    __shared__ float alpha;
    __shared__ float beta;
    for (int token = 0; token <= position; ++token) {
        const size_t cache = (static_cast<size_t>(token) * KVH + kv_head) * AD;
        const float score = block_sum(q * key_cache[cache + index], shared) /
                            sqrtf(static_cast<float>(AD));
        if (index == 0) {
            const float next_maximum = fmaxf(maximum, score);
            alpha = expf(maximum - next_maximum);
            beta = expf(score - next_maximum);
            denominator = denominator * alpha + beta;
            maximum = next_maximum;
        }
        __syncthreads();
        accumulator = accumulator * alpha + beta * value_cache[cache + index];
        __syncthreads();
    }
    __shared__ float normalizer;
    if (index == 0) normalizer = denominator;
    __syncthreads();
    const size_t offset = (static_cast<size_t>(query_token) * AH + head) * AD + index;
    const float z = gate[offset];
    output[offset] = accumulator / normalizer / (1.0f + expf(-z));
}

__global__ void rms_kernel(const float* input, const BF16* weight,
                           float* output, int count) {
    float square = 0.0f;
    for (int index = threadIdx.x; index < count; index += blockDim.x) {
        square += input[index] * input[index];
    }
    __shared__ float shared[BLOCK];
    const float total = block_sum(square, shared);
    const float scale = rsqrtf(total / count + EPS);
    for (int index = threadIdx.x; index < count; index += blockDim.x) {
        output[index] = input[index] * scale * (1.0f + f32(weight[index]));
    }
}

__global__ void swiglu_kernel(float* gate, const float* up) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < I) gate[index] = gate[index] / (1.0f + expf(-gate[index])) * up[index];
}

__global__ void conv_kernel(float* qkv, const BF16* weight, float* history) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= DQKV) return;
    float* past = history + static_cast<size_t>(channel) * (CK - 1);
    float sum = qkv[channel] * f32(weight[channel * CK + CK - 1]);
    for (int index = 0; index < CK - 1; ++index) {
        sum += past[index] * f32(weight[channel * CK + index]);
    }
    for (int index = 0; index < CK - 2; ++index) past[index] = past[index + 1];
    past[CK - 2] = qkv[channel];
    qkv[channel] = sum / (1.0f + expf(-sum));
}

__global__ void prepare_delta_qk_kernel(const float* qkv, float* query,
                                        float* key) {
    const int head = blockIdx.x;
    const int index = threadIdx.x;
    float q = index < KD ? qkv[head * KD + index] : 0.0f;
    float k = index < KD ? qkv[DQK + head * KD + index] : 0.0f;
    __shared__ float shared[BLOCK];
    const float q_square = block_sum(q * q, shared);
    const float q_scale = rsqrtf(q_square + EPS) / sqrtf(static_cast<float>(KD));
    if (index < KD) query[head * KD + index] = q * q_scale;
    __syncthreads();
    const float k_square = block_sum(k * k, shared);
    const float k_scale = rsqrtf(k_square + EPS);
    if (index < KD) key[head * KD + index] = k * k_scale;
}

__global__ void delta_rule_kernel(const float* query, const float* key,
                                  const float* value, const float* a,
                                  const float* b, const float* alog,
                                  const BF16* dt, float* memory,
                                  float* output) {
    const int head = blockIdx.x;
    const int thread = threadIdx.x;
    float* state = memory + static_cast<size_t>(head) * KD * VD;
    const float* q = query + head * KD;
    const float* k = key + head * KD;
    const float* v = value + head * VD;
    const float beta = 1.0f / (1.0f + expf(-b[head]));
    const float biased_a = a[head] + f32(dt[head]);
    const float softplus = biased_a > 20.0f
        ? biased_a : log1pf(expf(biased_a));
    const float decay = expf(-expf(alog[head]) * softplus);
    __shared__ float predicted[VD];
    if (thread < VD) {
        float sum = 0.0f;
        for (int index = 0; index < KD; ++index) {
            sum += k[index] * decay * state[index * VD + thread];
        }
        predicted[thread] = sum;
    }
    __syncthreads();
    for (int index = thread; index < KD * VD; index += blockDim.x) {
        const int key_index = index / VD;
        const int value_index = index % VD;
        state[index] = decay * state[index] +
                       k[key_index] * beta * (v[value_index] - predicted[value_index]);
    }
    __syncthreads();
    if (thread < VD) {
        float sum = 0.0f;
        for (int index = 0; index < KD; ++index) {
            sum += q[index] * state[index * VD + thread];
        }
        output[head * VD + thread] = sum;
    }
}

__global__ void gated_rms_kernel(float* values, const float* weight,
                                 const float* gate) {
    const int head = blockIdx.x;
    const int index = threadIdx.x;
    float* segment = values + head * VD;
    float square = index < VD ? segment[index] * segment[index] : 0.0f;
    __shared__ float shared[BLOCK];
    const float total = block_sum(square, shared);
    if (index < VD) {
        const float z = gate[head * VD + index];
        const float silu = z / (1.0f + expf(-z));
        segment[index] *= rsqrtf(total / VD + EPS) * weight[index] * silu;
    }
}

__global__ void prepare_query_kernel(const float* packed, const BF16* norm,
                                     const int* device_position,
                                     float* query, float* gate) {
    const int head = blockIdx.x;
    const int index = threadIdx.x;
    float value = packed[head * 2 * AD + index];
    query[head * AD + index] = value;
    gate[head * AD + index] = packed[head * 2 * AD + AD + index];
    __shared__ float shared[BLOCK];
    const float total = block_sum(value * value, shared);
    value *= rsqrtf(total / AD + EPS) * (1.0f + f32(norm[index]));
    query[head * AD + index] = value;
    __syncthreads();
    if (index < RD / 2) {
        const float angle = *device_position / powf(THETA, 2.0f * index / RD);
        const float left = query[head * AD + index];
        const float right = query[head * AD + index + RD / 2];
        const float cosine = cosf(angle), sine = sinf(angle);
        query[head * AD + index] = left * cosine - right * sine;
        query[head * AD + index + RD / 2] = right * cosine + left * sine;
    }
}

__global__ void prepare_key_kernel(float* key, const BF16* norm,
                                   const int* device_position) {
    const int head = blockIdx.x;
    const int index = threadIdx.x;
    float value = key[head * AD + index];
    __shared__ float shared[BLOCK];
    const float total = block_sum(value * value, shared);
    key[head * AD + index] = value * rsqrtf(total / AD + EPS) *
                             (1.0f + f32(norm[index]));
    __syncthreads();
    if (index < RD / 2) {
        const float angle = *device_position / powf(THETA, 2.0f * index / RD);
        const float left = key[head * AD + index];
        const float right = key[head * AD + index + RD / 2];
        const float cosine = cosf(angle), sine = sinf(angle);
        key[head * AD + index] = left * cosine - right * sine;
        key[head * AD + index + RD / 2] = right * cosine + left * sine;
    }
}

__global__ void store_kv_kernel(float* key_cache, float* value_cache,
                                const int* device_position, const float* key,
                                const float* value) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < KV_WIDTH) {
        const size_t offset = static_cast<size_t>(*device_position) * KV_WIDTH + index;
        key_cache[offset] = key[index];
        value_cache[offset] = value[index];
    }
}

__global__ void attention_kernel(const float* query, const float* gate,
                                 const float* key_cache,
                                 const float* value_cache,
                                 const int* device_position,
                                 float* output) {
    const int head = blockIdx.x;
    const int index = threadIdx.x;
    const int kv_head = head / (AH / KVH);
    const int position = *device_position;
    const float q = query[head * AD + index];
    float accumulator = 0.0f;
    float maximum = -3.402823466e+38F;
    float denominator = 0.0f;
    __shared__ float shared[BLOCK];
    __shared__ float alpha;
    __shared__ float beta;
    for (int token = 0; token <= position; ++token) {
        const size_t cache = (static_cast<size_t>(token) * KVH + kv_head) * AD;
        const float score = block_sum(q * key_cache[cache + index], shared) /
                            sqrtf(static_cast<float>(AD));
        if (index == 0) {
            const float next_maximum = fmaxf(maximum, score);
            alpha = expf(maximum - next_maximum);
            beta = expf(score - next_maximum);
            denominator = denominator * alpha + beta;
            maximum = next_maximum;
        }
        __syncthreads();
        accumulator = accumulator * alpha + beta * value_cache[cache + index];
        __syncthreads();
    }
    __shared__ float normalizer;
    if (index == 0) normalizer = denominator;
    __syncthreads();
    const float z = gate[head * AD + index];
    output[head * AD + index] = accumulator / normalizer /
                                (1.0f + expf(-z));
}

__global__ void advance_position_kernel(int* position) {
    if (threadIdx.x == 0) ++*position;
}

void mv(const Linear& linear, const float* input, float* output,
        cudaStream_t stream) {
    mv_kernel<<<linear.rows, BLOCK, 0, stream>>>(
        linear.weight, input, output, linear.rows, linear.columns);
}

void mv_add(const Linear& linear, const float* input, float* output,
            cudaStream_t stream) {
    mv_add_kernel<<<linear.rows, BLOCK, 0, stream>>>(
        linear.weight, input, output, linear.rows, linear.columns);
}

void rms(const float* input, const BF16* weight, int count, float* output,
         cudaStream_t stream) {
    rms_kernel<<<1, BLOCK, 0, stream>>>(input, weight, output, count);
}

void deltanet(const DeltaWeights& weights, float* conv, float* memory,
              const float* input, Work& work, float* output,
              cudaStream_t stream) {
    delta_projections_kernel<<<DQKV + DO + 2 * VH, BLOCK, 0, stream>>>(
        weights.qkv.weight, weights.z.weight, weights.a.weight,
        weights.b.weight, input, work.delta_qkv, work.delta_z,
        work.delta_a, work.delta_b);
    conv_kernel<<<(DQKV + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        work.delta_qkv, weights.conv, conv);
    prepare_delta_qk_kernel<<<VH, BLOCK, 0, stream>>>(
        work.delta_qkv, work.delta_q, work.delta_k);
    delta_rule_kernel<<<VH, BLOCK, 0, stream>>>(
        work.delta_q, work.delta_k, work.delta_qkv + 2 * DQK,
        work.delta_a, work.delta_b, weights.alog, weights.dt,
        memory, work.delta_output);
    gated_rms_kernel<<<VH, BLOCK, 0, stream>>>(
        work.delta_output, weights.norm, work.delta_z);
    mv_add(weights.out, work.delta_output, output, stream);
}

void attention(const AttentionWeights& weights, float* key_cache,
               float* value_cache, const int* position, const float* input,
               Work& work, float* output, cudaStream_t stream) {
    attention_projections_kernel<<<2 * AS + 2 * KV_WIDTH, BLOCK, 0, stream>>>(
        weights.q.weight, weights.k.weight, weights.v.weight, input,
        work.query_and_gate, work.key, work.value);
    prepare_query_kernel<<<AH, BLOCK, 0, stream>>>(
        work.query_and_gate, weights.qnorm, position, work.query,
        work.attention_gate);
    prepare_key_kernel<<<KVH, BLOCK, 0, stream>>>(
        work.key, weights.knorm, position);
    store_kv_kernel<<<(KV_WIDTH + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        key_cache, value_cache, position, work.key, work.value);
    attention_kernel<<<AH, BLOCK, 0, stream>>>(
        work.query, work.attention_gate, key_cache, value_cache,
        position, work.attention_output);
    mv_add(weights.out, work.attention_output, output, stream);
}

void ffn(const Layer& layer, const float* input, Work& work, float* output,
         cudaStream_t stream) {
    ffn_projections_kernel<<<2 * I, BLOCK, 0, stream>>>(
        layer.gate.weight, layer.up.weight, input,
        work.ffn_gate, work.ffn_up);
    swiglu_kernel<<<(I + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        work.ffn_gate, work.ffn_up);
    mv_add(layer.down, work.ffn_gate, output, stream);
}

void batch_deltanet(const DeltaWeights& weights, float* conv, float* memory,
                    const float* input, BatchWork& work, float* hidden,
                    int count, cudaStream_t stream) {
    const dim3 projection_grid(DQKV + DO + 2 * VH, count);
    batch_delta_projections_kernel<<<projection_grid, BLOCK, 0, stream>>>(
        weights.qkv.weight, weights.z.weight, weights.a.weight,
        weights.b.weight, input, count, work.delta_qkv, work.delta_z,
        work.delta_a, work.delta_b);
    batch_conv_kernel<<<(DQKV + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        work.delta_qkv, weights.conv, conv, count);
    const dim3 head_token_grid(VH, count);
    batch_prepare_delta_qk_kernel<<<head_token_grid, BLOCK, 0, stream>>>(
        work.delta_qkv, count, work.delta_q, work.delta_k);
    batch_delta_rule_kernel<<<VH, BLOCK, 0, stream>>>(
        work.delta_q, work.delta_k, work.delta_qkv,
        work.delta_a, work.delta_b, weights.alog, weights.dt,
        count, memory, work.delta_output);
    batch_gated_rms_kernel<<<head_token_grid, BLOCK, 0, stream>>>(
        work.delta_output, weights.norm, work.delta_z, count);
    const dim3 output_grid(H, count);
    batch_mv_add_kernel<<<output_grid, BLOCK, 0, stream>>>(
        weights.out.weight, work.delta_output, hidden,
        H, DO, count);
}

void batch_attention(const AttentionWeights& weights,
                     float* key_cache, float* value_cache,
                     int start, const float* input, BatchWork& work,
                     float* hidden, int count, cudaStream_t stream) {
    const dim3 projection_grid(2 * AS + 2 * KV_WIDTH, count);
    batch_attention_projections_kernel<<<projection_grid, BLOCK, 0, stream>>>(
        weights.q.weight, weights.k.weight, weights.v.weight,
        input, count, work.query_and_gate, work.key, work.value);
    const dim3 query_grid(AH, count);
    const dim3 key_grid(KVH, count);
    batch_prepare_query_kernel<<<query_grid, BLOCK, 0, stream>>>(
        work.query_and_gate, weights.qnorm, start, count,
        work.query, work.attention_gate);
    batch_prepare_key_kernel<<<key_grid, BLOCK, 0, stream>>>(
        work.key, weights.knorm, start, count);
    batch_store_kv_kernel<<<(count * KV_WIDTH + BLOCK - 1) / BLOCK,
                            BLOCK, 0, stream>>>(
        key_cache, value_cache, start, count, work.key, work.value);
    batch_attention_kernel<<<query_grid, BLOCK, 0, stream>>>(
        work.query, work.attention_gate, key_cache, value_cache,
        start, count, work.attention_output);
    const dim3 output_grid(H, count);
    batch_mv_add_kernel<<<output_grid, BLOCK, 0, stream>>>(
        weights.out.weight, work.attention_output, hidden,
        H, AS, count);
}

void batch_ffn(const Layer& layer, const float* input, BatchWork& work,
               float* hidden, int count, cudaStream_t stream) {
    const dim3 projection_grid(2 * I, count);
    batch_ffn_projections_kernel<<<projection_grid, BLOCK, 0, stream>>>(
        layer.gate.weight, layer.up.weight, input, count,
        work.ffn_gate, work.ffn_up);
    batch_swiglu_kernel<<<(count * I + BLOCK - 1) / BLOCK,
                           BLOCK, 0, stream>>>(
        work.ffn_gate, work.ffn_up, count);
    const dim3 output_grid(H, count);
    batch_mv_add_kernel<<<output_grid, BLOCK, 0, stream>>>(
        layer.down.weight, work.ffn_gate, hidden,
        H, I, count);
}

void prefill_chunk(const ModelData& model, StateData& state,
                   int start, int count, bool compute_logits) {
    Q35_ASSERT(count > 0 && count <= PREFILL_CHUNK,
               "CUDA prefill chunk count=%d limit=%d", count, PREFILL_CHUNK);
    BatchWork& work = state.batch;
    cudaStream_t stream = state.stream;
    batch_embed_kernel<<<(count * H + BLOCK - 1) / BLOCK,
                          BLOCK, 0, stream>>>(
        model.embedding, state.device_tokens, start, count, work.hidden);
    for (int index = 0; index < N; ++index) {
        const Layer& layer = model.layers[index];
        batch_rms_kernel<<<count, BLOCK, 0, stream>>>(
            work.hidden, layer.input_norm, H, count, work.normalized);
        if (layer.uses_deltanet) {
            batch_deltanet(layer.delta, state.conv[index], state.memory[index],
                           work.normalized, work, work.hidden, count, stream);
        } else {
            batch_attention(layer.attention, state.key_cache[index],
                            state.value_cache[index], start,
                            work.normalized, work, work.hidden, count, stream);
        }
        batch_rms_kernel<<<count, BLOCK, 0, stream>>>(
            work.hidden, layer.post_norm, H, count, work.normalized);
        batch_ffn(layer, work.normalized, work, work.hidden, count, stream);
    }
    if (compute_logits) {
        const float* last = work.hidden + static_cast<size_t>(count - 1) * H;
        rms(last, model.final_norm, H, state.work.normalized, stream);
        mv({model.embedding, V, H}, state.work.normalized,
           state.work.logits, stream);
    }
}

void forward(const ModelData& model, StateData& state, bool compute_logits) {
    Work& work = state.work;
    cudaStream_t stream = state.stream;
    embed_kernel<<<(H + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        model.embedding, state.device_tokens, state.device_position,
        work.hidden);
    for (int index = 0; index < N; ++index) {
        const Layer& layer = model.layers[index];
        rms(work.hidden, layer.input_norm, H, work.normalized, stream);
        if (layer.uses_deltanet) {
            deltanet(layer.delta, state.conv[index], state.memory[index],
                     work.normalized, work, work.hidden, stream);
        } else {
            attention(layer.attention, state.key_cache[index],
                      state.value_cache[index], state.device_position,
                      work.normalized, work, work.hidden, stream);
        }
        rms(work.hidden, layer.post_norm, H, work.normalized, stream);
        ffn(layer, work.normalized, work, work.hidden, stream);
    }
    if (compute_logits) {
        rms(work.hidden, model.final_norm, H, work.normalized, stream);
        mv({model.embedding, V, H}, work.normalized, work.logits, stream);
    }
    advance_position_kernel<<<1, 1, 0, stream>>>(state.device_position);
}

cudaGraphExec_t capture_forward(const ModelData& model, StateData& state,
                                bool compute_logits) {
    CUDA_OK(cudaStreamBeginCapture(state.stream, cudaStreamCaptureModeThreadLocal));
    forward(model, state, compute_logits);
    cudaGraph_t graph = nullptr;
    CUDA_OK(cudaStreamEndCapture(state.stream, &graph));
    cudaGraphExec_t executable = nullptr;
    CUDA_OK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    CUDA_OK(cudaGraphDestroy(graph));
    return executable;
}

struct File {
    int descriptor = -1;
    size_t size = 0;
    const uint8_t* data = nullptr;

    bool map(const char* path, const char** error) {
        descriptor = open(path, O_RDONLY);
        if (descriptor < 0) return fail(error, "cannot open model.bin");
        struct stat information {};
        if (fstat(descriptor, &information) || information.st_size < 16) {
            return fail(error, "bad model.bin");
        }
        size = static_cast<size_t>(information.st_size);
        data = static_cast<const uint8_t*>(
            mmap(nullptr, size, PROT_READ, MAP_PRIVATE, descriptor, 0));
        if (data == MAP_FAILED) {
            data = nullptr;
            return fail(error, "mmap model.bin failed");
        }
        uint32_t version = 0, reserved = 0;
        if (std::memcmp(data, "Q35MODL\0", 8) != 0) {
            return fail(error, "wrong model.bin magic");
        }
        std::memcpy(&version, data + 8, sizeof(version));
        std::memcpy(&reserved, data + 12, sizeof(reserved));
        if (version != MODEL_FORMAT_VERSION || reserved != 0) {
            return fail(error, "unsupported model.bin version");
        }
        return true;
    }

    ~File() { release(); }

    void release() {
        if (data) munmap(const_cast<uint8_t*>(data), size);
        if (descriptor >= 0) close(descriptor);
        data = nullptr;
        descriptor = -1;
        size = 0;
    }

private:
    bool fail(const char** error, const char* message) {
        *error = message;
        release();
        return false;
    }
};

struct Reader {
    const uint8_t* begin;
    const uint8_t* cursor;
    const uint8_t* end;
    const char* error = nullptr;

    explicit Reader(const File& file)
        : begin(file.data), cursor(file.data + 16), end(file.data + file.size) {}

    template <typename T> const T* take(size_t count) {
        if (error) return nullptr;
        const size_t offset = static_cast<size_t>(cursor - begin);
        const size_t padding = (64 - offset % 64) % 64;
        if (padding > static_cast<size_t>(end - cursor)) {
            error = "truncated model.bin";
            return nullptr;
        }
        cursor += padding;
        if (count > std::numeric_limits<size_t>::max() / sizeof(T) ||
            count * sizeof(T) > static_cast<size_t>(end - cursor)) {
            error = "truncated model.bin";
            return nullptr;
        }
        const T* result = reinterpret_cast<const T*>(cursor);
        cursor += count * sizeof(T);
        return result;
    }
};

struct Loader {
    ModelData* model;
    Reader reader;
    const char* error = nullptr;

    Loader(ModelData* output, const File& file) : model(output), reader(file) {}

    const BF16* bf16(size_t count) {
        const BF16* source = reader.take<BF16>(count);
        if (!source) return nullptr;
        BF16* device = nullptr;
        cudaError_t status = cudaMalloc(&device, count * sizeof(BF16));
        if (status != cudaSuccess) {
            error = "CUDA model allocation failed";
            return nullptr;
        }
        model->allocations.push_back(device);
        status = cudaMemcpy(device, source, count * sizeof(BF16),
                            cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            error = "CUDA model upload failed";
            return nullptr;
        }
        return device;
    }

    const float* f32(size_t count) {
        const float* source = reader.take<float>(count);
        if (!source) return nullptr;
        float* device = nullptr;
        cudaError_t status = cudaMalloc(&device, count * sizeof(float));
        if (status != cudaSuccess) {
            error = "CUDA model allocation failed";
            return nullptr;
        }
        model->allocations.push_back(device);
        status = cudaMemcpy(device, source, count * sizeof(float),
                            cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            error = "CUDA model upload failed";
            return nullptr;
        }
        return device;
    }

    Linear linear(int rows, int columns) {
        return {bf16(static_cast<size_t>(rows) * columns), rows, columns};
    }

    bool load() {
        model->embedding = bf16(static_cast<size_t>(V) * H);
        model->final_norm = bf16(H);
        for (int index = 0; index < N && good(); ++index) {
            Layer& layer = model->layers[index];
            layer.uses_deltanet = index % 4 != 3;
            layer.input_norm = bf16(H);
            if (layer.uses_deltanet) {
                layer.delta.qkv = linear(DQKV, H);
                layer.delta.z = linear(DO, H);
                layer.delta.a = linear(VH, H);
                layer.delta.b = linear(VH, H);
                layer.delta.conv = bf16(static_cast<size_t>(DQKV) * CK);
                layer.delta.alog = f32(VH);
                layer.delta.dt = bf16(VH);
                layer.delta.norm = f32(VD);
                layer.delta.out = linear(H, DO);
            } else {
                layer.attention.q = linear(2 * AS, H);
                layer.attention.k = linear(KV_WIDTH, H);
                layer.attention.v = linear(KV_WIDTH, H);
                layer.attention.qnorm = bf16(AD);
                layer.attention.knorm = bf16(AD);
                layer.attention.out = linear(H, AS);
            }
            layer.post_norm = bf16(H);
            layer.gate = linear(I, H);
            layer.up = linear(I, H);
            layer.down = linear(H, I);
        }
        if (!good()) return false;
        if (reader.cursor != reader.end) {
            error = "model.bin size does not match Qwen3.5-0.8B schema";
            return false;
        }
        return true;
    }

    bool good() {
        if (!error && reader.error) error = reader.error;
        return !error;
    }
};

bool load_model(const char* path, ModelData* model, const char** error) {
    File file;
    if (!file.map(path, error)) return false;
    Loader loader(model, file);
    if (!loader.load()) {
        *error = loader.error ? loader.error : "cannot load CUDA model";
        return false;
    }
    return true;
}

}  // namespace qwen35_cuda

namespace q35_backend {

struct Model {
    qwen35_cuda::ModelData data;
};

struct State {
    qwen35_cuda::StateData data;
    explicit State(int context_size) : data(context_size) {}
};

Model* model_create(const char* path, char* err, size_t errlen) {
    Q35_ASSERT(path, "CUDA model_create path is null");
    std::unique_ptr<Model> model(new Model());
    const char* message = nullptr;
    if (!qwen35_cuda::load_model(path, &model->data, &message)) {
        if (err && errlen > 0) std::snprintf(err, errlen, "%s", message);
        return nullptr;
    }
    if (err && errlen > 0) err[0] = '\0';
    return model.release();
}

void model_destroy(Model* model) { delete model; }

State* state_create(Model* model, int context_size) {
    Q35_ASSERT(model, "CUDA state_create model is null");
    Q35_ASSERT(context_size > 0 && context_size <= qwen35_cuda::MAX_CONTEXT,
               "CUDA state_create context_size=%d", context_size);
    return new State(context_size);
}

void state_destroy(State* state) { delete state; }

void state_reset(State* state) {
    Q35_ASSERT(state, "CUDA state_reset state is null");
    using namespace qwen35_cuda;
    for (int layer = 0; layer < N; ++layer) {
        if (layer % 4 == 3) continue;
        CUDA_OK(cudaMemsetAsync(
            state->data.conv[layer], 0,
            static_cast<size_t>(DQKV) * (CK - 1) * sizeof(float),
            state->data.stream));
        CUDA_OK(cudaMemsetAsync(
            state->data.memory[layer], 0,
            static_cast<size_t>(VH) * KD * VD * sizeof(float),
            state->data.stream));
    }
    state->data.position = 0;
    CUDA_OK(cudaMemcpyAsync(state->data.device_position,
                            &state->data.position, sizeof(int),
                            cudaMemcpyHostToDevice, state->data.stream));
    CUDA_OK(cudaStreamSynchronize(state->data.stream));
}

void state_forward(Model* model, State* state,
                   const int* tokens, int count, bool compute_logits) {
    Q35_ASSERT(model && state && tokens && count > 0,
               "CUDA state_forward model=%p state=%p tokens=%p count=%d",
               static_cast<void*>(model), static_cast<void*>(state),
               static_cast<const void*>(tokens), count);
    qwen35_cuda::StateData& data = state->data;
    Q35_ASSERT(data.position >= 0 && data.position + count <= data.capacity,
               "CUDA state_forward position=%d count=%d capacity=%d",
               data.position, count, data.capacity);
    for (int index = 0; index < count; ++index) {
        Q35_ASSERT(tokens[index] >= 0 && tokens[index] < qwen35_cuda::V,
                   "CUDA state_forward token[%d]=%d vocabulary=%d",
                   index, tokens[index], qwen35_cuda::V);
    }
    CUDA_OK(cudaMemcpyAsync(
        data.device_tokens + data.position, tokens,
        static_cast<size_t>(count) * sizeof(int),
        cudaMemcpyHostToDevice, data.stream));
    if (count == 1) {
        if (!data.forward_graph) {
            data.forward_graph = qwen35_cuda::capture_forward(
                model->data, data, false);
            data.logits_graph = qwen35_cuda::capture_forward(
                model->data, data, true);
        }
        CUDA_OK(cudaMemcpyAsync(data.device_position, &data.position, sizeof(int),
                                cudaMemcpyHostToDevice, data.stream));
        cudaGraphExec_t graph = compute_logits
            ? data.logits_graph : data.forward_graph;
        CUDA_OK(cudaGraphLaunch(graph, data.stream));
    } else {
        int offset = 0;
        while (offset < count) {
            const int chunk = std::min(qwen35_cuda::PREFILL_CHUNK,
                                       count - offset);
            qwen35_cuda::prefill_chunk(
                model->data, data, data.position + offset, chunk,
                compute_logits && offset + chunk == count);
            offset += chunk;
        }
        const int final_position = data.position + count;
        CUDA_OK(cudaMemcpyAsync(data.device_position, &final_position,
                                sizeof(int), cudaMemcpyHostToDevice,
                                data.stream));
    }
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaStreamSynchronize(data.stream));
    data.position += count;
}

void state_checkpoint_save(State* state) {
    Q35_ASSERT(state, "CUDA state_checkpoint_save state is null");
    using namespace qwen35_cuda;
    StateData& data = state->data;
    data.checkpoint_position = data.position;
    for (int layer = 0; layer < N; ++layer) {
        if (layer % 4 == 3) continue;
        CUDA_OK(cudaMemcpyAsync(
            data.checkpoint_conv[layer], data.conv[layer],
            static_cast<size_t>(DQKV) * (CK - 1) * sizeof(float),
            cudaMemcpyDeviceToDevice, data.stream));
        CUDA_OK(cudaMemcpyAsync(
            data.checkpoint_memory[layer], data.memory[layer],
            static_cast<size_t>(VH) * KD * VD * sizeof(float),
            cudaMemcpyDeviceToDevice, data.stream));
    }
    CUDA_OK(cudaMemcpyAsync(data.checkpoint_logits, data.work.logits,
                            static_cast<size_t>(V) * sizeof(float),
                            cudaMemcpyDeviceToDevice, data.stream));
    CUDA_OK(cudaStreamSynchronize(data.stream));
}

void state_checkpoint_restore(State* state) {
    Q35_ASSERT(state, "CUDA state_checkpoint_restore state is null");
    using namespace qwen35_cuda;
    StateData& data = state->data;
    for (int layer = 0; layer < N; ++layer) {
        if (layer % 4 == 3) continue;
        CUDA_OK(cudaMemcpyAsync(
            data.conv[layer], data.checkpoint_conv[layer],
            static_cast<size_t>(DQKV) * (CK - 1) * sizeof(float),
            cudaMemcpyDeviceToDevice, data.stream));
        CUDA_OK(cudaMemcpyAsync(
            data.memory[layer], data.checkpoint_memory[layer],
            static_cast<size_t>(VH) * KD * VD * sizeof(float),
            cudaMemcpyDeviceToDevice, data.stream));
    }
    CUDA_OK(cudaMemcpyAsync(data.work.logits, data.checkpoint_logits,
                            static_cast<size_t>(V) * sizeof(float),
                            cudaMemcpyDeviceToDevice, data.stream));
    data.position = data.checkpoint_position;
    CUDA_OK(cudaMemcpyAsync(data.device_position, &data.position, sizeof(int),
                            cudaMemcpyHostToDevice, data.stream));
    CUDA_OK(cudaStreamSynchronize(data.stream));
}

int state_argmax(const State* state) {
    if (!state) return -1;
    State* mutable_state = const_cast<State*>(state);
    CUDA_OK(cudaMemcpy(mutable_state->data.host_logits.data(),
                       state->data.work.logits,
                       static_cast<size_t>(qwen35_cuda::V) * sizeof(float),
                       cudaMemcpyDeviceToHost));
    return static_cast<int>(std::max_element(
        mutable_state->data.host_logits.begin(),
        mutable_state->data.host_logits.end()) -
        mutable_state->data.host_logits.begin());
}

void state_copy_logits(const State* state, float* output) {
    Q35_ASSERT(state && output, "CUDA state_copy_logits state=%p output=%p",
               static_cast<const void*>(state), static_cast<void*>(output));
    CUDA_OK(cudaMemcpy(output, state->data.work.logits,
                       static_cast<size_t>(qwen35_cuda::V) * sizeof(float),
                       cudaMemcpyDeviceToHost));
}

int vocab_size() { return qwen35_cuda::V; }
int max_context() { return qwen35_cuda::MAX_CONTEXT; }
bool token_is_stop(int token) {
    return token == qwen35_cuda::END_OF_TEXT_TOKEN ||
           token == qwen35_cuda::IM_END_TOKEN;
}

}  // namespace q35_backend

// arch/cuda/engine.cu -- readable CUDA engine for supported Qwen models.
//
// This is the same model data flow as engine.cpp with a different physical
// home for Model, State and Work:
//
//   Model: packed BF16/Q8_0/Q4_0 weights stay packed in device memory
//   State: recurrent/KV state and its checkpoint stay on the GPU
//   Work:  Session-owned GPU scratch buffers reused by every forward
//
// Dense prefill uses explicit token chunks and dense decode uses a CUDA Graph.
// Qwen3.6 prefill/decode launches the same readable one-token forward. Both
// stay behind q3x_backend::state_forward() and do not enter runtime.cpp.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include "internal.h"
#include "model_config.h"
#include "q4.h"
#include "q8.h"

namespace qwen3x_cuda {

constexpr int AD = 256, RD = 64;
constexpr int KH = 16, KD = 128, VD = 128, CK = 4;
constexpr int DQK = KH * KD;
constexpr int MAX_CONTEXT = q3x_model::MAX_CONTEXT;
constexpr int END_OF_TEXT_TOKEN = 248044, IM_END_TOKEN = 248046;
constexpr float EPS = 1e-6f, THETA = 10000000.0f;
constexpr int BLOCK = 256;
constexpr int PREFILL_CHUNK = 256;
using BF16 = uint16_t;

void cuda_fatal(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    q3x_internal::report_assertion(
        "CUDA operation succeeded", __FILE__, __LINE__, "%s: %s",
        operation, cudaGetErrorString(status));
    std::abort();
}

#define CUDA_OK(call) qwen3x_cuda::cuda_fatal((call), #call)

void cublas_fatal(cublasStatus_t status, const char* operation) {
    if (status == CUBLAS_STATUS_SUCCESS) return;
    q3x_internal::report_assertion(
        "cuBLAS operation succeeded", __FILE__, __LINE__, "%s: %s",
        operation, cublasGetStatusString(status));
    std::abort();
}

#define CUBLAS_OK(call) qwen3x_cuda::cublas_fatal((call), #call)

struct Linear {
    const void* weight = nullptr;
    int rows = 0;
    int columns = 0;
    q3x_model::MatrixType type = q3x_model::MATRIX_BF16;
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

struct ExpertLinear {
    const void* weight = nullptr;
    int experts = 0;
    int rows = 0;
    int columns = 0;
    q3x_model::MatrixType type = q3x_model::MATRIX_Q4_0;
};

struct MoeWeights {
    Linear router;
    ExpertLinear gate_up, down;
    Linear shared_gate, shared_up, shared_down, shared_scale;
};

struct Layer {
    const BF16* input_norm = nullptr;
    const BF16* post_norm = nullptr;
    Linear gate, up, down;
    MoeWeights moe;
    DeltaWeights delta;
    AttentionWeights attention;
};

}  // namespace qwen3x_cuda

namespace q3x_backend {

struct Model {
    const q3x_model::ModelConfig* config = nullptr;
    qwen3x_cuda::Linear embedding, lm_head;
    const qwen3x_cuda::BF16* final_norm = nullptr;
    void* weights = nullptr;
    std::unique_ptr<qwen3x_cuda::Layer[]> layers;

    ~Model() {
        if (weights) cudaFree(weights);
    }
    bool load(const char* path, const char** error);
};

}  // namespace q3x_backend

namespace qwen3x_cuda {

using q3x_backend::Model;

struct DeviceFloatStorage {
    float* memory = nullptr;
    size_t count = 0;
    size_t used = 0;

    void allocate(size_t size) {
        Q3X_ASSERT(size > 0, "CUDA storage size=%zu", size);
        CUDA_OK(cudaMalloc(&memory, size * sizeof(float)));
        count = size;
        used = 0;
    }

    float* take(size_t size) {
        Q3X_ASSERT(used <= count && size <= count - used,
                   "CUDA storage exhausted used=%zu take=%zu count=%zu",
                   used, size, count);
        float* result = memory + used;
        used += size;
        return result;
    }

    void finish() const {
        Q3X_ASSERT(used == count, "CUDA storage unused=%zu count=%zu",
                   count - used, count);
    }

    void release() {
        if (memory) cudaFree(memory);
        memory = nullptr;
        count = 0;
        used = 0;
    }
};

struct LayerState {
    float* conv = nullptr;
    float* memory = nullptr;
    float* key_cache = nullptr;
    float* value_cache = nullptr;
};

struct Work {
    float* storage = nullptr;
    size_t allocated_bytes = 0;
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
    float* router_logits = nullptr;
    float* router_probabilities = nullptr;
    float* expert_gate_up = nullptr;
    float* expert_hidden = nullptr;
    float* expert_weights = nullptr;
    int* expert_ids = nullptr;

    void allocate(int V, int H, int I, int AH, int KVH, int VH,
                  int experts, int top_k) {
        const int AS = AH * AD, KVW = KVH * AD;
        const int DO = VH * VD, DQKV = 2 * DQK + DO;
        const size_t count = 2 * H + V + 2 * I + DQKV + DO + 2 * VH +
                             3 * DO + 5 * AS + 2 * KVW + 2ull * experts +
                             static_cast<size_t>(top_k) * (3 * I + 1);
        CUDA_OK(cudaMalloc(&storage, count * sizeof(float)));
        allocated_bytes = count * sizeof(float) +
                          static_cast<size_t>(top_k) * sizeof(int);
        float* cursor = storage;
        hidden = cursor; cursor += H;
        normalized = cursor; cursor += H;
        logits = cursor; cursor += V;
        ffn_gate = cursor; cursor += I;
        ffn_up = cursor; cursor += I;
        delta_qkv = cursor; cursor += DQKV;
        delta_z = cursor; cursor += DO;
        delta_a = cursor; cursor += VH;
        delta_b = cursor; cursor += VH;
        delta_q = cursor; cursor += DO;
        delta_k = cursor; cursor += DO;
        delta_output = cursor; cursor += DO;
        query_and_gate = cursor; cursor += 2 * AS;
        query = cursor; cursor += AS;
        attention_gate = cursor; cursor += AS;
        key = cursor; cursor += KVW;
        value = cursor; cursor += KVW;
        attention_output = cursor; cursor += AS;
        router_logits = cursor; cursor += experts;
        router_probabilities = cursor; cursor += experts;
        expert_gate_up = cursor; cursor += static_cast<size_t>(top_k) * 2 * I;
        expert_hidden = cursor; cursor += static_cast<size_t>(top_k) * I;
        expert_weights = cursor; cursor += top_k;
        if (top_k) CUDA_OK(cudaMalloc(&expert_ids, top_k * sizeof(int)));
        Q3X_ASSERT(cursor == storage + count,
                   "CUDA Work layout cursor=%p end=%p",
                   static_cast<void*>(cursor),
                   static_cast<void*>(storage + count));
    }

    void release() {
        if (storage) cudaFree(storage);
        if (expert_ids) cudaFree(expert_ids);
        storage = nullptr;
        expert_ids = nullptr;
        allocated_bytes = 0;
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
    BF16* converted = nullptr;

    void allocate(int H, int I, int AH, int KVH, int VH,
                  bool needs_bf16_conversion) {
        const int AS = AH * AD, KVW = KVH * AD;
        const int DO = VH * VD, DQKV = 2 * DQK + DO;
        const size_t stride = 2 * H + DQKV + DO + 2 * VH + 3 * DO +
                              5 * AS + 2 * KVW + 2 * I;
        CUDA_OK(cudaMalloc(&storage,
                           static_cast<size_t>(PREFILL_CHUNK) * stride *
                           sizeof(float)));
        if (needs_bf16_conversion) {
            const int max_input = I > DO ? (I > AS ? I : AS) :
                                          (DO > AS ? DO : AS);
            CUDA_OK(cudaMalloc(&converted,
                               static_cast<size_t>(PREFILL_CHUNK) * max_input *
                               sizeof(BF16)));
        }
        float* cursor = storage;
        hidden = cursor; cursor += static_cast<size_t>(PREFILL_CHUNK) * H;
        normalized = cursor; cursor += static_cast<size_t>(PREFILL_CHUNK) * H;
        delta_qkv = cursor; cursor += static_cast<size_t>(PREFILL_CHUNK) * DQKV;
        delta_z = cursor; cursor += static_cast<size_t>(PREFILL_CHUNK) * DO;
        delta_a = cursor; cursor += static_cast<size_t>(PREFILL_CHUNK) * VH;
        delta_b = cursor; cursor += static_cast<size_t>(PREFILL_CHUNK) * VH;
        delta_q = cursor; cursor += static_cast<size_t>(PREFILL_CHUNK) * DO;
        delta_k = cursor; cursor += static_cast<size_t>(PREFILL_CHUNK) * DO;
        delta_output = cursor; cursor += static_cast<size_t>(PREFILL_CHUNK) * DO;
        query_and_gate = cursor; cursor += static_cast<size_t>(PREFILL_CHUNK) * 2 * AS;
        query = cursor; cursor += static_cast<size_t>(PREFILL_CHUNK) * AS;
        attention_gate = cursor; cursor += static_cast<size_t>(PREFILL_CHUNK) * AS;
        key = cursor; cursor += static_cast<size_t>(PREFILL_CHUNK) * KVW;
        value = cursor; cursor += static_cast<size_t>(PREFILL_CHUNK) * KVW;
        attention_output = cursor; cursor += static_cast<size_t>(PREFILL_CHUNK) * AS;
        ffn_gate = cursor; cursor += static_cast<size_t>(PREFILL_CHUNK) * I;
        ffn_up = cursor; cursor += static_cast<size_t>(PREFILL_CHUNK) * I;
        Q3X_ASSERT(cursor == storage + PREFILL_CHUNK * stride,
                   "CUDA BatchWork layout cursor=%p end=%p",
                   static_cast<void*>(cursor),
                   static_cast<void*>(storage + PREFILL_CHUNK * stride));
    }

    void release() {
        if (storage) cudaFree(storage);
        if (converted) cudaFree(converted);
        storage = nullptr;
        converted = nullptr;
    }
};

struct StateData {
    const q3x_model::ModelConfig* config = nullptr;
    int position = 0;
    int capacity = 0;
    int checkpoint_position = 0;
    std::unique_ptr<LayerState[]> layer;
    DeviceFloatStorage recurrent;
    DeviceFloatStorage kv_cache;
    DeviceFloatStorage checkpoint;
    float* checkpoint_recurrent = nullptr;
    float* checkpoint_logits = nullptr;
    int* device_tokens = nullptr;
    int* device_position = nullptr;
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas = nullptr;
    cudaGraphExec_t forward_graph = nullptr;
    cudaGraphExec_t logits_graph = nullptr;
    Work work;
    BatchWork batch;
    std::unique_ptr<float[]> host_logits;

    StateData(const q3x_model::ModelConfig& c, int context_size)
        : config(&c), capacity(context_size),
          layer(new (std::nothrow) LayerState[c.N]),
          host_logits(new (std::nothrow) float[c.V]) {
        const int H = c.H, I = c.I, N = c.N, AI = c.AI;
        const int AH = c.AH, KVH = c.KVH, VH = c.VH;
        Q3X_ASSERT(layer && host_logits,
                   "CUDA State allocation failed layers=%d logits=%d", c.N, c.V);
        Q3X_ASSERT(c.AD == AD && c.RD == RD && c.KH == KH && c.KD == KD &&
                   c.VD == VD && c.CK == CK,
                   "CUDA fixed dimensions model=%s", c.name);
        const int KVW = KVH * AD;
        const int DO = VH * VD, DQKV = 2 * DQK + DO;
        const int delta_layers = N - N / AI;
        const int attention_layers = N / AI;
        const size_t conv_count = static_cast<size_t>(DQKV) * (CK - 1);
        const size_t memory_count = static_cast<size_t>(VH) * KD * VD;
        const size_t cache_count = static_cast<size_t>(capacity) * KVW;
        const size_t recurrent_count = static_cast<size_t>(delta_layers) *
                                       (conv_count + memory_count);
        CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUBLAS_OK(cublasCreate(&cublas));
        CUBLAS_OK(cublasSetStream(cublas, stream));
        work.allocate(c.V, H, I, AH, KVH, VH, c.experts, c.top_k);
        if (!c.experts) {
            batch.allocate(H, I, AH, KVH, VH,
                           c.matrix_type == q3x_model::MATRIX_BF16 &&
                           c.id == q3x_model::QWEN35_4B.id);
        }
        recurrent.allocate(recurrent_count);
        kv_cache.allocate(static_cast<size_t>(attention_layers) * 2 * cache_count);
        checkpoint.allocate(recurrent_count + c.V);
        checkpoint_recurrent = checkpoint.take(recurrent_count);
        checkpoint_logits = checkpoint.take(c.V);
        checkpoint.finish();
        CUDA_OK(cudaMalloc(&device_tokens,
                           static_cast<size_t>(capacity) * sizeof(int)));
        CUDA_OK(cudaMalloc(&device_position, sizeof(int)));
        CUDA_OK(cudaMemsetAsync(device_position, 0, sizeof(int), stream));
        for (int index = 0; index < N; ++index) {
            if (index % AI != AI - 1) {
                layer[index].conv = recurrent.take(conv_count);
                layer[index].memory = recurrent.take(memory_count);
            } else {
                layer[index].key_cache = kv_cache.take(cache_count);
                layer[index].value_cache = kv_cache.take(cache_count);
            }
        }
        recurrent.finish();
        kv_cache.finish();
        CUDA_OK(cudaMemsetAsync(recurrent.memory, 0,
                                recurrent.count * sizeof(float), stream));
        if (c.experts) {
            const size_t recurrent_bytes = recurrent.count * sizeof(float);
            const size_t kv_bytes = kv_cache.count * sizeof(float);
            const size_t checkpoint_bytes = checkpoint.count * sizeof(float);
            const size_t token_bytes = static_cast<size_t>(capacity) * sizeof(int);
            const size_t state_bytes = work.allocated_bytes + recurrent_bytes +
                                       kv_bytes + checkpoint_bytes + token_bytes +
                                       sizeof(int);
            LOG_INFO("CUDA Qwen3.6 state context=%d state_bytes=%zu "
                     "work_bytes=%zu recurrent_bytes=%zu kv_bytes=%zu "
                     "checkpoint_bytes=%zu",
                     capacity, state_bytes, work.allocated_bytes,
                     recurrent_bytes, kv_bytes, checkpoint_bytes);
        }
    }

    ~StateData() {
        if (forward_graph) cudaGraphExecDestroy(forward_graph);
        if (logits_graph) cudaGraphExecDestroy(logits_graph);
        work.release();
        batch.release();
        recurrent.release();
        kv_cache.release();
        checkpoint.release();
        if (device_tokens) cudaFree(device_tokens);
        if (device_position) cudaFree(device_position);
        if (cublas) cublasDestroy(cublas);
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

__device__ float f16(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 113;
            while (!(mantissa & 0x0400u)) { mantissa <<= 1; --exponent; }
            bits = sign | (exponent << 23) | ((mantissa & 0x03ffu) << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
    }
    return __uint_as_float(bits);
}

__global__ void fp32_to_bf16_kernel(const float* input, BF16* output,
                                    int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        reinterpret_cast<__nv_bfloat16*>(output)[index] =
            __float2bfloat16_rn(input[index]);
    }
}

__global__ void embed_kernel(const BF16* table, const int* tokens,
                             const int* position, float* output, int H) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < H) {
        const int token = tokens[*position];
        output[index] = f32(table[static_cast<size_t>(token) * H + index]);
    }
}

__global__ void embed_q8_kernel(const q3x_q8::Block* table,
                                const int* tokens, const int* position,
                                float* output, int H) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < H) {
        const int token = tokens[*position];
        const int blocks = H / q3x_q8::BLOCK_SIZE;
        const q3x_q8::Block& block = table[
            static_cast<size_t>(token) * blocks + index / q3x_q8::BLOCK_SIZE];
        output[index] = f16(block.scale) * block.values[index % q3x_q8::BLOCK_SIZE];
    }
}

__global__ void embed_q4_kernel(const q3x_q4::Block* table,
                                const int* tokens, const int* position,
                                float* output, int H) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < H) {
        const int token = tokens[*position];
        const int blocks = H / q3x_q4::BLOCK_SIZE;
        const q3x_q4::Block& block = table[
            static_cast<size_t>(token) * blocks + index / q3x_q4::BLOCK_SIZE];
        const int within = index % q3x_q4::BLOCK_SIZE;
        const uint8_t packed = block.values[within % 16];
        const int quant = within < 16 ? packed & 0x0f : packed >> 4;
        output[index] = f16(block.scale) * (quant - 8);
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

__device__ float dot_row_q8(const q3x_q8::Block* weight, int row,
                            const float* input, int columns, float* shared) {
    const int blocks = columns / q3x_q8::BLOCK_SIZE;
    const q3x_q8::Block* source = weight + static_cast<size_t>(row) * blocks;
    float sum = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const q3x_q8::Block& block = source[column / q3x_q8::BLOCK_SIZE];
        sum += f16(block.scale) * block.values[column % q3x_q8::BLOCK_SIZE] *
               input[column];
    }
    return block_sum(sum, shared);
}

__device__ float dot_row_q4(const q3x_q4::Block* weight, int row,
                            const float* input, int columns, float* shared) {
    const int blocks = columns / q3x_q4::BLOCK_SIZE;
    const q3x_q4::Block* source = weight + static_cast<size_t>(row) * blocks;
    float sum = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const q3x_q4::Block& block = source[column / q3x_q4::BLOCK_SIZE];
        const int within = column % q3x_q4::BLOCK_SIZE;
        const uint8_t packed = block.values[within % 16];
        const int quant = within < 16 ? packed & 0x0f : packed >> 4;
        sum += f16(block.scale) * (quant - 8) * input[column];
    }
    return block_sum(sum, shared);
}

__global__ void mv_q8_kernel(const q3x_q8::Block* weight,
                             const float* input, float* output,
                             int rows, int columns) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float shared[BLOCK];
    const float total = dot_row_q8(weight, row, input, columns, shared);
    if (threadIdx.x == 0) output[row] = total;
}

__global__ void mv_add_q8_kernel(const q3x_q8::Block* weight,
                                 const float* input, float* output,
                                 int rows, int columns) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float shared[BLOCK];
    const float total = dot_row_q8(weight, row, input, columns, shared);
    if (threadIdx.x == 0) output[row] += total;
}

__global__ void mv_q4_kernel(const q3x_q4::Block* weight,
                             const float* input, float* output,
                             int rows, int columns) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float shared[BLOCK];
    const float total = dot_row_q4(weight, row, input, columns, shared);
    if (threadIdx.x == 0) output[row] = total;
}

__global__ void mv_add_q4_kernel(const q3x_q4::Block* weight,
                                 const float* input, float* output,
                                 int rows, int columns) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float shared[BLOCK];
    const float total = dot_row_q4(weight, row, input, columns, shared);
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
    const float* input, float* qkv, float* z, float* a, float* b,
    int H, int DQKV, int DO, int VH) {
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
    const float* input, float* query_and_gate, float* key, float* value,
    int H, int AS, int KVW) {
    const int block = blockIdx.x;
    int row = block;
    const BF16* weight = q_weight;
    float* output = query_and_gate;
    if (block >= 2 * AS + KVW) {
        row = block - 2 * AS - KVW;
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
    const float* input, float* gate, float* up, int H, int I) {
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
                                   int start, int count, float* output, int H) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = count * H;
    if (index < total) {
        const int token_index = index / H;
        const int hidden_index = index % H;
        const int token = tokens[start + token_index];
        output[index] = f32(table[static_cast<size_t>(token) * H + hidden_index]);
    }
}

__global__ void batch_embed_q8_kernel(const q3x_q8::Block* table,
                                      const int* tokens, int start, int count,
                                      float* output, int H) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count * H) {
        const int token_index = index / H;
        const int hidden_index = index % H;
        const int token = tokens[start + token_index];
        const int blocks = H / q3x_q8::BLOCK_SIZE;
        const q3x_q8::Block& block = table[
            static_cast<size_t>(token) * blocks +
            hidden_index / q3x_q8::BLOCK_SIZE];
        output[index] = f16(block.scale) *
                        block.values[hidden_index % q3x_q8::BLOCK_SIZE];
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

constexpr int Q8_THREADS_X = 16;
constexpr int Q8_THREADS_Y = 16;
constexpr int Q8_ROWS_PER_THREAD = 4;
constexpr int Q8_TOKENS_PER_THREAD = 8;
constexpr int Q8_ROW_TILE = Q8_THREADS_X * Q8_ROWS_PER_THREAD;
constexpr int Q8_TOKEN_TILE = Q8_THREADS_Y * Q8_TOKENS_PER_THREAD;
constexpr int Q8_COLUMN_TILE = 32;

// One CTA computes output[128 tokens, 64 rows]. Each Q8 block is unpacked once
// into shared memory, then reused by every token in the tile.
__global__ void batch_mv_q8_kernel(const q3x_q8::Block* weight,
                                   const float* input, float* output,
                                   int rows, int columns, int count, bool add) {
    const int blocks = columns / q3x_q8::BLOCK_SIZE;
    __shared__ float weight_tile[Q8_ROW_TILE][Q8_COLUMN_TILE + 1];
    __shared__ float input_tile[Q8_TOKEN_TILE][Q8_COLUMN_TILE + 1];
    __shared__ float scale_tile[Q8_ROW_TILE];
    const int thread = threadIdx.y * Q8_THREADS_X + threadIdx.x;
    constexpr int threads = Q8_THREADS_X * Q8_THREADS_Y;
    float sum[Q8_TOKENS_PER_THREAD][Q8_ROWS_PER_THREAD] = {};
    for (int column = 0; column < columns; column += Q8_COLUMN_TILE) {
        for (int index = thread; index < Q8_ROW_TILE * Q8_COLUMN_TILE;
             index += threads) {
            const int tile_row = index / Q8_COLUMN_TILE;
            const int tile_column = index % Q8_COLUMN_TILE;
            const int source_row = blockIdx.x * Q8_ROW_TILE + tile_row;
            float value = 0.0f;
            if (source_row < rows) {
                const q3x_q8::Block& block = weight[
                    static_cast<size_t>(source_row) * blocks +
                    (column + tile_column) / q3x_q8::BLOCK_SIZE];
                value = block.values[tile_column % q3x_q8::BLOCK_SIZE];
            }
            weight_tile[tile_row][tile_column] = value;
        }
        for (int tile_row = thread; tile_row < Q8_ROW_TILE;
             tile_row += threads) {
            const int source_row = blockIdx.x * Q8_ROW_TILE + tile_row;
            scale_tile[tile_row] = source_row < rows
                ? f16(weight[static_cast<size_t>(source_row) * blocks +
                             column / q3x_q8::BLOCK_SIZE].scale)
                : 0.0f;
        }
        for (int index = thread; index < Q8_TOKEN_TILE * Q8_COLUMN_TILE;
             index += threads) {
            const int tile_token = index / Q8_COLUMN_TILE;
            const int tile_column = index % Q8_COLUMN_TILE;
            const int source_token = blockIdx.y * Q8_TOKEN_TILE + tile_token;
            input_tile[tile_token][tile_column] = source_token < count
                ? input[static_cast<size_t>(source_token) * columns +
                        column + tile_column]
                : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int token_part = 0; token_part < Q8_TOKENS_PER_THREAD;
             ++token_part) {
#pragma unroll
            for (int row_part = 0; row_part < Q8_ROWS_PER_THREAD;
                 ++row_part) {
                const int tile_token = threadIdx.y + token_part * Q8_THREADS_Y;
                const int tile_row = threadIdx.x + row_part * Q8_THREADS_X;
                float inner = 0.0f;
#pragma unroll
                for (int index = 0; index < Q8_COLUMN_TILE; ++index) {
                    inner += weight_tile[tile_row][index] *
                             input_tile[tile_token][index];
                }
                sum[token_part][row_part] += scale_tile[tile_row] * inner;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int token_part = 0; token_part < Q8_TOKENS_PER_THREAD; ++token_part) {
#pragma unroll
        for (int row_part = 0; row_part < Q8_ROWS_PER_THREAD; ++row_part) {
            const int row = blockIdx.x * Q8_ROW_TILE + threadIdx.x +
                            row_part * Q8_THREADS_X;
            const int token = blockIdx.y * Q8_TOKEN_TILE + threadIdx.y +
                              token_part * Q8_THREADS_Y;
            if (row < rows && token < count) {
                float& destination = output[static_cast<size_t>(token) * rows + row];
                if (add) destination += sum[token_part][row_part];
                else destination = sum[token_part][row_part];
            }
        }
    }
}

__global__ void batch_delta_projections_kernel(
    const BF16* qkv_weight, const BF16* z_weight,
    const BF16* a_weight, const BF16* b_weight,
    const float* input, int count,
    float* qkv, float* z, float* a, float* b,
    int H, int DQKV, int DO, int VH) {
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
    float* query_and_gate, float* key, float* value,
    int H, int AS, int KVW) {
    const int token = blockIdx.y;
    if (token >= count) return;
    const int block = blockIdx.x;
    int row = block;
    int width = 2 * AS;
    const BF16* weight = q_weight;
    float* output = query_and_gate;
    if (block >= 2 * AS + KVW) {
        row = block - 2 * AS - KVW;
        width = KVW;
        weight = v_weight;
        output = value;
    } else if (block >= 2 * AS) {
        row = block - 2 * AS;
        width = KVW;
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
    const float* input, int count, float* gate, float* up, int H, int I) {
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

__global__ void batch_swiglu_kernel(float* gate, const float* up,
                                    int count, int I) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count * I) {
        gate[index] = gate[index] / (1.0f + expf(-gate[index])) * up[index];
    }
}

__global__ void batch_conv_kernel(float* qkv, const BF16* weight,
                                  float* history, int count, int DQKV) {
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
    const float* qkv, int count, float* query, float* key,
    int DQKV, int VH) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    if (token >= count) return;
    const int index = threadIdx.x;
    const float* source = qkv + static_cast<size_t>(token) * DQKV;
    const int qk_head = head / (VH / KH);
    float q = index < KD ? source[qk_head * KD + index] : 0.0f;
    float k = index < KD ? source[DQK + qk_head * KD + index] : 0.0f;
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
    int count, float* memory, float* output, int DQKV, int VH) {
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
                                       const float* gate, int count, int VH) {
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
    float* query, float* gate, int AH, int AS) {
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
    float* key, const BF16* norm, int start, int count, int KVH) {
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
    const float* key, const float* value, int KVW) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count * KVW) {
        const int token = index / KVW;
        const int channel = index % KVW;
        const size_t cache = static_cast<size_t>(start + token) * KVW + channel;
        key_cache[cache] = key[index];
        value_cache[cache] = value[index];
    }
}

__global__ void batch_attention_kernel(
    const float* query, const float* gate,
    const float* key_cache, const float* value_cache,
    int start, int count, float* output, int AH, int KVH) {
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

__global__ void swiglu_kernel(float* gate, const float* up, int I) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < I) gate[index] = gate[index] / (1.0f + expf(-gate[index])) * up[index];
}

__global__ void router_top_k_kernel(const float* logits, float* probabilities,
                                    int* ids, float* weights,
                                    int experts, int top_k) {
    if (blockIdx.x || threadIdx.x) return;
    float maximum = -3.402823466e+38F;
    for (int expert = 0; expert < experts; ++expert)
        maximum = fmaxf(maximum, logits[expert]);
    float total = 0.0f;
    for (int expert = 0; expert < experts; ++expert) {
        probabilities[expert] = expf(logits[expert] - maximum);
        total += probabilities[expert];
    }
    for (int expert = 0; expert < experts; ++expert)
        probabilities[expert] /= total;
    float selected_total = 0.0f;
    for (int slot = 0; slot < top_k; ++slot) {
        int selected = -1;
        for (int expert = 0; expert < experts; ++expert) {
            bool used = false;
            for (int previous = 0; previous < slot; ++previous)
                used = used || ids[previous] == expert;
            if (!used && (selected < 0 ||
                          probabilities[expert] > probabilities[selected]))
                selected = expert;
        }
        ids[slot] = selected;
        selected_total += probabilities[selected];
    }
    for (int slot = 0; slot < top_k; ++slot) {
        weights[slot] = probabilities[ids[slot]] / selected_total;
    }
}

__global__ void expert_gate_up_q4_kernel(
    const q3x_q4::Block* packed, const float* input, float* output,
    const int* ids, int experts, int rows, int columns, int top_k) {
    const int output_row = blockIdx.x;
    const int slot = output_row / rows;
    const int row = output_row % rows;
    if (slot >= top_k) return;
    const int source_expert = ids[slot];
    if (source_expert >= experts) return;
    const int blocks = columns / q3x_q4::BLOCK_SIZE;
    const q3x_q4::Block* weight = packed +
        (static_cast<size_t>(source_expert) * rows + row) * blocks;
    float sum = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const q3x_q4::Block& block = weight[column / q3x_q4::BLOCK_SIZE];
        const int within = column % q3x_q4::BLOCK_SIZE;
        const uint8_t byte = block.values[within % 16];
        const int quant = within < 16 ? byte & 0x0f : byte >> 4;
        sum += f16(block.scale) * (quant - 8) * input[column];
    }
    __shared__ float shared[BLOCK];
    const float total = block_sum(sum, shared);
    if (threadIdx.x == 0) output[static_cast<size_t>(slot) * rows + row] = total;
}

__global__ void expert_swiglu_kernel(const float* gate_up, float* hidden,
                                     int intermediate, int top_k) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= top_k * intermediate) return;
    const int slot = index / intermediate;
    const int channel = index % intermediate;
    const float gate = gate_up[static_cast<size_t>(slot) * 2 * intermediate + channel];
    const float up = gate_up[static_cast<size_t>(slot) * 2 * intermediate +
                             intermediate + channel];
    hidden[index] = gate / (1.0f + expf(-gate)) * up;
}

__global__ void expert_down_weighted_q4_kernel(
    const q3x_q4::Block* packed, const float* hidden,
    const int* ids, const float* routing_weights, float* output,
    int experts, int rows, int columns, int top_k) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int blocks = columns / q3x_q4::BLOCK_SIZE;
    float sum = 0.0f;
    for (int slot = 0; slot < top_k; ++slot) {
        const int source_expert = ids[slot];
        if (source_expert >= experts) continue;
        const q3x_q4::Block* weight = packed +
            (static_cast<size_t>(source_expert) * rows + row) * blocks;
        float expert_sum = 0.0f;
        for (int column = threadIdx.x; column < columns; column += blockDim.x) {
            const q3x_q4::Block& block = weight[column / q3x_q4::BLOCK_SIZE];
            const int within = column % q3x_q4::BLOCK_SIZE;
            const uint8_t byte = block.values[within % 16];
            const int quant = within < 16 ? byte & 0x0f : byte >> 4;
            expert_sum += f16(block.scale) * (quant - 8) *
                          hidden[static_cast<size_t>(slot) * columns + column];
        }
        sum += routing_weights[slot] * expert_sum;
    }
    __shared__ float shared[BLOCK];
    const float total = block_sum(sum, shared);
    if (threadIdx.x == 0) output[row] += total;
}

__global__ void shared_scaled_add_kernel(float* output, const float* shared,
                                         const float* gate, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const float scale = 1.0f / (1.0f + expf(-gate[0]));
        output[index] += scale * shared[index];
    }
}

__global__ void conv_kernel(float* qkv, const BF16* weight, float* history,
                            int DQKV) {
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
                                        float* key, int VH) {
    const int head = blockIdx.x;
    const int index = threadIdx.x;
    const int qk_head = head / (VH / KH);
    float q = index < KD ? qkv[qk_head * KD + index] : 0.0f;
    float k = index < KD ? qkv[DQK + qk_head * KD + index] : 0.0f;
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
                                     float* query, float* gate, int AS) {
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
                                const float* value, int KVW) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < KVW) {
        const size_t offset = static_cast<size_t>(*device_position) * KVW + index;
        key_cache[offset] = key[index];
        value_cache[offset] = value[index];
    }
}

__global__ void attention_kernel(const float* query, const float* gate,
                                 const float* key_cache,
                                 const float* value_cache,
                                 const int* device_position,
                                 float* output, int AH, int KVH) {
    const int head = blockIdx.x;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int kv_head = head / (AH / KVH);
    const int position = *device_position;
    float q[AD / 32];
    float accumulator[AD / 32] {};
#pragma unroll
    for (int part = 0; part < AD / 32; ++part) {
        q[part] = query[head * AD + part * 32 + lane];
    }
    float maximum = -3.402823466e+38F;
    float denominator = 0.0f;
    for (int token = warp; token <= position; token += BLOCK / 32) {
        const size_t cache = (static_cast<size_t>(token) * KVH + kv_head) * AD;
        float score = 0.0f;
#pragma unroll
        for (int part = 0; part < AD / 32; ++part) {
            score += q[part] * key_cache[cache + part * 32 + lane];
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            score += __shfl_down_sync(0xffffffff, score, offset);
        }
        score *= rsqrtf(static_cast<float>(AD));
        float alpha = 0.0f, beta = 0.0f;
        if (lane == 0) {
            const float next_maximum = fmaxf(maximum, score);
            alpha = expf(maximum - next_maximum);
            beta = expf(score - next_maximum);
            denominator = denominator * alpha + beta;
            maximum = next_maximum;
        }
        alpha = __shfl_sync(0xffffffff, alpha, 0);
        beta = __shfl_sync(0xffffffff, beta, 0);
#pragma unroll
        for (int part = 0; part < AD / 32; ++part) {
            const int index = part * 32 + lane;
            accumulator[part] = accumulator[part] * alpha +
                                beta * value_cache[cache + index];
        }
    }

    __shared__ float warp_maximum[BLOCK / 32];
    __shared__ float warp_denominator[BLOCK / 32];
    __shared__ float warp_accumulator[BLOCK / 32][AD];
    if (lane == 0) {
        warp_maximum[warp] = maximum;
        warp_denominator[warp] = denominator;
    }
#pragma unroll
    for (int part = 0; part < AD / 32; ++part) {
        warp_accumulator[warp][part * 32 + lane] = accumulator[part];
    }
    __syncthreads();

    const int index = threadIdx.x;
    float combined_maximum = -3.402823466e+38F;
    for (int source = 0; source < BLOCK / 32; ++source) {
        if (warp_denominator[source] > 0.0f) {
            combined_maximum = fmaxf(combined_maximum, warp_maximum[source]);
        }
    }
    float combined_denominator = 0.0f;
    float combined_accumulator = 0.0f;
    for (int source = 0; source < BLOCK / 32; ++source) {
        if (warp_denominator[source] > 0.0f) {
            const float scale = expf(warp_maximum[source] - combined_maximum);
            combined_denominator += warp_denominator[source] * scale;
            combined_accumulator += warp_accumulator[source][index] * scale;
        }
    }
    const float z = gate[head * AD + index];
    output[head * AD + index] = combined_accumulator / combined_denominator /
                                (1.0f + expf(-z));
}

__global__ void advance_position_kernel(int* position) {
    if (threadIdx.x == 0) ++*position;
}

const BF16* bf16_weight(const Linear& linear) {
    return static_cast<const BF16*>(linear.weight);
}

const q3x_q8::Block* q8_weight(const Linear& linear) {
    return static_cast<const q3x_q8::Block*>(linear.weight);
}

const q3x_q4::Block* q4_weight(const Linear& linear) {
    return static_cast<const q3x_q4::Block*>(linear.weight);
}

void embed(const Linear& table, const int* tokens, const int* position,
           float* output, cudaStream_t stream) {
    const int blocks = (table.columns + BLOCK - 1) / BLOCK;
    if (table.type == q3x_model::MATRIX_BF16) {
        embed_kernel<<<blocks, BLOCK, 0, stream>>>(
            bf16_weight(table), tokens, position, output, table.columns);
    } else if (table.type == q3x_model::MATRIX_Q8_0) {
        embed_q8_kernel<<<blocks, BLOCK, 0, stream>>>(
            q8_weight(table), tokens, position, output, table.columns);
    } else if (table.type == q3x_model::MATRIX_Q4_0) {
        embed_q4_kernel<<<blocks, BLOCK, 0, stream>>>(
            q4_weight(table), tokens, position, output, table.columns);
    } else {
        Q3X_ASSERT(false, "CUDA unknown embedding type=%u", table.type);
    }
}

void batch_embed(const Linear& table, const int* tokens, int start, int count,
                 float* output, cudaStream_t stream) {
    const int values = count * table.columns;
    const int blocks = (values + BLOCK - 1) / BLOCK;
    if (table.type == q3x_model::MATRIX_BF16) {
        batch_embed_kernel<<<blocks, BLOCK, 0, stream>>>(
            bf16_weight(table), tokens, start, count, output, table.columns);
    } else if (table.type == q3x_model::MATRIX_Q8_0) {
        batch_embed_q8_kernel<<<blocks, BLOCK, 0, stream>>>(
            q8_weight(table), tokens, start, count, output, table.columns);
    } else {
        Q3X_ASSERT(false, "CUDA batch embedding unsupported type=%u", table.type);
    }
}

void mv(const Linear& linear, const float* input, float* output,
        cudaStream_t stream) {
    if (linear.type == q3x_model::MATRIX_BF16) {
        mv_kernel<<<linear.rows, BLOCK, 0, stream>>>(
            bf16_weight(linear), input, output, linear.rows, linear.columns);
    } else if (linear.type == q3x_model::MATRIX_Q8_0) {
        mv_q8_kernel<<<linear.rows, BLOCK, 0, stream>>>(
            q8_weight(linear), input, output, linear.rows, linear.columns);
    } else if (linear.type == q3x_model::MATRIX_Q4_0) {
        mv_q4_kernel<<<linear.rows, BLOCK, 0, stream>>>(
            q4_weight(linear), input, output, linear.rows, linear.columns);
    } else {
        Q3X_ASSERT(false, "CUDA unknown matrix type=%u", linear.type);
    }
}

void mv_add(const Linear& linear, const float* input, float* output,
            cudaStream_t stream) {
    if (linear.type == q3x_model::MATRIX_BF16) {
        mv_add_kernel<<<linear.rows, BLOCK, 0, stream>>>(
            bf16_weight(linear), input, output, linear.rows, linear.columns);
    } else if (linear.type == q3x_model::MATRIX_Q8_0) {
        mv_add_q8_kernel<<<linear.rows, BLOCK, 0, stream>>>(
            q8_weight(linear), input, output, linear.rows, linear.columns);
    } else if (linear.type == q3x_model::MATRIX_Q4_0) {
        mv_add_q4_kernel<<<linear.rows, BLOCK, 0, stream>>>(
            q4_weight(linear), input, output, linear.rows, linear.columns);
    } else {
        Q3X_ASSERT(false, "CUDA unknown add matrix type=%u", linear.type);
    }
}

// Row-major W[rows,columns] and input[count,columns] are viewed as column-major
// W^T and input^T. cuBLAS then writes column-major [rows,count], which is the
// same bytes as the runtime's row-major output[count,rows]. Activations are
// rounded to BF16 only for batch prefill so Tensor Cores can execute the GEMM;
// decode keeps the readable FP32-activation GEMV path above.
void batch_mv(cublasHandle_t handle, const Linear& linear,
              const float* input, float* output, int count, bool add,
              BF16* converted, cudaStream_t stream) {
    if (linear.type == q3x_model::MATRIX_Q8_0) {
        const dim3 grid((linear.rows + Q8_ROW_TILE - 1) / Q8_ROW_TILE,
                        (count + Q8_TOKEN_TILE - 1) / Q8_TOKEN_TILE);
        const dim3 threads(Q8_THREADS_X, Q8_THREADS_Y);
        batch_mv_q8_kernel<<<grid, threads, 0, stream>>>(
            q8_weight(linear), input, output, linear.rows, linear.columns,
            count, add);
        return;
    }
    Q3X_ASSERT(linear.type == q3x_model::MATRIX_BF16,
               "CUDA batch matrix unsupported type=%u", linear.type);
    Q3X_ASSERT(converted, "BF16 batch matrix conversion buffer is null");
    const int values = count * linear.columns;
    fp32_to_bf16_kernel<<<(values + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        input, converted, values);
    const float alpha = 1.0f;
    const float beta = add ? 1.0f : 0.0f;
    CUBLAS_OK(cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        linear.rows, count, linear.columns,
        &alpha,
        bf16_weight(linear), CUDA_R_16BF, linear.columns,
        converted, CUDA_R_16BF, linear.columns,
        &beta,
        output, CUDA_R_32F, linear.rows,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void delta_projections_q8(const DeltaWeights& weights, const float* input,
                          Work& work, cudaStream_t stream) {
    mv(weights.qkv, input, work.delta_qkv, stream);
    mv(weights.z, input, work.delta_z, stream);
    mv(weights.a, input, work.delta_a, stream);
    mv(weights.b, input, work.delta_b, stream);
}

void attention_projections_q8(const AttentionWeights& weights,
                              const float* input, Work& work,
                              cudaStream_t stream) {
    mv(weights.q, input, work.query_and_gate, stream);
    mv(weights.k, input, work.key, stream);
    mv(weights.v, input, work.value, stream);
}

void ffn_projections_q8(const Layer& layer, const float* input, Work& work,
                        cudaStream_t stream) {
    mv(layer.gate, input, work.ffn_gate, stream);
    mv(layer.up, input, work.ffn_up, stream);
}

void delta_projections(const DeltaWeights& weights, const float* input,
                       Work& work, int DQKV, int DO, int VH,
                       cudaStream_t stream) {
    if (weights.qkv.type == q3x_model::MATRIX_BF16) {
        delta_projections_kernel<<<DQKV + DO + 2 * VH, BLOCK, 0, stream>>>(
            bf16_weight(weights.qkv), bf16_weight(weights.z),
            bf16_weight(weights.a), bf16_weight(weights.b), input,
            work.delta_qkv, work.delta_z, work.delta_a, work.delta_b,
            weights.qkv.columns, DQKV, DO, VH);
    } else {
        delta_projections_q8(weights, input, work, stream);
    }
}

void attention_projections(const AttentionWeights& weights,
                           const float* input, Work& work, int AS, int KVW,
                           cudaStream_t stream) {
    if (weights.q.type == q3x_model::MATRIX_BF16) {
        attention_projections_kernel<<<2 * AS + 2 * KVW, BLOCK, 0, stream>>>(
            bf16_weight(weights.q), bf16_weight(weights.k),
            bf16_weight(weights.v), input, work.query_and_gate,
            work.key, work.value, weights.q.columns, AS, KVW);
    } else {
        attention_projections_q8(weights, input, work, stream);
    }
}

void ffn_projections(const Layer& layer, const float* input, Work& work,
                     int I, cudaStream_t stream) {
    if (layer.gate.type == q3x_model::MATRIX_BF16) {
        ffn_projections_kernel<<<2 * I, BLOCK, 0, stream>>>(
            bf16_weight(layer.gate), bf16_weight(layer.up), input,
            work.ffn_gate, work.ffn_up, layer.gate.columns, I);
    } else {
        ffn_projections_q8(layer, input, work, stream);
    }
}

void rms(const float* input, const BF16* weight, int count, float* output,
         cudaStream_t stream) {
    rms_kernel<<<1, BLOCK, 0, stream>>>(input, weight, output, count);
}

void deltanet(const DeltaWeights& weights, float* conv, float* memory,
              const float* input, Work& work, float* output,
              int VH, cudaStream_t stream) {
    const int DO = VH * VD, DQKV = 2 * DQK + DO;
    delta_projections(weights, input, work, DQKV, DO, VH, stream);
    conv_kernel<<<(DQKV + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        work.delta_qkv, weights.conv, conv, DQKV);
    prepare_delta_qk_kernel<<<VH, BLOCK, 0, stream>>>(
        work.delta_qkv, work.delta_q, work.delta_k, VH);
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
               Work& work, float* output, int AH, int KVH,
               cudaStream_t stream) {
    const int AS = AH * AD, KVW = KVH * AD;
    attention_projections(weights, input, work, AS, KVW, stream);
    prepare_query_kernel<<<AH, BLOCK, 0, stream>>>(
        work.query_and_gate, weights.qnorm, position, work.query,
        work.attention_gate, AS);
    prepare_key_kernel<<<KVH, BLOCK, 0, stream>>>(
        work.key, weights.knorm, position);
    store_kv_kernel<<<(KVW + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        key_cache, value_cache, position, work.key, work.value, KVW);
    attention_kernel<<<AH, BLOCK, 0, stream>>>(
        work.query, work.attention_gate, key_cache, value_cache,
        position, work.attention_output, AH, KVH);
    mv_add(weights.out, work.attention_output, output, stream);
}

void ffn(const Layer& layer, const float* input, Work& work, float* output,
         int I, cudaStream_t stream) {
    ffn_projections(layer, input, work, I, stream);
    swiglu_kernel<<<(I + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        work.ffn_gate, work.ffn_up, I);
    mv_add(layer.down, work.ffn_gate, output, stream);
}

void moe(const MoeWeights& weights, const float* input, Work& work, float* output,
         const q3x_model::ModelConfig& c, cudaStream_t stream) {
    mv(weights.router, input, work.router_logits, stream);
    router_top_k_kernel<<<1, 1, 0, stream>>>(
        work.router_logits, work.router_probabilities,
        work.expert_ids, work.expert_weights, c.experts, c.top_k);

    expert_gate_up_q4_kernel<<<c.top_k * 2 * c.I, BLOCK, 0, stream>>>(
        static_cast<const q3x_q4::Block*>(weights.gate_up.weight), input,
        work.expert_gate_up, work.expert_ids, c.experts,
        2 * c.I, c.H, c.top_k);
    expert_swiglu_kernel<<<(c.top_k * c.I + BLOCK - 1) / BLOCK,
                            BLOCK, 0, stream>>>(
        work.expert_gate_up, work.expert_hidden, c.I, c.top_k);
    expert_down_weighted_q4_kernel<<<c.H, BLOCK, 0, stream>>>(
        static_cast<const q3x_q4::Block*>(weights.down.weight),
        work.expert_hidden, work.expert_ids, work.expert_weights,
        output, c.experts, c.H, c.I, c.top_k);

    mv(weights.shared_gate, input, work.ffn_gate, stream);
    mv(weights.shared_up, input, work.ffn_up, stream);
    swiglu_kernel<<<(c.shared_I + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        work.ffn_gate, work.ffn_up, c.shared_I);
    mv(weights.shared_down, work.ffn_gate, work.delta_output, stream);
    mv(weights.shared_scale, input, work.router_logits, stream);
    shared_scaled_add_kernel<<<(c.H + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        output, work.delta_output, work.router_logits, c.H);
}

void batch_deltanet_fp32(const DeltaWeights& weights,
                         float* conv, float* memory,
                         const float* input, BatchWork& work, float* hidden,
                         int count, int H, int VH, cudaStream_t stream) {
    const int DO = VH * VD, DQKV = 2 * DQK + DO;
    const dim3 projection_grid(DQKV + DO + 2 * VH, count);
    batch_delta_projections_kernel<<<projection_grid, BLOCK, 0, stream>>>(
        bf16_weight(weights.qkv), bf16_weight(weights.z),
        bf16_weight(weights.a), bf16_weight(weights.b),
        input, count, work.delta_qkv, work.delta_z,
        work.delta_a, work.delta_b, H, DQKV, DO, VH);
    batch_conv_kernel<<<(DQKV + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        work.delta_qkv, weights.conv, conv, count, DQKV);
    const dim3 head_token_grid(VH, count);
    batch_prepare_delta_qk_kernel<<<head_token_grid, BLOCK, 0, stream>>>(
        work.delta_qkv, count, work.delta_q, work.delta_k, DQKV, VH);
    batch_delta_rule_kernel<<<VH, BLOCK, 0, stream>>>(
        work.delta_q, work.delta_k, work.delta_qkv,
        work.delta_a, work.delta_b, weights.alog, weights.dt,
        count, memory, work.delta_output, DQKV, VH);
    batch_gated_rms_kernel<<<head_token_grid, BLOCK, 0, stream>>>(
        work.delta_output, weights.norm, work.delta_z, count, VH);
    const dim3 output_grid(H, count);
    batch_mv_add_kernel<<<output_grid, BLOCK, 0, stream>>>(
        bf16_weight(weights.out), work.delta_output, hidden, H, DO, count);
}

void batch_attention_fp32(const AttentionWeights& weights,
                          float* key_cache, float* value_cache,
                          int start, const float* input, BatchWork& work,
                          float* hidden, int count, int H, int AH, int KVH,
                          cudaStream_t stream) {
    const int AS = AH * AD, KVW = KVH * AD;
    const dim3 projection_grid(2 * AS + 2 * KVW, count);
    batch_attention_projections_kernel<<<projection_grid, BLOCK, 0, stream>>>(
        bf16_weight(weights.q), bf16_weight(weights.k), bf16_weight(weights.v),
        input, count, work.query_and_gate, work.key, work.value,
        H, AS, KVW);
    const dim3 query_grid(AH, count);
    const dim3 key_grid(KVH, count);
    batch_prepare_query_kernel<<<query_grid, BLOCK, 0, stream>>>(
        work.query_and_gate, weights.qnorm, start, count,
        work.query, work.attention_gate, AH, AS);
    batch_prepare_key_kernel<<<key_grid, BLOCK, 0, stream>>>(
        work.key, weights.knorm, start, count, KVH);
    batch_store_kv_kernel<<<(count * KVW + BLOCK - 1) / BLOCK,
                            BLOCK, 0, stream>>>(
        key_cache, value_cache, start, count, work.key, work.value, KVW);
    batch_attention_kernel<<<query_grid, BLOCK, 0, stream>>>(
        work.query, work.attention_gate, key_cache, value_cache,
        start, count, work.attention_output, AH, KVH);
    const dim3 output_grid(H, count);
    batch_mv_add_kernel<<<output_grid, BLOCK, 0, stream>>>(
        bf16_weight(weights.out), work.attention_output, hidden, H, AS, count);
}

void batch_ffn_fp32(const Layer& layer, const float* input, BatchWork& work,
                    float* hidden, int count, int H, int I,
                    cudaStream_t stream) {
    const dim3 projection_grid(2 * I, count);
    batch_ffn_projections_kernel<<<projection_grid, BLOCK, 0, stream>>>(
        bf16_weight(layer.gate), bf16_weight(layer.up), input, count,
        work.ffn_gate, work.ffn_up, H, I);
    batch_swiglu_kernel<<<(count * I + BLOCK - 1) / BLOCK,
                           BLOCK, 0, stream>>>(
        work.ffn_gate, work.ffn_up, count, I);
    const dim3 output_grid(H, count);
    batch_mv_add_kernel<<<output_grid, BLOCK, 0, stream>>>(
        bf16_weight(layer.down), work.ffn_gate, hidden, H, I, count);
}

void batch_deltanet_matrix(cublasHandle_t handle, const DeltaWeights& weights,
                           float* conv, float* memory,
                           const float* input, BatchWork& work, float* hidden,
                           int count, int VH, cudaStream_t stream) {
    const int DQKV = 2 * DQK + VH * VD;
    batch_mv(handle, weights.qkv, input, work.delta_qkv, count, false,
             work.converted, stream);
    batch_mv(handle, weights.z, input, work.delta_z, count, false,
             work.converted, stream);
    batch_mv(handle, weights.a, input, work.delta_a, count, false,
             work.converted, stream);
    batch_mv(handle, weights.b, input, work.delta_b, count, false,
             work.converted, stream);
    batch_conv_kernel<<<(DQKV + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
        work.delta_qkv, weights.conv, conv, count, DQKV);
    const dim3 head_token_grid(VH, count);
    batch_prepare_delta_qk_kernel<<<head_token_grid, BLOCK, 0, stream>>>(
        work.delta_qkv, count, work.delta_q, work.delta_k, DQKV, VH);
    batch_delta_rule_kernel<<<VH, BLOCK, 0, stream>>>(
        work.delta_q, work.delta_k, work.delta_qkv,
        work.delta_a, work.delta_b, weights.alog, weights.dt,
        count, memory, work.delta_output, DQKV, VH);
    batch_gated_rms_kernel<<<head_token_grid, BLOCK, 0, stream>>>(
        work.delta_output, weights.norm, work.delta_z, count, VH);
    batch_mv(handle, weights.out, work.delta_output, hidden, count, true,
             work.converted, stream);
}

void batch_attention_matrix(cublasHandle_t handle,
                            const AttentionWeights& weights,
                            float* key_cache, float* value_cache,
                            int start, const float* input, BatchWork& work,
                            float* hidden, int count, int AH, int KVH,
                            cudaStream_t stream) {
    const int AS = AH * AD, KVW = KVH * AD;
    batch_mv(handle, weights.q, input, work.query_and_gate, count, false,
             work.converted, stream);
    batch_mv(handle, weights.k, input, work.key, count, false,
             work.converted, stream);
    batch_mv(handle, weights.v, input, work.value, count, false,
             work.converted, stream);
    const dim3 query_grid(AH, count);
    const dim3 key_grid(KVH, count);
    batch_prepare_query_kernel<<<query_grid, BLOCK, 0, stream>>>(
        work.query_and_gate, weights.qnorm, start, count,
        work.query, work.attention_gate, AH, AS);
    batch_prepare_key_kernel<<<key_grid, BLOCK, 0, stream>>>(
        work.key, weights.knorm, start, count, KVH);
    batch_store_kv_kernel<<<(count * KVW + BLOCK - 1) / BLOCK,
                            BLOCK, 0, stream>>>(
        key_cache, value_cache, start, count, work.key, work.value, KVW);
    batch_attention_kernel<<<query_grid, BLOCK, 0, stream>>>(
        work.query, work.attention_gate, key_cache, value_cache,
        start, count, work.attention_output, AH, KVH);
    batch_mv(handle, weights.out, work.attention_output, hidden, count, true,
             work.converted, stream);
}

void batch_ffn_matrix(cublasHandle_t handle, const Layer& layer,
                      const float* input, BatchWork& work,
                      float* hidden, int count, int I, cudaStream_t stream) {
    batch_mv(handle, layer.gate, input, work.ffn_gate, count, false,
             work.converted, stream);
    batch_mv(handle, layer.up, input, work.ffn_up, count, false,
             work.converted, stream);
    batch_swiglu_kernel<<<(count * I + BLOCK - 1) / BLOCK,
                           BLOCK, 0, stream>>>(
        work.ffn_gate, work.ffn_up, count, I);
    batch_mv(handle, layer.down, work.ffn_gate, hidden, count, true,
             work.converted, stream);
}

void prefill_chunk_fp32(const Model& model, StateData& state,
                        int start, int count, bool compute_logits) {
    Q3X_ASSERT(model.config, "CUDA prefill model config is null");
    const q3x_model::ModelConfig& c = *model.config;
    const int H = c.H, I = c.I, N = c.N;
    const int AH = c.AH, KVH = c.KVH, VH = c.VH;
    Q3X_ASSERT(count > 0 && count <= PREFILL_CHUNK,
               "CUDA prefill chunk count=%d limit=%d", count, PREFILL_CHUNK);
    BatchWork& work = state.batch;
    cudaStream_t stream = state.stream;
    batch_embed(model.embedding, state.device_tokens, start, count,
                work.hidden, stream);
    for (int index = 0; index < N; ++index) {
        const Layer& layer = model.layers[index];
        batch_rms_kernel<<<count, BLOCK, 0, stream>>>(
            work.hidden, layer.input_norm, H, count, work.normalized);
        if (index % c.AI != c.AI - 1) {
            batch_deltanet_fp32(layer.delta,
                                state.layer[index].conv, state.layer[index].memory,
                                work.normalized, work, work.hidden,
                                count, H, VH, stream);
        } else {
            batch_attention_fp32(layer.attention, state.layer[index].key_cache,
                                 state.layer[index].value_cache, start,
                                 work.normalized, work, work.hidden, count,
                                 H, AH, KVH, stream);
        }
        batch_rms_kernel<<<count, BLOCK, 0, stream>>>(
            work.hidden, layer.post_norm, H, count, work.normalized);
        batch_ffn_fp32(layer, work.normalized, work,
                       work.hidden, count, H, I, stream);
    }
    if (compute_logits) {
        const float* last = work.hidden + static_cast<size_t>(count - 1) * H;
        rms(last, model.final_norm, H, state.work.normalized, stream);
        mv(model.lm_head, state.work.normalized, state.work.logits, stream);
    }
}

void prefill_chunk_matrix(const Model& model, StateData& state,
                          int start, int count, bool compute_logits) {
    Q3X_ASSERT(model.config, "CUDA prefill model config is null");
    const q3x_model::ModelConfig& c = *model.config;
    const int H = c.H, I = c.I, N = c.N;
    const int AH = c.AH, KVH = c.KVH, VH = c.VH;
    Q3X_ASSERT(count > 0 && count <= PREFILL_CHUNK,
               "CUDA prefill chunk count=%d limit=%d", count, PREFILL_CHUNK);
    BatchWork& work = state.batch;
    cudaStream_t stream = state.stream;
    batch_embed(model.embedding, state.device_tokens, start, count,
                work.hidden, stream);
    for (int index = 0; index < N; ++index) {
        const Layer& layer = model.layers[index];
        batch_rms_kernel<<<count, BLOCK, 0, stream>>>(
            work.hidden, layer.input_norm, H, count, work.normalized);
        if (index % c.AI != c.AI - 1) {
            batch_deltanet_matrix(state.cublas, layer.delta,
                                  state.layer[index].conv,
                                  state.layer[index].memory,
                                  work.normalized, work, work.hidden,
                                  count, VH, stream);
        } else {
            batch_attention_matrix(state.cublas, layer.attention,
                                   state.layer[index].key_cache,
                                   state.layer[index].value_cache, start,
                                   work.normalized, work, work.hidden, count,
                                   AH, KVH, stream);
        }
        batch_rms_kernel<<<count, BLOCK, 0, stream>>>(
            work.hidden, layer.post_norm, H, count, work.normalized);
        batch_ffn_matrix(state.cublas, layer, work.normalized, work,
                         work.hidden, count, I, stream);
    }
    if (compute_logits) {
        const float* last = work.hidden + static_cast<size_t>(count - 1) * H;
        rms(last, model.final_norm, H, state.work.normalized, stream);
        mv(model.lm_head, state.work.normalized, state.work.logits, stream);
    }
}

void forward(const Model& model, StateData& state, bool compute_logits) {
    Q3X_ASSERT(model.config, "CUDA forward model config is null");
    const q3x_model::ModelConfig& c = *model.config;
    const int H = c.H, I = c.I, N = c.N;
    const int AH = c.AH, KVH = c.KVH, VH = c.VH;
    Work& work = state.work;
    cudaStream_t stream = state.stream;
    embed(model.embedding, state.device_tokens, state.device_position,
          work.hidden, stream);
    for (int index = 0; index < N; ++index) {
        const Layer& layer = model.layers[index];
        rms(work.hidden, layer.input_norm, H, work.normalized, stream);
        if (index % c.AI != c.AI - 1) {
            deltanet(layer.delta, state.layer[index].conv,
                     state.layer[index].memory,
                     work.normalized, work, work.hidden, VH, stream);
        } else {
            attention(layer.attention, state.layer[index].key_cache,
                      state.layer[index].value_cache, state.device_position,
                      work.normalized, work, work.hidden, AH, KVH, stream);
        }
        rms(work.hidden, layer.post_norm, H, work.normalized, stream);
        if (c.experts)
            moe(layer.moe, work.normalized, work, work.hidden, c, stream);
        else
            ffn(layer, work.normalized, work, work.hidden, I, stream);
    }
    if (compute_logits) {
        rms(work.hidden, model.final_norm, H, work.normalized, stream);
        mv(model.lm_head, work.normalized, work.logits, stream);
    }
    advance_position_kernel<<<1, 1, 0, stream>>>(state.device_position);
}

cudaGraphExec_t capture_forward(const Model& model, StateData& state,
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

template <typename T>
const T* take(const void* base, size_t size, size_t& cursor,
              size_t count, const char** error) {
    if (*error) return nullptr;
    const size_t padding = (64 - cursor % 64) % 64;
    if (cursor > size || padding > size - cursor) {
        *error = "truncated model.bin";
        return nullptr;
    }
    cursor += padding;
    if (count > std::numeric_limits<size_t>::max() / sizeof(T) ||
        count * sizeof(T) > size - cursor) {
        *error = "truncated model.bin";
        return nullptr;
    }
    const T* result = reinterpret_cast<const T*>(
        static_cast<const uint8_t*>(base) + cursor);
    cursor += count * sizeof(T);
    return result;
}

}  // namespace qwen3x_cuda

namespace q3x_backend {

bool Model::load(const char* path, const char** error) {
    using qwen3x_cuda::BF16;
    using qwen3x_cuda::Layer;
    using qwen3x_cuda::Linear;

    int fd = -1;
    size_t size = 0;
    const uint8_t* file = nullptr;
    auto close_file = [&]() {
        if (file) munmap(const_cast<uint8_t*>(file), size);
        if (fd >= 0) close(fd);
        file = nullptr;
        fd = -1;
    };
    auto fail = [&](const char* message) {
        *error = message;
        close_file();
        return false;
    };

    fd = open(path, O_RDONLY);
    if (fd < 0) return fail("cannot open model.bin");
    struct stat information {};
    if (fstat(fd, &information) ||
        information.st_size < static_cast<off_t>(q3x_model::HEADER_SIZE)) {
        return fail("bad model.bin");
    }
    size = static_cast<size_t>(information.st_size);
    file = static_cast<const uint8_t*>(
        mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (file == MAP_FAILED) {
        file = nullptr;
        return fail("mmap model.bin failed");
    }
    if (std::memcmp(file, "Q3XMODL\0", 8) != 0)
        return fail("wrong model.bin magic");
    const uint32_t id = q3x_model::header_field(file, q3x_model::MODEL_ID);
    config = q3x_model::config_for_id(id);
    if (!config) return fail("unsupported Qwen model ID");
    if (!q3x_model::header_matches(file, size, *config))
        return fail("Qwen model.bin header mismatch");

    cudaError_t status = cudaMalloc(&weights, size);
    if (status == cudaErrorMemoryAllocation)
        return fail("CUDA model does not fit in device memory");
    if (status != cudaSuccess) return fail("CUDA model allocation failed");
    status = cudaMemcpy(weights, file, size, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) return fail("CUDA model upload failed");

    size_t cursor = q3x_model::HEADER_SIZE;
    const auto& c = *config;
    const int AS = c.AH * c.AD, KVW = c.KVH * c.AD;
    const int DO = c.VH * c.VD, DQKV = 2 * c.KH * c.KD + DO;
    auto bf16 = [&](size_t count) {
        return qwen3x_cuda::take<BF16>(weights, size, cursor, count, error);
    };
    auto fp32 = [&](size_t count) {
        return qwen3x_cuda::take<float>(weights, size, cursor, count, error);
    };
    auto linear = [&](int rows, int columns,
                      q3x_model::MatrixType type) -> Linear {
        switch (type) {
        case q3x_model::MATRIX_BF16:
            return Linear {qwen3x_cuda::take<BF16>(
                               weights, size, cursor,
                               static_cast<size_t>(rows) * columns, error),
                           rows, columns, type};
        case q3x_model::MATRIX_Q8_0:
            Q3X_ASSERT(columns % q3x_q8::BLOCK_SIZE == 0,
                       "CUDA Q8 matrix rows=%d columns=%d", rows, columns);
            return Linear {qwen3x_cuda::take<q3x_q8::Block>(
                               weights, size, cursor,
                               static_cast<size_t>(rows) * columns /
                               q3x_q8::BLOCK_SIZE, error),
                           rows, columns, type};
        case q3x_model::MATRIX_Q4_0:
            Q3X_ASSERT(columns % q3x_q4::BLOCK_SIZE == 0,
                       "CUDA Q4 matrix rows=%d columns=%d", rows, columns);
            return Linear {qwen3x_cuda::take<q3x_q4::Block>(
                               weights, size, cursor,
                               static_cast<size_t>(rows) * columns /
                               q3x_q4::BLOCK_SIZE, error),
                           rows, columns, type};
        }
        Q3X_ASSERT(false, "unknown CUDA loader matrix type=%u", type);
        return Linear {};
    };
    auto model_linear = [&](int rows, int columns) {
        return linear(rows, columns, c.matrix_type);
    };
    auto expert = [&](int rows, int columns) {
        Q3X_ASSERT(c.matrix_type == q3x_model::MATRIX_Q4_0 &&
                   columns % q3x_q4::BLOCK_SIZE == 0,
                   "CUDA expert matrix rows=%d columns=%d type=%u",
                   rows, columns, c.matrix_type);
        return qwen3x_cuda::ExpertLinear {
            qwen3x_cuda::take<q3x_q4::Block>(
                weights, size, cursor,
                static_cast<size_t>(c.experts) * rows * columns /
                q3x_q4::BLOCK_SIZE, error),
            c.experts, rows, columns, c.matrix_type,
        };
    };
    layers.reset(new (std::nothrow) Layer[c.N]);
    if (!layers) return fail("cannot allocate CUDA model layer table");
    embedding = model_linear(c.V, c.H);
    lm_head = c.tied_embeddings ? embedding : model_linear(c.V, c.H);
    final_norm = bf16(c.H);
    for (int i = 0; i < c.N; ++i) {
        Layer& l = layers[i];
        l.input_norm = bf16(c.H);
        if (i % c.AI != c.AI - 1) {
            l.delta.qkv = model_linear(DQKV, c.H);
            l.delta.z = model_linear(DO, c.H);
            l.delta.a = model_linear(c.VH, c.H);
            l.delta.b = model_linear(c.VH, c.H);
            l.delta.conv = bf16(static_cast<size_t>(DQKV) * c.CK);
            l.delta.alog = fp32(c.VH);
            l.delta.dt = bf16(c.VH);
            l.delta.norm = fp32(c.VD);
            l.delta.out = model_linear(c.H, DO);
        } else {
            l.attention.q = model_linear(2 * AS, c.H);
            l.attention.k = model_linear(KVW, c.H);
            l.attention.v = model_linear(KVW, c.H);
            l.attention.qnorm = bf16(c.AD);
            l.attention.knorm = bf16(c.AD);
            l.attention.out = model_linear(c.H, AS);
        }
        l.post_norm = bf16(c.H);
        if (c.experts) {
            l.moe.router = linear(c.experts, c.H, q3x_model::MATRIX_BF16);
            l.moe.gate_up = expert(2 * c.I, c.H);
            l.moe.down = expert(c.H, c.I);
            l.moe.shared_gate = model_linear(c.shared_I, c.H);
            l.moe.shared_up = model_linear(c.shared_I, c.H);
            l.moe.shared_down = model_linear(c.H, c.shared_I);
            l.moe.shared_scale = linear(1, c.H, q3x_model::MATRIX_BF16);
        } else {
            l.gate = model_linear(c.I, c.H);
            l.up = model_linear(c.I, c.H);
            l.down = model_linear(c.H, c.I);
        }
    }
    if (*error) return fail(*error);
    if (cursor != size)
        return fail("model.bin size does not match selected Qwen schema");
    if (c.experts) {
        size_t free_memory = 0, total_memory = 0;
        CUDA_OK(cudaMemGetInfo(&free_memory, &total_memory));
        LOG_INFO("CUDA Qwen3.6 weights device_allocation_bytes=%zu "
                 "free_bytes=%zu total_bytes=%zu",
                 size, free_memory, total_memory);
    }
    close_file();
    return true;
}

struct State {
    qwen3x_cuda::StateData data;
    State(const q3x_model::ModelConfig& config, int context_size)
        : data(config, context_size) {}
};

Model* model_create(const char* path, char* err, size_t errlen) {
    Q3X_ASSERT(path, "CUDA model_create path is null");
    std::unique_ptr<Model> model(new (std::nothrow) Model());
    const char* message = nullptr;
    if (!model || !model->load(path, &message)) {
        if (err && errlen > 0)
            std::snprintf(err, errlen, "%s", message ? message : "allocation failed");
        return nullptr;
    }
    if (err && errlen > 0) err[0] = '\0';
    return model.release();
}

void model_destroy(Model* model) { delete model; }

State* state_create(Model* model, int context_size) {
    Q3X_ASSERT(model, "CUDA state_create model is null");
    Q3X_ASSERT(context_size > 0 && context_size <= qwen3x_cuda::MAX_CONTEXT,
               "CUDA state_create context_size=%d", context_size);
    Q3X_ASSERT(model->config, "CUDA state_create model config is null");
    return new State(*model->config, context_size);
}

void state_destroy(State* state) { delete state; }

void reset_state(State* state) {
    CUDA_OK(cudaMemsetAsync(state->data.recurrent.memory, 0,
                            state->data.recurrent.count * sizeof(float),
                            state->data.stream));
    state->data.position = 0;
    CUDA_OK(cudaMemcpyAsync(state->data.device_position,
                            &state->data.position, sizeof(int),
                            cudaMemcpyHostToDevice, state->data.stream));
    CUDA_OK(cudaStreamSynchronize(state->data.stream));
}

void state_reset(State* state) {
    Q3X_ASSERT(state, "CUDA state_reset state is null");
    reset_state(state);
}

void state_forward(Model* model, State* state,
                   const int* tokens, int count, bool compute_logits) {
    Q3X_ASSERT(model && state && tokens && count > 0,
               "CUDA state_forward model=%p state=%p tokens=%p count=%d",
               static_cast<void*>(model), static_cast<void*>(state),
               static_cast<const void*>(tokens), count);
    Q3X_ASSERT(state->data.position >= 0 &&
               state->data.position + count <= state->data.capacity,
               "CUDA state_forward position=%d count=%d capacity=%d",
               state->data.position, count, state->data.capacity);
    Q3X_ASSERT(model->config, "CUDA state_forward model config is null");
    for (int index = 0; index < count; ++index) {
        Q3X_ASSERT(tokens[index] >= 0 && tokens[index] < model->config->V,
                   "CUDA state_forward token[%d]=%d vocabulary=%d",
                   index, tokens[index], model->config->V);
    }
    qwen3x_cuda::StateData& data = state->data;
    CUDA_OK(cudaMemcpyAsync(
        data.device_tokens + data.position, tokens,
        static_cast<size_t>(count) * sizeof(int),
        cudaMemcpyHostToDevice, data.stream));
    if (model->config->experts) {
        CUDA_OK(cudaMemcpyAsync(data.device_position, &data.position,
                                sizeof(int), cudaMemcpyHostToDevice,
                                data.stream));
        for (int index = 0; index < count; ++index) {
            qwen3x_cuda::forward(*model, data,
                                 compute_logits && index + 1 == count);
        }
    } else if (count == 1) {
        if (!data.forward_graph) {
            data.forward_graph = qwen3x_cuda::capture_forward(
                *model, data, false);
            data.logits_graph = qwen3x_cuda::capture_forward(
                *model, data, true);
        }
        CUDA_OK(cudaMemcpyAsync(data.device_position, &data.position,
                                sizeof(int), cudaMemcpyHostToDevice,
                                data.stream));
        CUDA_OK(cudaGraphLaunch(compute_logits ? data.logits_graph
                                               : data.forward_graph,
                                data.stream));
    } else {
        for (int offset = 0; offset < count;) {
            const int chunk = std::min(qwen3x_cuda::PREFILL_CHUNK,
                                       count - offset);
            switch (model->config->id) {
            case q3x_model::QWEN35_08B.id:
                qwen3x_cuda::prefill_chunk_fp32(
                    *model, data, data.position + offset, chunk,
                    compute_logits && offset + chunk == count);
                break;
            case q3x_model::QWEN35_4B.id:
            case q3x_model::QWEN35_9B.id:
                qwen3x_cuda::prefill_chunk_matrix(
                    *model, data, data.position + offset, chunk,
                    compute_logits && offset + chunk == count);
                break;
            default:
                Q3X_ASSERT(false, "CUDA state_forward unsupported model_id=%u",
                           model->config->id);
            }
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
    Q3X_ASSERT(state, "CUDA state_checkpoint_save state is null");
    qwen3x_cuda::StateData& data = state->data;
    data.checkpoint_position = data.position;
    CUDA_OK(cudaMemcpyAsync(data.checkpoint_recurrent, data.recurrent.memory,
                            data.recurrent.count * sizeof(float),
                            cudaMemcpyDeviceToDevice, data.stream));
    CUDA_OK(cudaMemcpyAsync(data.checkpoint_logits, data.work.logits,
                            static_cast<size_t>(data.config->V) * sizeof(float),
                            cudaMemcpyDeviceToDevice, data.stream));
    CUDA_OK(cudaStreamSynchronize(data.stream));
}

void state_checkpoint_restore(State* state) {
    Q3X_ASSERT(state, "CUDA state_checkpoint_restore state is null");
    qwen3x_cuda::StateData& data = state->data;
    CUDA_OK(cudaMemcpyAsync(data.recurrent.memory, data.checkpoint_recurrent,
                            data.recurrent.count * sizeof(float),
                            cudaMemcpyDeviceToDevice, data.stream));
    CUDA_OK(cudaMemcpyAsync(data.work.logits, data.checkpoint_logits,
                            static_cast<size_t>(data.config->V) * sizeof(float),
                            cudaMemcpyDeviceToDevice, data.stream));
    data.position = data.checkpoint_position;
    CUDA_OK(cudaMemcpyAsync(data.device_position, &data.position, sizeof(int),
                            cudaMemcpyHostToDevice, data.stream));
    CUDA_OK(cudaStreamSynchronize(data.stream));
}

int state_argmax(const State* state) {
    if (!state) return -1;
    State* mutable_state = const_cast<State*>(state);
    CUDA_OK(cudaMemcpy(mutable_state->data.host_logits.get(),
                       state->data.work.logits,
                       static_cast<size_t>(state->data.config->V) * sizeof(float),
                       cudaMemcpyDeviceToHost));
    return static_cast<int>(std::max_element(
        mutable_state->data.host_logits.get(),
        mutable_state->data.host_logits.get() + state->data.config->V) -
        mutable_state->data.host_logits.get());
}

void state_copy_logits(const State* state, float* output) {
    Q3X_ASSERT(state && output, "CUDA state_copy_logits state=%p output=%p",
               static_cast<const void*>(state), static_cast<void*>(output));
    CUDA_OK(cudaMemcpy(output, state->data.work.logits,
                       static_cast<size_t>(state->data.config->V) * sizeof(float),
                       cudaMemcpyDeviceToHost));
}

int vocab_size() { return q3x_model::QWEN35_08B.V; }
int max_context() { return qwen3x_cuda::MAX_CONTEXT; }
bool token_is_stop(int token) {
    return token == qwen3x_cuda::END_OF_TEXT_TOKEN ||
           token == qwen3x_cuda::IM_END_TOKEN;
}

uint32_t model_id(const Model* model) {
    Q3X_ASSERT(model, "CUDA model_id model is null");
    Q3X_ASSERT(model->config, "CUDA model_id config is null");
    return model->config->id;
}

}  // namespace q3x_backend

// qwen38_tiny_cuda.cu -- CUDA correctness backend for one fixed Qwen shape.
//
// This is intentionally direct: every operation in the 4-layer fake model
// stays on the GPU, while the host only sequences batch-1 decode. GEMV is a
// readable one-block-per-row kernel, not a performance implementation.  This
// source defaults to the tiny fake shape; qwen38_08b_cuda.cu defines the
// Qwen3.5-0.8B development shape before including it.

#define QWEN38_NO_MAIN 1
#ifndef QWEN38_CUDA_MODEL_DEFINE
#define QWEN38_TINY_MODEL 1
#endif
#include "../qwen38.cpp"

#include <cuda_runtime.h>

namespace qwen38 {
namespace {

[[noreturn]] void cuda_fail(cudaError_t error, const char* operation) {
    std::fprintf(stderr, "qwen38: CUDA %s: %s\n", operation, cudaGetErrorString(error));
    std::exit(1);
}

void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) cuda_fail(error, operation);
}

#define CUDA_CHECK(call) cuda_check((call), #call)
#define CUDA_LAUNCH() cuda_check(cudaGetLastError(), "kernel launch")

__device__ float bf16_to_float_device(uint16_t value) {
    return __uint_as_float(static_cast<uint32_t>(value) << 16);
}

__device__ uint16_t float_to_bf16_device(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t bias = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + bias) >> 16);
}

__device__ float device_sigmoid(float x) {
    if (x >= 0.0f) {
        const float e = expf(-x);
        return 1.0f / (1.0f + e);
    }
    const float e = expf(x);
    return e / (1.0f + e);
}

__device__ float device_silu(float x) { return x * device_sigmoid(x); }

__device__ float device_softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

__global__ void embedding_kernel(const uint16_t* embedding, int token, int hidden, float* output) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden) output[i] = bf16_to_float_device(embedding[static_cast<size_t>(token) * hidden + i]);
}

__global__ void matvec_bf16_kernel(const uint16_t* weight, const float* input, int out_dim, int in_dim, float* output) {
    const int row = blockIdx.x;
    const int thread = threadIdx.x;
    if (row >= out_dim) return;
    float sum = 0.0f;
    const size_t offset = static_cast<size_t>(row) * in_dim;
    for (int col = thread; col < in_dim; col += blockDim.x) sum += bf16_to_float_device(weight[offset + col]) * input[col];
    __shared__ float partial[128];
    partial[thread] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) partial[thread] += partial[thread + stride];
        __syncthreads();
    }
    if (thread == 0) output[row] = partial[0];
}

__global__ void rmsnorm_plus_kernel(const float* input, const uint16_t* weight, int size, float* output) {
    __shared__ float partial[128];
    const int thread = threadIdx.x;
    float sum = 0.0f;
    for (int i = thread; i < size; i += blockDim.x) sum += input[i] * input[i];
    partial[thread] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) partial[thread] += partial[thread + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / size + Config::kRmsNormEps);
    for (int i = thread; i < size; i += blockDim.x) {
        output[i] = input[i] * scale * (1.0f + bf16_to_float_device(weight[i]));
    }
}

__global__ void add_kernel(float* destination, const float* source, int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) destination[i] += source[i];
}

__global__ void swiglu_kernel(float* gate, const float* up, int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) gate[i] = device_silu(gate[i]) * up[i];
}

__global__ void depthwise_conv_kernel(float* values, const uint16_t* weight, float* state, int channels, int kernel_size) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    const int history = kernel_size - 1;
    float* past = state + static_cast<size_t>(channel) * history;
    const uint16_t* w = weight + static_cast<size_t>(channel) * kernel_size;
    float sum = 0.0f;
    for (int i = 0; i < history; ++i) sum += past[i] * bf16_to_float_device(w[i]);
    sum += values[channel] * bf16_to_float_device(w[history]);
    for (int i = 0; i + 1 < history; ++i) past[i] = past[i + 1];
    past[history - 1] = values[channel];
    values[channel] = device_silu(sum);
}

__global__ void delta_prepare_kernel(const float* qkv, const float* a, const float* b, const float* a_log,
                                     const uint16_t* dt_bias, float* q, float* k, float* beta, float* log_decay) {
    const int head = blockIdx.x * blockDim.x + threadIdx.x;
    if (head >= Config::kDeltaValueHeads) return;
    const int qk_head = head / (Config::kDeltaValueHeads / Config::kDeltaKeyHeads);
    const float* q_small = qkv + static_cast<size_t>(qk_head) * Config::kDeltaKeyDim;
    const float* k_small = qkv + Config::kDeltaQkSize + static_cast<size_t>(qk_head) * Config::kDeltaKeyDim;
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (int i = 0; i < Config::kDeltaKeyDim; ++i) {
        q_sum += q_small[i] * q_small[i];
        k_sum += k_small[i] * k_small[i];
    }
    const float q_scale = rsqrtf(q_sum + Config::kRmsNormEps) / sqrtf(static_cast<float>(Config::kDeltaKeyDim));
    const float k_scale = rsqrtf(k_sum + Config::kRmsNormEps);
    for (int i = 0; i < Config::kDeltaKeyDim; ++i) {
        q[static_cast<size_t>(head) * Config::kDeltaKeyDim + i] = q_small[i] * q_scale;
        k[static_cast<size_t>(head) * Config::kDeltaKeyDim + i] = k_small[i] * k_scale;
    }
    beta[head] = device_sigmoid(b[head]);
    log_decay[head] = -expf(a_log[head]) * device_softplus(a[head] + bf16_to_float_device(dt_bias[head]));
}

__global__ void delta_recurrence_kernel(const float* q, const float* k, const float* value, const float* beta,
                                        const float* log_decay, float* state, float* output) {
    const int head = blockIdx.x * blockDim.x + threadIdx.x;
    if (head >= Config::kDeltaValueHeads) return;
    const float decay = expf(log_decay[head]);
    const float* q_head = q + static_cast<size_t>(head) * Config::kDeltaKeyDim;
    const float* k_head = k + static_cast<size_t>(head) * Config::kDeltaKeyDim;
    const float* value_head = value + static_cast<size_t>(head) * Config::kDeltaValueDim;
    float* matrix = state + static_cast<size_t>(head) * Config::kDeltaKeyDim * Config::kDeltaValueDim;
    float memory[Config::kDeltaValueDim];

    for (int key = 0; key < Config::kDeltaKeyDim; ++key) {
        for (int value_index = 0; value_index < Config::kDeltaValueDim; ++value_index) {
            matrix[key * Config::kDeltaValueDim + value_index] *= decay;
        }
    }
    for (int value_index = 0; value_index < Config::kDeltaValueDim; ++value_index) {
        float sum = 0.0f;
        for (int key = 0; key < Config::kDeltaKeyDim; ++key) sum += k_head[key] * matrix[key * Config::kDeltaValueDim + value_index];
        memory[value_index] = sum;
    }
    for (int key = 0; key < Config::kDeltaKeyDim; ++key) {
        for (int value_index = 0; value_index < Config::kDeltaValueDim; ++value_index) {
            matrix[key * Config::kDeltaValueDim + value_index] +=
                k_head[key] * beta[head] * (value_head[value_index] - memory[value_index]);
        }
    }
    for (int value_index = 0; value_index < Config::kDeltaValueDim; ++value_index) {
        float sum = 0.0f;
        for (int key = 0; key < Config::kDeltaKeyDim; ++key) sum += q_head[key] * matrix[key * Config::kDeltaValueDim + value_index];
        output[static_cast<size_t>(head) * Config::kDeltaValueDim + value_index] = sum;
    }
}

__global__ void delta_gated_norm_kernel(float* values, const float* norm, const float* z) {
    const int head = blockIdx.x * blockDim.x + threadIdx.x;
    if (head >= Config::kDeltaValueHeads) return;
    float* value = values + static_cast<size_t>(head) * Config::kDeltaValueDim;
    const float* z_head = z + static_cast<size_t>(head) * Config::kDeltaValueDim;
    float square_sum = 0.0f;
    for (int i = 0; i < Config::kDeltaValueDim; ++i) square_sum += value[i] * value[i];
    const float scale = rsqrtf(square_sum / Config::kDeltaValueDim + Config::kRmsNormEps);
    for (int i = 0; i < Config::kDeltaValueDim; ++i) {
        value[i] = value[i] * scale * norm[i] * device_silu(z_head[i]);
    }
}

__global__ void attention_query_kernel(const float* projection, const uint16_t* norm, int position, float* query,
                                       float* gate) {
    const int head = blockIdx.x * blockDim.x + threadIdx.x;
    if (head >= Config::kAttentionHeads) return;
    const float* projected = projection + static_cast<size_t>(head) * 2 * Config::kAttentionHeadDim;
    float square_sum = 0.0f;
    for (int i = 0; i < Config::kAttentionHeadDim; ++i) square_sum += projected[i] * projected[i];
    const float scale = rsqrtf(square_sum / Config::kAttentionHeadDim + Config::kRmsNormEps);
    float* q = query + static_cast<size_t>(head) * Config::kAttentionHeadDim;
    float* g = gate + static_cast<size_t>(head) * Config::kAttentionHeadDim;
    for (int i = 0; i < Config::kAttentionHeadDim; ++i) {
        q[i] = projected[i] * scale * (1.0f + bf16_to_float_device(norm[i]));
        g[i] = projected[Config::kAttentionHeadDim + i];
    }
    for (int i = 0; i < Config::kRotaryDim / 2; ++i) {
        const float angle = static_cast<float>(position) /
                            powf(Config::kRopeTheta, static_cast<float>(2 * i) / Config::kRotaryDim);
        const float first = q[i];
        const float second = q[i + Config::kRotaryDim / 2];
        q[i] = first * cosf(angle) - second * sinf(angle);
        q[i + Config::kRotaryDim / 2] = second * cosf(angle) + first * sinf(angle);
    }
}

__global__ void attention_key_kernel(float* key, const uint16_t* norm, int position) {
    const int head = blockIdx.x * blockDim.x + threadIdx.x;
    if (head >= Config::kKvHeads) return;
    float* k = key + static_cast<size_t>(head) * Config::kAttentionHeadDim;
    float square_sum = 0.0f;
    for (int i = 0; i < Config::kAttentionHeadDim; ++i) square_sum += k[i] * k[i];
    const float scale = rsqrtf(square_sum / Config::kAttentionHeadDim + Config::kRmsNormEps);
    for (int i = 0; i < Config::kAttentionHeadDim; ++i) k[i] = k[i] * scale * (1.0f + bf16_to_float_device(norm[i]));
    for (int i = 0; i < Config::kRotaryDim / 2; ++i) {
        const float angle = static_cast<float>(position) /
                            powf(Config::kRopeTheta, static_cast<float>(2 * i) / Config::kRotaryDim);
        const float first = k[i];
        const float second = k[i + Config::kRotaryDim / 2];
        k[i] = first * cosf(angle) - second * sinf(angle);
        k[i + Config::kRotaryDim / 2] = second * cosf(angle) + first * sinf(angle);
    }
}

// CPU's tiny test model deliberately keeps a BF16 cache to exercise the
// storage format.  The real 0.8B reference uses FP32 cache values so the
// scalar CPU and CUDA oracles can both compare directly to Transformers.
#ifdef QWEN38_TINY_MODEL
using CudaAttentionCacheElement = uint16_t;
__device__ float cache_to_float(CudaAttentionCacheElement value) { return bf16_to_float_device(value); }
__device__ CudaAttentionCacheElement float_to_cache(float value) { return float_to_bf16_device(value); }
#else
using CudaAttentionCacheElement = float;
__device__ float cache_to_float(CudaAttentionCacheElement value) { return value; }
__device__ CudaAttentionCacheElement float_to_cache(float value) { return value; }
#endif

__global__ void attention_append_kernel(const float* key, const float* value, CudaAttentionCacheElement* key_cache,
                                        CudaAttentionCacheElement* value_cache,
                                        int position) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Config::kAttentionKvSize) return;
    const size_t offset = static_cast<size_t>(position) * Config::kAttentionKvSize + i;
    key_cache[offset] = float_to_cache(key[i]);
    value_cache[offset] = float_to_cache(value[i]);
}

__global__ void attention_decode_kernel(const float* query, const CudaAttentionCacheElement* key_cache,
                                        const CudaAttentionCacheElement* value_cache,
                                        int tokens, float* output) {
    const int head = blockIdx.x * blockDim.x + threadIdx.x;
    if (head >= Config::kAttentionHeads) return;
    const int queries_per_kv = Config::kAttentionHeads / Config::kKvHeads;
    const int kv_head = head / queries_per_kv;
    const float* q = query + static_cast<size_t>(head) * Config::kAttentionHeadDim;
    float maximum = -INFINITY;
    const float attention_scale = rsqrtf(static_cast<float>(Config::kAttentionHeadDim));
    for (int token = 0; token < tokens; ++token) {
        const CudaAttentionCacheElement* k =
            key_cache + (static_cast<size_t>(token) * Config::kKvHeads + kv_head) * Config::kAttentionHeadDim;
        float score = 0.0f;
        for (int d = 0; d < Config::kAttentionHeadDim; ++d) score += q[d] * cache_to_float(k[d]);
        maximum = fmaxf(maximum, score * attention_scale);
    }
    float denominator = 0.0f;
    for (int token = 0; token < tokens; ++token) {
        const CudaAttentionCacheElement* k =
            key_cache + (static_cast<size_t>(token) * Config::kKvHeads + kv_head) * Config::kAttentionHeadDim;
        float score = 0.0f;
        for (int d = 0; d < Config::kAttentionHeadDim; ++d) score += q[d] * cache_to_float(k[d]);
        denominator += expf(score * attention_scale - maximum);
    }
    float* out = output + static_cast<size_t>(head) * Config::kAttentionHeadDim;
    for (int d = 0; d < Config::kAttentionHeadDim; ++d) out[d] = 0.0f;
    for (int token = 0; token < tokens; ++token) {
        const CudaAttentionCacheElement* k =
            key_cache + (static_cast<size_t>(token) * Config::kKvHeads + kv_head) * Config::kAttentionHeadDim;
        const CudaAttentionCacheElement* v =
            value_cache + (static_cast<size_t>(token) * Config::kKvHeads + kv_head) * Config::kAttentionHeadDim;
        float score = 0.0f;
        for (int d = 0; d < Config::kAttentionHeadDim; ++d) score += q[d] * cache_to_float(k[d]);
        const float probability = expf(score * attention_scale - maximum) / denominator;
        for (int d = 0; d < Config::kAttentionHeadDim; ++d) out[d] += probability * cache_to_float(v[d]);
    }
}

__global__ void attention_gate_kernel(float* values, const float* gate, int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) values[i] *= device_sigmoid(gate[i]);
}

struct DeviceU16 {
    uint16_t* data = nullptr;
    size_t count = 0;

    DeviceU16() = default;
    explicit DeviceU16(size_t count_) : count(count_) {
        if (count) CUDA_CHECK(cudaMalloc(&data, count * sizeof(uint16_t)));
    }
    explicit DeviceU16(const Tensor& tensor) : DeviceU16(static_cast<size_t>(tensor.elements())) {
        require(tensor.info.dtype == "BF16", "CUDA BF16 buffer received a non-BF16 tensor");
        CUDA_CHECK(cudaMemcpy(data, tensor.data, count * sizeof(uint16_t), cudaMemcpyHostToDevice));
    }
    ~DeviceU16() { if (data) cudaFree(data); }
    DeviceU16(const DeviceU16&) = delete;
    DeviceU16& operator=(const DeviceU16&) = delete;
    DeviceU16(DeviceU16&& other) noexcept : data(other.data), count(other.count) { other.data = nullptr; other.count = 0; }
    DeviceU16& operator=(DeviceU16&& other) noexcept {
        if (this != &other) {
            if (data) cudaFree(data);
            data = other.data; count = other.count; other.data = nullptr; other.count = 0;
        }
        return *this;
    }
};

struct DeviceFloat {
    float* data = nullptr;
    size_t count = 0;

    DeviceFloat() = default;
    explicit DeviceFloat(size_t count_) : count(count_) {
        if (count) CUDA_CHECK(cudaMalloc(&data, count * sizeof(float)));
    }
    explicit DeviceFloat(const Tensor& tensor) : DeviceFloat(static_cast<size_t>(tensor.elements())) {
        std::vector<float> host(count);
        for (size_t index = 0; index < count; ++index) host[index] = scalar_at(tensor, index);
        CUDA_CHECK(cudaMemcpy(data, host.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    }
    ~DeviceFloat() { if (data) cudaFree(data); }
    DeviceFloat(const DeviceFloat&) = delete;
    DeviceFloat& operator=(const DeviceFloat&) = delete;
    DeviceFloat(DeviceFloat&& other) noexcept : data(other.data), count(other.count) { other.data = nullptr; other.count = 0; }
    DeviceFloat& operator=(DeviceFloat&& other) noexcept {
        if (this != &other) {
            if (data) cudaFree(data);
            data = other.data; count = other.count; other.data = nullptr; other.count = 0;
        }
        return *this;
    }
};

struct CudaLinear {
    DeviceU16 weight;
    int out_dim = 0;
    int in_dim = 0;
    CudaLinear() = default;
    explicit CudaLinear(const LinearWeight& source) : weight(*source.weight), out_dim(source.out_dim), in_dim(source.in_dim) {}
};

struct CudaNorm {
    DeviceU16 weight;
    int size = 0;
    CudaNorm() = default;
    explicit CudaNorm(const NormWeight& source) : weight(*source.weight), size(source.size) {}
};

struct CudaMlp { CudaLinear gate, up, down; };
// Qwen3.5's A_log and DeltaNet gated-RMSNorm scale are FP32. Keeping their
// device representation explicit also makes the fake BF16 fixture follow the
// same code path after a one-time host conversion.
struct CudaDelta { CudaLinear qkv, z, a, b, out; DeviceU16 conv, dt_bias; DeviceFloat a_log, norm; };
struct CudaAttention { CudaLinear q, k, v, out; CudaNorm q_norm, k_norm; };
struct CudaLayer { bool is_deltanet = false; CudaNorm input_norm, post_attention_norm; CudaMlp mlp; CudaDelta delta; CudaAttention attention; };

#ifdef QWEN38_TINY_MODEL
using CudaAttentionCache = DeviceU16;
#else
using CudaAttentionCache = DeviceFloat;
#endif

class CudaModel {
  public:
    explicit CudaModel(const TextModel& host) : embedding_(host.embedding()), final_norm_(host.final_norm()) {
        // Qwen3.5-0.8B ties these weights. Do not silently spend a second
        // vocabulary matrix on the GPU just because the CUDA view is simple.
        if (!Config::kTiedWordEmbeddings) lm_head_ = DeviceU16(host.lm_head());
        layers_.reserve(Config::kLayers);
        for (int index = 0; index < Config::kLayers; ++index) {
            const LayerWeight& source = host.layer(index);
            CudaLayer layer;
            layer.is_deltanet = source.is_deltanet;
            layer.input_norm = CudaNorm(source.input_norm);
            layer.post_attention_norm = CudaNorm(source.post_attention_norm);
            layer.mlp.gate = CudaLinear(source.mlp.gate);
            layer.mlp.up = CudaLinear(source.mlp.up);
            layer.mlp.down = CudaLinear(source.mlp.down);
            if (layer.is_deltanet) {
                layer.delta.qkv = CudaLinear(source.deltanet.qkv);
                layer.delta.z = CudaLinear(source.deltanet.z);
                layer.delta.a = CudaLinear(source.deltanet.a);
                layer.delta.b = CudaLinear(source.deltanet.b);
                layer.delta.out = CudaLinear(source.deltanet.out);
                layer.delta.conv = DeviceU16(*source.deltanet.conv);
                layer.delta.a_log = DeviceFloat(*source.deltanet.a_log);
                layer.delta.dt_bias = DeviceU16(*source.deltanet.dt_bias);
                layer.delta.norm = DeviceFloat(*source.deltanet.norm.weight);
            } else {
                layer.attention.q = CudaLinear(source.attention.q);
                layer.attention.k = CudaLinear(source.attention.k);
                layer.attention.v = CudaLinear(source.attention.v);
                layer.attention.out = CudaLinear(source.attention.out);
                layer.attention.q_norm = CudaNorm(source.attention.q_norm);
                layer.attention.k_norm = CudaNorm(source.attention.k_norm);
            }
            layers_.push_back(std::move(layer));
        }
    }

    const DeviceU16& embedding() const { return embedding_; }
    const DeviceU16& lm_head() const { return Config::kTiedWordEmbeddings ? embedding_ : lm_head_; }
    const CudaNorm& final_norm() const { return final_norm_; }
    const CudaLayer& layer(int index) const { return layers_[index]; }

  private:
    DeviceU16 embedding_;
    DeviceU16 lm_head_;
    CudaNorm final_norm_;
    std::vector<CudaLayer> layers_;
};

struct CudaState {
    int position = 0;
    int max_tokens = 0;
    std::vector<DeviceFloat> conv;
    std::vector<DeviceFloat> recurrent;
    std::vector<CudaAttentionCache> keys;
    std::vector<CudaAttentionCache> values;

    explicit CudaState(int max_tokens_) : max_tokens(max_tokens_), conv(Config::kLayers), recurrent(Config::kLayers),
                                          keys(Config::kLayers), values(Config::kLayers) {
        for (int layer = 0; layer < Config::kLayers; ++layer) {
            if (Config::is_deltanet_layer(layer)) {
                conv[layer] = DeviceFloat(Config::kDeltaQkvSize * (Config::kDeltaConvKernel - 1));
                recurrent[layer] = DeviceFloat(Config::kDeltaValueHeads * Config::kDeltaKeyDim * Config::kDeltaValueDim);
                CUDA_CHECK(cudaMemset(conv[layer].data, 0, conv[layer].count * sizeof(float)));
                CUDA_CHECK(cudaMemset(recurrent[layer].data, 0, recurrent[layer].count * sizeof(float)));
            } else {
                keys[layer] = CudaAttentionCache(static_cast<size_t>(max_tokens) * Config::kAttentionKvSize);
                values[layer] = CudaAttentionCache(static_cast<size_t>(max_tokens) * Config::kAttentionKvSize);
            }
        }
    }
};

struct CudaWorkspace {
    DeviceFloat hidden{Config::kHiddenSize};
    DeviceFloat normalized{Config::kHiddenSize};
    DeviceFloat mixer{Config::kHiddenSize};
    DeviceFloat mlp_gate{Config::kIntermediateSize};
    DeviceFloat mlp_up{Config::kIntermediateSize};
    DeviceFloat delta_qkv{Config::kDeltaQkvSize};
    DeviceFloat delta_z{Config::kDeltaOutputSize};
    DeviceFloat delta_a{Config::kDeltaValueHeads};
    DeviceFloat delta_b{Config::kDeltaValueHeads};
    DeviceFloat delta_q{Config::kDeltaOutputSize};
    DeviceFloat delta_k{Config::kDeltaOutputSize};
    DeviceFloat delta_beta{Config::kDeltaValueHeads};
    DeviceFloat delta_decay{Config::kDeltaValueHeads};
    DeviceFloat delta_out{Config::kDeltaOutputSize};
    DeviceFloat attention_projection{Config::kAttentionQProjectionSize};
    DeviceFloat attention_q{Config::kAttentionSize};
    DeviceFloat attention_gate{Config::kAttentionSize};
    DeviceFloat attention_k{Config::kAttentionKvSize};
    DeviceFloat attention_v{Config::kAttentionKvSize};
    DeviceFloat attention_out{Config::kAttentionSize};
    DeviceFloat logits{Config::kVocabSize};
};

void gpu_matvec(const CudaLinear& linear, const float* input, float* output) {
    matvec_bf16_kernel<<<linear.out_dim, 128>>>(linear.weight.data, input, linear.out_dim, linear.in_dim, output);
    CUDA_LAUNCH();
}

void gpu_rmsnorm(const float* input, const CudaNorm& norm, float* output) {
    rmsnorm_plus_kernel<<<1, 128>>>(input, norm.weight.data, norm.size, output);
    CUDA_LAUNCH();
}

void gpu_mlp(const CudaMlp& mlp, const float* input, CudaWorkspace* workspace, float* output) {
    gpu_matvec(mlp.gate, input, workspace->mlp_gate.data);
    gpu_matvec(mlp.up, input, workspace->mlp_up.data);
    swiglu_kernel<<<(Config::kIntermediateSize + 127) / 128, 128>>>(workspace->mlp_gate.data, workspace->mlp_up.data,
                                                                      Config::kIntermediateSize);
    CUDA_LAUNCH();
    gpu_matvec(mlp.down, workspace->mlp_gate.data, output);
}

void gpu_deltanet(const CudaDelta& weights, DeviceFloat* conv_state, DeviceFloat* recurrent, const float* input,
                  CudaWorkspace* workspace, float* output) {
    gpu_matvec(weights.qkv, input, workspace->delta_qkv.data);
    gpu_matvec(weights.z, input, workspace->delta_z.data);
    gpu_matvec(weights.a, input, workspace->delta_a.data);
    gpu_matvec(weights.b, input, workspace->delta_b.data);
    depthwise_conv_kernel<<<(Config::kDeltaQkvSize + 127) / 128, 128>>>(workspace->delta_qkv.data, weights.conv.data,
                                                                           conv_state->data, Config::kDeltaQkvSize,
                                                                           Config::kDeltaConvKernel);
    CUDA_LAUNCH();
    delta_prepare_kernel<<<1, Config::kDeltaValueHeads>>>(workspace->delta_qkv.data, workspace->delta_a.data,
                                                            workspace->delta_b.data, weights.a_log.data, weights.dt_bias.data,
                                                            workspace->delta_q.data, workspace->delta_k.data,
                                                            workspace->delta_beta.data, workspace->delta_decay.data);
    CUDA_LAUNCH();
    delta_recurrence_kernel<<<1, Config::kDeltaValueHeads>>>(workspace->delta_q.data, workspace->delta_k.data,
                                                               workspace->delta_qkv.data + 2 * Config::kDeltaQkSize,
                                                               workspace->delta_beta.data, workspace->delta_decay.data,
                                                               recurrent->data, workspace->delta_out.data);
    CUDA_LAUNCH();
    delta_gated_norm_kernel<<<1, Config::kDeltaValueHeads>>>(workspace->delta_out.data, weights.norm.data,
                                                               workspace->delta_z.data);
    CUDA_LAUNCH();
    gpu_matvec(weights.out, workspace->delta_out.data, output);
}

void gpu_attention(const CudaAttention& weights, CudaAttentionCache* keys, CudaAttentionCache* values, int position,
                   const float* input,
                   CudaWorkspace* workspace, float* output) {
    gpu_matvec(weights.q, input, workspace->attention_projection.data);
    gpu_matvec(weights.k, input, workspace->attention_k.data);
    gpu_matvec(weights.v, input, workspace->attention_v.data);
    attention_query_kernel<<<1, Config::kAttentionHeads>>>(workspace->attention_projection.data, weights.q_norm.weight.data,
                                                             position, workspace->attention_q.data, workspace->attention_gate.data);
    attention_key_kernel<<<1, Config::kKvHeads>>>(workspace->attention_k.data, weights.k_norm.weight.data, position);
    CUDA_LAUNCH();
    attention_append_kernel<<<1, Config::kAttentionKvSize>>>(workspace->attention_k.data, workspace->attention_v.data,
                                                               keys->data, values->data, position);
    CUDA_LAUNCH();
    attention_decode_kernel<<<1, Config::kAttentionHeads>>>(workspace->attention_q.data, keys->data, values->data,
                                                              position + 1, workspace->attention_out.data);
    CUDA_LAUNCH();
    attention_gate_kernel<<<(Config::kAttentionSize + 127) / 128, 128>>>(workspace->attention_out.data,
                                                                            workspace->attention_gate.data,
                                                                            Config::kAttentionSize);
    CUDA_LAUNCH();
    gpu_matvec(weights.out, workspace->attention_out.data, output);
}

void forward_token_cuda(const CudaModel& model, CudaState* state, int token, CudaWorkspace* workspace) {
    if (token < 0 || token >= Config::kVocabSize) fail("token id outside model vocabulary");
    if (state->position >= state->max_tokens) fail("CUDA context limit reached");
    embedding_kernel<<<(Config::kHiddenSize + 127) / 128, 128>>>(model.embedding().data, token, Config::kHiddenSize,
                                                                    workspace->hidden.data);
    CUDA_LAUNCH();
    for (int layer_index = 0; layer_index < Config::kLayers; ++layer_index) {
        const CudaLayer& layer = model.layer(layer_index);
        gpu_rmsnorm(workspace->hidden.data, layer.input_norm, workspace->normalized.data);
        if (layer.is_deltanet) {
            gpu_deltanet(layer.delta, &state->conv[layer_index], &state->recurrent[layer_index], workspace->normalized.data,
                         workspace, workspace->mixer.data);
        } else {
            gpu_attention(layer.attention, &state->keys[layer_index], &state->values[layer_index], state->position,
                          workspace->normalized.data, workspace, workspace->mixer.data);
        }
        add_kernel<<<(Config::kHiddenSize + 127) / 128, 128>>>(workspace->hidden.data, workspace->mixer.data,
                                                                 Config::kHiddenSize);
        CUDA_LAUNCH();
        gpu_rmsnorm(workspace->hidden.data, layer.post_attention_norm, workspace->normalized.data);
        gpu_mlp(layer.mlp, workspace->normalized.data, workspace, workspace->mixer.data);
        add_kernel<<<(Config::kHiddenSize + 127) / 128, 128>>>(workspace->hidden.data, workspace->mixer.data,
                                                                 Config::kHiddenSize);
        CUDA_LAUNCH();
    }
    gpu_rmsnorm(workspace->hidden.data, model.final_norm(), workspace->normalized.data);
    matvec_bf16_kernel<<<Config::kVocabSize, 128>>>(model.lm_head().data, workspace->normalized.data,
                                                     Config::kVocabSize, Config::kHiddenSize, workspace->logits.data);
    CUDA_LAUNCH();
    ++state->position;
}

std::vector<float> run_cuda_tokens(const char* checkpoint_directory, const std::vector<int>& tokens) {
    require(!tokens.empty(), "CUDA forward requires at least one token");
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    std::printf("CUDA backend: %s, %.1f GiB total memory\n", properties.name,
                static_cast<double>(properties.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
    TextModel host_model(checkpoint_directory);
    CudaModel model(host_model);
    CudaState state(static_cast<int>(tokens.size()) + 128);
    CudaWorkspace workspace;
    for (int token : tokens) forward_token_cuda(model, &state, token, &workspace);
    std::vector<float> logits(Config::kVocabSize);
    CUDA_CHECK(cudaMemcpy(logits.data(), workspace.logits.data, logits.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return logits;
}

void run_cuda_forward(const char* checkpoint_directory, const char* token_string) {
    const std::vector<int> tokens = parse_token_ids(token_string);
    const std::vector<float> logits = run_cuda_tokens(checkpoint_directory, tokens);
    const int next = argmax(logits);
    std::printf("cuda forward: %zu input token(s), next token id %d, logit %.6f\n", tokens.size(), next, logits[next]);
}

void run_cuda_compare(const char* checkpoint_directory, const char* token_string) {
    const std::vector<int> tokens = parse_token_ids(token_string);
    TextModel cpu_model(checkpoint_directory);
    RuntimeState cpu_state;
    ForwardWorkspace cpu_workspace;
    for (int token : tokens) forward_token_cpu(cpu_model, &cpu_state, token, &cpu_workspace);
    const std::vector<float> gpu_logits = run_cuda_tokens(checkpoint_directory, tokens);
    float max_abs_error = 0.0f;
    for (int i = 0; i < Config::kVocabSize; ++i) {
        max_abs_error = std::max(max_abs_error, std::fabs(cpu_workspace.logits[i] - gpu_logits[i]));
    }
    std::printf("cuda compare: CPU next %d, CUDA next %d, logits max_abs_error %.8f\n", argmax(cpu_workspace.logits),
                argmax(gpu_logits), max_abs_error);
    require(argmax(cpu_workspace.logits) == argmax(gpu_logits), "CUDA next token differs from CPU reference");
#ifdef QWEN38_TINY_MODEL
    constexpr float kCudaLogitsTolerance = 2.0e-4f;
#else
    // The real 0.8B path accumulates each BF16 GEMV in a different reduction
    // order from scalar CPU. The stricter 1e-3 bound is still tighter than
    // the independently validated CPU-vs-Transformers trace bound.
    constexpr float kCudaLogitsTolerance = 1.0e-3f;
#endif
    require(max_abs_error < kCudaLogitsTolerance, "CUDA logits differ from CPU reference");
}

void print_cuda_usage(const char* program) {
    std::printf("usage: %s [--self-test|--describe|--make-fake <empty-directory>|--inspect <checkpoint-dir>]\n", program);
    std::printf("       %s [--forward <checkpoint-dir> <ids>|--generate <checkpoint-dir> <ids> <count> [sampling options]]\n", program);
    std::printf("       %s [--cuda-forward <checkpoint-dir> <ids>|--cuda-compare <checkpoint-dir> <ids>]\n", program);
}

}  // namespace
}  // namespace qwen38

int main(int argc, char** argv) {
    if (argc == 1 || std::strcmp(argv[1], "--self-test") == 0) {
        qwen38::run_self_test();
        return 0;
    }
    if (std::strcmp(argv[1], "--describe") == 0) {
        qwen38::print_description();
        return 0;
    }
    if (std::strcmp(argv[1], "--make-fake") == 0) {
        if (argc != 3) qwen38::fail("--make-fake requires an empty output directory");
        qwen38::make_fake_checkpoint(argv[2]);
        return 0;
    }
    if (std::strcmp(argv[1], "--inspect") == 0) {
        if (argc != 3) qwen38::fail("--inspect requires a checkpoint directory");
        qwen38::inspect_checkpoint(argv[2]);
        return 0;
    }
    if (std::strcmp(argv[1], "--forward") == 0) {
        if (argc != 4) qwen38::fail("--forward requires a checkpoint directory and token ids");
        qwen38::run_forward(argv[2], argv[3]);
        return 0;
    }
    if (std::strcmp(argv[1], "--generate") == 0) {
        if (argc < 5) qwen38::fail("--generate requires a checkpoint directory, token ids, and count");
        qwen38::run_generate(argv[2], argv[3], qwen38::parse_positive_int(argv[4], "--generate"),
                              qwen38::parse_sampling_options(argc, argv, 5));
        return 0;
    }
    if (std::strcmp(argv[1], "--cuda-forward") == 0) {
        if (argc != 4) qwen38::fail("--cuda-forward requires a checkpoint directory and token ids");
        qwen38::run_cuda_forward(argv[2], argv[3]);
        return 0;
    }
    if (std::strcmp(argv[1], "--cuda-compare") == 0) {
        if (argc != 4) qwen38::fail("--cuda-compare requires a checkpoint directory and token ids");
        qwen38::run_cuda_compare(argv[2], argv[3]);
        return 0;
    }
    qwen38::print_cuda_usage(argv[0]);
    return 1;
}

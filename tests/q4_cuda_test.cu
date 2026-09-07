#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../arch/cuda/engine.cu"

namespace {

void fill_constant(std::vector<q3x_q4::Block>& blocks, int experts,
                   int rows, const uint16_t* scales) {
    blocks.resize(static_cast<size_t>(experts) * rows);
    for (int expert = 0; expert < experts; ++expert) {
        for (int row = 0; row < rows; ++row) {
            auto& block = blocks[static_cast<size_t>(expert) * rows + row];
            block.scale = scales[expert];
            std::memset(block.values, 0x99, sizeof(block.values));
        }
    }
}

template <typename T>
T* device_copy(const T* source, size_t count) {
    T* result = nullptr;
    CUDA_OK(cudaMalloc(&result, count * sizeof(T)));
    CUDA_OK(cudaMemcpy(result, source, count * sizeof(T), cudaMemcpyHostToDevice));
    return result;
}

void q4_mv_test() {
    q3x_q4::Block host[2] {};
    host[0].scale = 0x3c00;
    host[1].scale = 0x3800;
    for (int i = 0; i < 16; ++i) {
        host[0].values[i] = static_cast<uint8_t>(i | ((15 - i) << 4));
        host[1].values[i] = static_cast<uint8_t>((15 - i) | (i << 4));
    }
    float input[32];
    for (int i = 0; i < 32; ++i) input[i] = (i % 9 - 4) * 0.25f;
    float expected[2] {};
    for (int row = 0; row < 2; ++row) {
        const float scale = row == 0 ? 1.0f : 0.5f;
        for (int i = 0; i < 16; ++i) {
            expected[row] += scale * ((host[row].values[i] & 15) - 8) * input[i];
            expected[row] += scale * ((host[row].values[i] >> 4) - 8) * input[i + 16];
        }
    }
    auto* weight = device_copy(host, 2);
    auto* device_input = device_copy(input, 32);
    float* device_output = nullptr;
    CUDA_OK(cudaMalloc(&device_output, 2 * sizeof(float)));
    qwen3x_cuda::mv_q4_kernel<<<2, qwen3x_cuda::BLOCK>>>(
        weight, device_input, device_output, 2, 32);
    float output[2] {};
    CUDA_OK(cudaMemcpy(output, device_output, sizeof(output), cudaMemcpyDeviceToHost));
    assert(std::abs(output[0] - expected[0]) < 1e-5f);
    assert(std::abs(output[1] - expected[1]) < 1e-5f);
    cudaFree(weight); cudaFree(device_input); cudaFree(device_output);
}

void expert_moe_test() {
    constexpr int EXPERTS = 4, TOP_K = 2, H = 32, I = 32;
    const uint16_t scales[EXPERTS] = {0x3400, 0x3800, 0x3c00, 0x4000};
    std::vector<q3x_q4::Block> host_gate, host_down;
    fill_constant(host_gate, EXPERTS, 2 * I, scales);
    fill_constant(host_down, EXPERTS, H, scales);
    auto* gate = device_copy(host_gate.data(), host_gate.size());
    auto* down = device_copy(host_down.data(), host_down.size());
    float input[H];
    for (int i = 0; i < H; ++i) input[i] = 0.03125f * (i + 1);
    const int ids[TOP_K] = {2, 1};
    const float routing[TOP_K] = {0.75f, 0.25f};
    auto* device_input = device_copy(input, H);
    auto* device_ids = device_copy(ids, TOP_K);
    auto* device_routing = device_copy(routing, TOP_K);
    float *gate_up = nullptr, *hidden = nullptr, *output = nullptr;
    CUDA_OK(cudaMalloc(&gate_up, TOP_K * 2 * I * sizeof(float)));
    CUDA_OK(cudaMalloc(&hidden, TOP_K * I * sizeof(float)));
    CUDA_OK(cudaMalloc(&output, H * sizeof(float)));
    CUDA_OK(cudaMemset(output, 0, H * sizeof(float)));

    qwen3x_cuda::expert_gate_up_q4_kernel<<<TOP_K * 2 * I, qwen3x_cuda::BLOCK>>>(
        gate, device_input, gate_up, device_ids,
        EXPERTS, 2 * I, H, TOP_K);
    qwen3x_cuda::expert_swiglu_kernel<<<1, qwen3x_cuda::BLOCK>>>(
        gate_up, hidden, I, TOP_K);
    qwen3x_cuda::expert_down_weighted_q4_kernel<<<H, qwen3x_cuda::BLOCK>>>(
        down, hidden, device_ids, device_routing, output,
        EXPERTS, H, I, TOP_K);

    float actual[H];
    CUDA_OK(cudaMemcpy(actual, output, sizeof(actual), cudaMemcpyDeviceToHost));
    const float input_sum = 16.5f;
    const float gate2 = input_sum;
    const float gate1 = input_sum * 0.5f;
    const float hidden2 = gate2 / (1.0f + std::exp(-gate2)) * gate2;
    const float hidden1 = gate1 / (1.0f + std::exp(-gate1)) * gate1;
    const float expected = routing[0] * I * hidden2 +
                           routing[1] * 0.5f * I * hidden1;
    for (int i = 0; i < H; ++i) {
        assert(std::isfinite(actual[i]));
        assert(std::abs(actual[i] - expected) < 1e-2f);
    }
    cudaFree(gate); cudaFree(down); cudaFree(device_input); cudaFree(device_ids);
    cudaFree(device_routing); cudaFree(gate_up); cudaFree(hidden); cudaFree(output);
}

void router_test() {
    float logits[256];
    for (int expert = 0; expert < 256; ++expert)
        logits[expert] = expert * 0.03125f;
    auto* device_logits = device_copy(logits, 256);
    float *probabilities = nullptr, *weights = nullptr;
    int* ids = nullptr;
    CUDA_OK(cudaMalloc(&probabilities, 256 * sizeof(float)));
    CUDA_OK(cudaMalloc(&weights, 8 * sizeof(float)));
    CUDA_OK(cudaMalloc(&ids, 8 * sizeof(int)));
    qwen3x_cuda::router_top_k_kernel<<<1, 1>>>(
        device_logits, probabilities, ids, weights, 256, 8);
    int host_ids[8]; float host_weights[8];
    CUDA_OK(cudaMemcpy(host_ids, ids, sizeof(host_ids), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(host_weights, weights, sizeof(host_weights), cudaMemcpyDeviceToHost));
    float sum = 0.0f;
    for (int slot = 0; slot < 8; ++slot) {
        assert(host_ids[slot] == 255 - slot);
        sum += host_weights[slot];
    }
    assert(std::abs(sum - 1.0f) < 1e-6f);
    cudaFree(device_logits); cudaFree(probabilities); cudaFree(weights); cudaFree(ids);
}

void shared_scaled_add_test() {
    constexpr int N = 32;
    float output[N], shared[N], gate[1] = {0.75f};
    for (int i = 0; i < N; ++i) {
        output[i] = i * 0.125f;
        shared[i] = (i - 11) * 0.25f;
    }
    auto* device_output = device_copy(output, N);
    auto* device_shared = device_copy(shared, N);
    auto* device_gate = device_copy(gate, 1);
    qwen3x_cuda::shared_scaled_add_kernel<<<1, qwen3x_cuda::BLOCK>>>(
        device_output, device_shared, device_gate, N);
    float actual[N];
    CUDA_OK(cudaMemcpy(actual, device_output, sizeof(actual), cudaMemcpyDeviceToHost));
    const float scale = 1.0f / (1.0f + std::exp(-gate[0]));
    for (int i = 0; i < N; ++i)
        assert(std::abs(actual[i] - (output[i] + scale * shared[i])) < 1e-6f);
    cudaFree(device_output); cudaFree(device_shared); cudaFree(device_gate);
}

}  // namespace

int main() {
    q4_mv_test();
    router_test();
    expert_moe_test();
    shared_scaled_add_test();
    std::puts("q4-cuda-test: ok");
}

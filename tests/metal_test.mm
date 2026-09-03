// No model download: nonzero, deterministic tiny Qwen-shaped models exercise
// every Metal kernel and compare complete forward/state against the CPU engine.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "internal.h"

#define q35_backend q35_cpu_backend
#include "../engine.cpp"
#undef q35_backend
#include "../arch/metal/engine.mm"

namespace cpu = q35_cpu_backend;
namespace gpu = q35_backend;

struct Fixture {
    std::vector<uint8_t> bytes;
    uint32_t random = 42;
    Fixture() { bytes.reserve(32 * 1024 * 1024); }
    int next() { random = random * 1664525u + 1013904223u; return int(random >> 24) - 128; }
    size_t take(size_t n) {
        const size_t offset = (bytes.size() + 63) / 64 * 64;
        Q35_ASSERT(offset + n <= bytes.capacity(), "test fixture capacity");
        bytes.resize(offset + n); return offset;
    }
    size_t bf16(size_t n) {
        const size_t offset = take(n * 2);
        auto* out = reinterpret_cast<uint16_t*>(bytes.data() + offset);
        for (size_t i = 0; i < n; ++i) {
            const float value = next() / 4096.0f;
            uint32_t bits; std::memcpy(&bits, &value, 4); out[i] = bits >> 16;
        }
        return offset;
    }
    size_t fp32(size_t n, float value) {
        const size_t offset = take(n * 4);
        auto* out = reinterpret_cast<float*>(bytes.data() + offset);
        for (size_t i = 0; i < n; ++i) out[i] = value;
        return offset;
    }
    size_t matrix(int rows, int cols, q35_model::MatrixType type) {
        if (type == q35_model::MATRIX_BF16) return bf16(size_t(rows) * cols);
        const size_t count = size_t(rows) * cols / 32;
        const size_t offset = take(count * sizeof(q35_q8::Block));
        auto* out = reinterpret_cast<q35_q8::Block*>(bytes.data() + offset);
        for (size_t i = 0; i < count; ++i) {
            // Different exact FP16 scales, signed values, and zero blocks.
            out[i].scale = i % 7 == 0 ? 0 : uint16_t(0x1000 + (i % 4) * 0x0400);
            for (int j = 0; j < 32; ++j) out[i].values[j] = static_cast<int8_t>(next());
        }
        return offset;
    }
    void linear(cpu::Linear& a, gpu::Linear& b, int rows, int cols, q35_model::MatrixType type) {
        const size_t offset = matrix(rows, cols, type);
        a = {bytes.data() + offset, rows, cols, type}; b = {offset, rows, cols, type};
    }
    void norm(const uint16_t*& a, size_t& b, size_t n) {
        b = bf16(n); a = reinterpret_cast<const uint16_t*>(bytes.data() + b);
    }
    void floats(const float*& a, size_t& b, size_t n, float value) {
        b = fp32(n, value); a = reinterpret_cast<const float*>(bytes.data() + b);
    }
    void build(cpu::Model& a, gpu::Model& b, const q35_model::ModelConfig& c) {
        a.config = b.config = &c;
        a.layer.reset(new cpu::Layer[c.N]); b.layer.reset(new gpu::Layer[c.N]);
        const int AS = c.AH * c.AD, KVW = c.KVH * c.AD, DO = c.VH * c.VD;
        const int DQKV = 2 * c.KH * c.KD + DO;
        linear(a.embedding, b.embedding, c.V, c.H, c.matrix_type);
        if (c.tied_embeddings) { a.lm_head = a.embedding; b.lm_head = b.embedding; }
        else linear(a.lm_head, b.lm_head, c.V, c.H, c.matrix_type);
        norm(a.final_norm, b.final_norm, c.H);
        for (int i = 0; i < c.N; ++i) {
            auto& x = a.layer[i]; auto& y = b.layer[i];
            norm(x.input_norm, y.input_norm, c.H);
            if (i % c.AI != c.AI - 1) {
                linear(x.delta.qkv, y.delta.qkv, DQKV, c.H, c.matrix_type);
                linear(x.delta.z, y.delta.z, DO, c.H, c.matrix_type);
                linear(x.delta.a, y.delta.a, c.VH, c.H, c.matrix_type);
                linear(x.delta.b, y.delta.b, c.VH, c.H, c.matrix_type);
                norm(x.delta.conv, y.delta.conv, DQKV * c.CK);
                floats(x.delta.alog, y.delta.alog, c.VH, -2.0f);
                norm(x.delta.dt, y.delta.dt, c.VH);
                floats(x.delta.norm, y.delta.norm, c.VD, 1.0f);
                linear(x.delta.out, y.delta.out, c.H, DO, c.matrix_type);
            } else {
                linear(x.attention.q, y.attention.q, 2 * AS, c.H, c.matrix_type);
                linear(x.attention.k, y.attention.k, KVW, c.H, c.matrix_type);
                linear(x.attention.v, y.attention.v, KVW, c.H, c.matrix_type);
                norm(x.attention.qnorm, y.attention.qnorm, c.AD);
                norm(x.attention.knorm, y.attention.knorm, c.AD);
                linear(x.attention.out, y.attention.out, c.H, AS, c.matrix_type);
            }
            norm(x.post_norm, y.post_norm, c.H);
            linear(x.gate, y.gate, c.I, c.H, c.matrix_type);
            linear(x.up, y.up, c.I, c.H, c.matrix_type);
            linear(x.down, y.down, c.H, c.I, c.matrix_type);
        }
        b.weights = [b.device newBufferWithBytes:bytes.data() length:bytes.size() options:MTLResourceStorageModeShared];
        Q35_ASSERT(b.weights, "test weight allocation");
        for (int i = 0; i < c.N; ++i) b.layer[i].weights = b.weights;
    }
};

void compare(const float* a, const float* b, size_t n, const char* name) {
    float maximum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float error = std::fabs(a[i] - b[i]);
        maximum = std::max(maximum, error);
        Q35_ASSERT(std::isfinite(a[i]) && std::isfinite(b[i]) && error <= 2e-4f + 2e-4f * std::fabs(a[i]),
                   "%s index=%zu cpu=%g metal=%g error=%g", name, i, a[i], b[i], error);
    }
    std::printf("  %s: max_abs_error=%g\n", name, maximum);
}
void compare_state(const cpu::State& a, const gpu::State& b) {
    Q35_ASSERT(a.position == b.position, "position mismatch");
    compare(a.work.logits, b.work.storage.data(b.work.logits), a.config->V, "logits");
    compare(a.recurrent.data.get(), b.recurrent.data(), a.recurrent.count, "recurrent");
    const size_t used = size_t(a.position) * a.config->KVH * a.config->AD;
    for (int i = a.config->AI - 1; i < a.config->N; i += a.config->AI) {
        compare(a.layer[i].key, b.kv.data(b.layer[i].key), used, "key cache");
        compare(a.layer[i].value, b.kv.data(b.layer[i].value), used, "value cache");
    }
}
void check_delta_decay(gpu::Model& model) {
    constexpr int heads = 4, KD = 128, VD = 128;
    std::vector<float> q(heads * KD), k(heads * KD), v(heads * VD);
    std::vector<float> memory(heads * KD * VD, 1.0f), expected(heads * VD);
    const float a[] = {-20.0f, -14.0f, 0.0f, 30.0f}, b[heads] = {};
    const float alog[] = {16.0f, 10.0f, -2.0f, -5.0f};
    const uint16_t dt[heads] = {};
    for (int h = 0; h < heads; ++h) q[h * KD] = 1.0f;
    auto buffer = [&](const void* data, size_t bytes) {
        id<MTLBuffer> result = [model.device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
        Q35_ASSERT(result, "delta decay test allocation"); return result;
    };
    id<MTLBuffer> qb = buffer(q.data(), q.size() * 4), kb = buffer(k.data(), k.size() * 4);
    id<MTLBuffer> vb = buffer(v.data(), v.size() * 4), ab = buffer(a, sizeof(a)), bb = buffer(b, sizeof(b));
    id<MTLBuffer> alb = buffer(alog, sizeof(alog)), dtb = buffer(dt, sizeof(dt));
    id<MTLBuffer> state = buffer(memory.data(), memory.size() * 4), out = buffer(expected.data(), expected.size() * 4);
    id<MTLCommandQueue> queue = [model.device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
    Q35_ASSERT(queue && command && enc, "delta decay test command allocation");
    [enc setBuffer:qb offset:0 atIndex:0]; [enc setBuffer:kb offset:0 atIndex:1];
    [enc setBuffer:vb offset:0 atIndex:2]; [enc setBuffer:ab offset:0 atIndex:3];
    [enc setBuffer:bb offset:0 atIndex:4]; [enc setBuffer:alb offset:0 atIndex:5];
    [enc setBuffer:dtb offset:0 atIndex:6]; [enc setBuffer:state offset:0 atIndex:7];
    [enc setBuffer:out offset:0 atIndex:8];
    gpu::launch(enc, model.kernels.delta_rule, heads);
    [enc endEncoding]; [command commit]; [command waitUntilCompleted];
    Q35_ASSERT(command.status == MTLCommandBufferStatusCompleted, "delta decay test execution failed");
    for (int h = 0; h < heads; ++h)
        cpu::delta_rule(q.data() + h * KD, k.data() + h * KD, v.data() + h * VD,
                        -std::exp(alog[h]) * cpu::softplus(a[h]), 0.5f,
                        memory.data() + h * KD * VD, expected.data() + h * VD, KD, VD);
    compare(expected.data(), static_cast<const float*>(out.contents), expected.size(), "softplus/Delta decay tails");
    compare(memory.data(), static_cast<const float*>(state.contents), memory.size(), "Delta decay state");
}
void run(id<MTLDevice> device, q35_model::MatrixType type) {
    const q35_model::ModelConfig config = {
        800, "synthetic-metal-smoke", 64, 96, 160, 4, 4, 8, 2, 256, 64,
        16, type == q35_model::MATRIX_BF16 ? 16 : 32, 128, 128, 4,
        type, type == q35_model::MATRIX_BF16,
    };
    Fixture fixture;
    cpu::Model a; gpu::Model b; b.device = device;
    const char* error = nullptr;
    Q35_ASSERT(b.kernels.load(device, &error), "Metal pipelines: %s", error);
    if (type == q35_model::MATRIX_BF16) check_delta_decay(b);
    fixture.build(a, b, config);
    cpu::State ac(config, 17); gpu::State bc(b, 17);
    std::printf("Metal %s complete forward/state\n", type == q35_model::MATRIX_BF16 ? "BF16" : "Q8_0");
    const int tokens[] = {1, 7, 3, 16, 2, 5, 9, 11, 23, 4};
    for (int i = 0; i < 3; ++i) {
        cpu::state_forward(&a, &ac, tokens + i, 1, true);
        gpu::state_forward(&b, &bc, tokens + i, 1, true);
        compare_state(ac, bc);
    }
    cpu::state_checkpoint_save(&ac); gpu::state_checkpoint_save(&bc);
    cpu::state_forward(&a, &ac, tokens + 3, 7, true);
    gpu::state_forward(&b, &bc, tokens + 3, 7, true);
    compare_state(ac, bc);
    cpu::state_checkpoint_restore(&ac); gpu::state_checkpoint_restore(&bc);
    compare_state(ac, bc);
    const int branch[] = {13, 17, 19, 21, 22, 24, 26, 28, 29};
    cpu::state_forward(&a, &ac, branch, 9, true); // crosses Metal's 8-token encode boundary
    gpu::state_forward(&b, &bc, branch, 9, true);
    compare_state(ac, bc);
    cpu::state_reset(&ac); gpu::state_reset(&bc);
    cpu::state_forward(&a, &ac, tokens, 10, true);
    gpu::state_forward(&b, &bc, tokens, 10, true);
    compare_state(ac, bc);
    Q35_ASSERT(cpu::state_argmax(&ac) == gpu::state_argmax(&bc), "argmax mismatch");
}

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || !device.hasUnifiedMemory || ![device supportsFamily:MTLGPUFamilyApple7]) {
            std::fprintf(stderr, "metal-test: NOT RUN (Apple Silicon Metal GPU unavailable)\n");
            return 77;
        }
        std::printf("Metal device: %s\n", device.name.UTF8String);
        run(device, q35_model::MATRIX_BF16);
        run(device, q35_model::MATRIX_Q8_0);
        std::puts("metal-test: ok");
    }
}

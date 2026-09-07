// Qwen Metal engine. This platform boundary uses Objective-C++ and system
// frameworks only. Model owns packed weights; State owns shared GPU buffers.
// Decode uses the readable single-token forward. Q4 prefill keeps recurrent
// operations token-ordered while batching its matrix projections.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdio>
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
#include "q4.h"
#include "q8.h"
#include "kernels_metallib.h"

#if !defined(__arm64__)
#error "The Metal backend targets Apple Silicon Macs"
#endif

namespace q3x_backend {

constexpr int BLOCK = 256, ENCODE_TOKENS = 8, Q4_BATCH_TOKENS = 4;
struct Linear {
    size_t offset = 0;
    int rows = 0, cols = 0;
    q3x_model::MatrixType type = q3x_model::MATRIX_BF16;
};
struct DeltaWeights {
    Linear qkv, z, a, b, out;
    size_t conv = 0, alog = 0, dt = 0, norm = 0;
};
struct AttentionWeights { Linear q, k, v, out; size_t qnorm = 0, knorm = 0; };
struct Layer {
    id<MTLBuffer> weights;
    size_t input_norm = 0, post_norm = 0;
    Linear gate, up, down;
    DeltaWeights delta;
    AttentionWeights attention;
};
struct Kernels {
    id<MTLComputePipelineState> embed_bf16, embed_q8, embed_q4, batch_embed_q4;
    id<MTLComputePipelineState> mv_bf16, mv_q8, mv_q4, batch_mv_q4;
    id<MTLComputePipelineState> rms, swiglu, batch_rms, batch_swiglu;
    id<MTLComputePipelineState> conv, prepare_delta_qk, delta_rule, gated_rms;
    id<MTLComputePipelineState> batch_conv, batch_prepare_delta_qk;
    id<MTLComputePipelineState> batch_delta_rule, batch_gated_rms;
    id<MTLComputePipelineState> prepare_query, prepare_key, store_kv, attention;
    id<MTLComputePipelineState> batch_prepare_query, batch_prepare_key;
    id<MTLComputePipelineState> batch_store_kv, batch_attention;
    bool load(id<MTLDevice> device, const char** error) {
        dispatch_data_t data = dispatch_data_create(kernels_metallib, kernels_metallib_len,
                                                    nullptr, ^{});
        NSError* detail = nil;
        id<MTLLibrary> library = [device newLibraryWithData:data error:&detail];
        if (!library) { LOG_ERROR("Metal library: %s", detail.localizedDescription.UTF8String);
            *error = "cannot load embedded Metal library"; return false; }
        auto make = [&](const char* name) {
            NSError* pipeline_error = nil;
            id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
            id<MTLComputePipelineState> pipeline = function
                ? [device newComputePipelineStateWithFunction:function error:&pipeline_error] : nil;
            if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < BLOCK) {
                LOG_ERROR("Metal pipeline %s: %s", name,
                          pipeline_error ? pipeline_error.localizedDescription.UTF8String : "missing kernel or thread limit");
                *error = "cannot create Metal compute pipeline";
            }
            return pipeline;
        };
        embed_bf16 = make("embed_bf16"); embed_q8 = make("embed_q8");
        embed_q4 = make("embed_q4");
        batch_embed_q4 = make("batch_embed_q4");
        mv_bf16 = make("mv_bf16"); mv_q8 = make("mv_q8"); mv_q4 = make("mv_q4");
        batch_mv_q4 = make("batch_mv_q4");
        rms = make("rms"); swiglu = make("swiglu");
        batch_rms = make("batch_rms"); batch_swiglu = make("batch_swiglu");
        conv = make("conv");
        prepare_delta_qk = make("prepare_delta_qk"); delta_rule = make("delta_rule");
        gated_rms = make("gated_rms"); prepare_query = make("prepare_query");
        batch_conv = make("batch_conv");
        batch_prepare_delta_qk = make("batch_prepare_delta_qk");
        batch_delta_rule = make("batch_delta_rule");
        batch_gated_rms = make("batch_gated_rms");
        prepare_key = make("prepare_key"); store_kv = make("store_kv"); attention = make("attention");
        batch_prepare_query = make("batch_prepare_query");
        batch_prepare_key = make("batch_prepare_key");
        batch_store_kv = make("batch_store_kv");
        batch_attention = make("batch_attention");
        return !*error;
    }
};
struct Model {
    const q3x_model::ModelConfig* config = nullptr;
    id<MTLDevice> device;
    id<MTLBuffer> weights;
    Kernels kernels;
    Linear embedding, lm_head;
    size_t final_norm = 0;
    std::unique_ptr<Layer[]> layer;
    bool load(const char* path, const char** error);
};

// All offsets are bytes. Shared buffers have one owner and are cut once at
// creation; no activation or Q8-expanded weight allocation occurs in forward.
struct Storage {
    id<MTLBuffer> buffer;
    size_t count = 0, used = 0;
    void allocate(id<MTLDevice> device, size_t n) {
        Q3X_ASSERT(n <= device.maxBufferLength / sizeof(float), "Metal buffer too large count=%zu", n);
        buffer = [device newBufferWithLength:n * sizeof(float) options:MTLResourceStorageModeShared];
        Q3X_ASSERT(buffer, "Metal allocation failed count=%zu", n);
        count = n;
    }
    size_t take(size_t n) {
        Q3X_ASSERT(used <= count && n <= count - used, "Metal storage used=%zu take=%zu count=%zu", used, n, count);
        const size_t offset = used * sizeof(float); used += n; return offset;
    }
    float* data(size_t offset = 0) const {
        return reinterpret_cast<float*>(static_cast<uint8_t*>(buffer.contents) + offset);
    }
};
struct Work {
    Storage storage;
    size_t hidden, normalized, logits, ffn_gate, ffn_up;
    size_t delta_qkv, delta_z, delta_a, delta_b, delta_q, delta_k, delta_output;
    size_t query_and_gate, query, attention_gate, key, value, attention_output;
    Work(id<MTLDevice> device, const q3x_model::ModelConfig& c) {
        const int DO = c.VH * c.VD, DQKV = 2 * c.KH * c.KD + DO;
        const int AS = c.AH * c.AD, KVW = c.KVH * c.AD;
        storage.allocate(device, 2ull * c.H + c.V + 2ull * c.I + DQKV + 4ull * DO +
                                 2ull * c.VH + 5ull * AS + 2ull * KVW);
        hidden = storage.take(c.H); normalized = storage.take(c.H); logits = storage.take(c.V);
        ffn_gate = storage.take(c.I); ffn_up = storage.take(c.I);
        delta_qkv = storage.take(DQKV); delta_z = storage.take(DO);
        delta_a = storage.take(c.VH); delta_b = storage.take(c.VH);
        delta_q = storage.take(DO); delta_k = storage.take(DO); delta_output = storage.take(DO);
        query_and_gate = storage.take(2 * AS); query = storage.take(AS);
        attention_gate = storage.take(AS); key = storage.take(KVW); value = storage.take(KVW);
        attention_output = storage.take(AS);
        Q3X_ASSERT(storage.used == storage.count, "Metal Work layout mismatch");
    }
};
struct BatchWork {
    Storage storage;
    size_t hidden, normalized, ffn_gate, ffn_up;
    size_t delta_qkv, delta_z, delta_a, delta_b, delta_q, delta_k, delta_output;
    size_t query_and_gate, query, attention_gate, key, value, attention_output;
    BatchWork(id<MTLDevice> device, const q3x_model::ModelConfig& c) {
        const int DO = c.VH * c.VD, DQKV = 2 * c.KH * c.KD + DO;
        const int AS = c.AH * c.AD, KVW = c.KVH * c.AD;
        const size_t stride = 2ull * c.H + 2ull * c.I + DQKV + 4ull * DO +
                              2ull * c.VH + 5ull * AS + 2ull * KVW;
        storage.allocate(device, Q4_BATCH_TOKENS * stride);
        auto take = [&](size_t width) {
            return storage.take(Q4_BATCH_TOKENS * width);
        };
        hidden = take(c.H); normalized = take(c.H);
        ffn_gate = take(c.I); ffn_up = take(c.I);
        delta_qkv = take(DQKV); delta_z = take(DO);
        delta_a = take(c.VH); delta_b = take(c.VH);
        delta_q = take(DO); delta_k = take(DO); delta_output = take(DO);
        query_and_gate = take(2 * AS); query = take(AS);
        attention_gate = take(AS); key = take(KVW); value = take(KVW);
        attention_output = take(AS);
        Q3X_ASSERT(storage.used == storage.count, "Metal BatchWork layout mismatch");
    }
};
struct LayerState { size_t conv = 0, memory = 0, key = 0, value = 0; };
struct State {
    const q3x_model::ModelConfig* config;
    int position = 0, capacity, checkpoint_position = 0;
    id<MTLCommandQueue> queue;
    std::unique_ptr<LayerState[]> layer;
    Storage recurrent, kv, checkpoint;
    Work work;
    BatchWork batch;
    State(Model& model, int context) : config(model.config), capacity(context),
                                       work(model.device, *model.config),
                                       batch(model.device, *model.config) {
        const auto& c = *config;
        Q3X_ASSERT(c.AD == 256 && c.RD == 64 && c.KH == 16 && c.KD == 128 && c.VD == 128 && c.CK == 4,
                   "Metal fixed dimensions model=%s", c.name);
        queue = [model.device newCommandQueue];
        layer.reset(new (std::nothrow) LayerState[c.N]);
        Q3X_ASSERT(queue && layer, "Metal State allocation failed");
        const size_t conv = static_cast<size_t>(2 * c.KH * c.KD + c.VH * c.VD) * (c.CK - 1);
        const size_t memory = static_cast<size_t>(c.VH) * c.KD * c.VD;
        const size_t cache = static_cast<size_t>(context) * c.KVH * c.AD;
        recurrent.allocate(model.device, static_cast<size_t>(c.N - c.N / c.AI) * (conv + memory));
        kv.allocate(model.device, static_cast<size_t>(c.N / c.AI) * 2 * cache);
        checkpoint.allocate(model.device, recurrent.count + c.V);
        for (int i = 0; i < c.N; ++i) {
            if (i % c.AI != c.AI - 1) {
                layer[i].conv = recurrent.take(conv); layer[i].memory = recurrent.take(memory);
            } else { layer[i].key = kv.take(cache); layer[i].value = kv.take(cache); }
        }
        Q3X_ASSERT(recurrent.used == recurrent.count && kv.used == kv.count, "Metal State layout mismatch");
        std::memset(recurrent.data(), 0, recurrent.count * sizeof(float));
        std::memset(work.storage.data(work.logits), 0, c.V * sizeof(float));
        const size_t state_bytes =
            (recurrent.count + kv.count + checkpoint.count +
             work.storage.count + batch.storage.count) * sizeof(float);
        LOG_INFO("Metal state model=%s context=%d bytes=%zu allocated=%llu",
                 c.name, context, state_bytes,
                 static_cast<unsigned long long>(model.device.currentAllocatedSize));
    }
};

void launch_grid(id<MTLComputeCommandEncoder> enc,
                 id<MTLComputePipelineState> pipeline,
                 MTLSize groups, MTLSize threads) {
    [enc setComputePipelineState:pipeline];
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    // State/Work intentionally reuse buffers. Make dispatch-to-dispatch RAW/WAR
    // dependencies explicit, including read/write aliases within one buffer.
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
}
void launch(id<MTLComputeCommandEncoder> enc,
            id<MTLComputePipelineState> pipeline, int groups,
            int threads = BLOCK) {
    launch_grid(enc, pipeline, MTLSizeMake(groups, 1, 1),
                MTLSizeMake(threads, 1, 1));
}
void embed(id<MTLComputeCommandEncoder> enc, const Model& model, State& state, int token) {
    const Linear& w = model.embedding;
    const uint32_t p[] = {static_cast<uint32_t>(w.cols), static_cast<uint32_t>(token)};
    [enc setBuffer:model.weights offset:w.offset atIndex:0];
    [enc setBuffer:state.work.storage.buffer offset:state.work.hidden atIndex:1];
    [enc setBytes:p length:sizeof(p) atIndex:2];
    switch (w.type) {
    case q3x_model::MATRIX_BF16: launch(enc, model.kernels.embed_bf16, (w.cols + BLOCK - 1) / BLOCK); break;
    case q3x_model::MATRIX_Q8_0: launch(enc, model.kernels.embed_q8, (w.cols + BLOCK - 1) / BLOCK); break;
    case q3x_model::MATRIX_Q4_0: launch(enc, model.kernels.embed_q4, (w.cols + BLOCK - 1) / BLOCK); break;
    }
}
void mv(id<MTLComputeCommandEncoder> enc, const Model& model, id<MTLBuffer> weights, const Linear& w,
        Work& work, size_t input, size_t output, bool add = false) {
    const uint32_t p[] = {static_cast<uint32_t>(w.cols), add ? 1u : 0u};
    [enc setBuffer:weights offset:w.offset atIndex:0];
    [enc setBuffer:work.storage.buffer offset:input atIndex:1];
    [enc setBuffer:work.storage.buffer offset:output atIndex:2];
    [enc setBytes:p length:sizeof(p) atIndex:3];
    switch (w.type) {
    case q3x_model::MATRIX_BF16: launch(enc, model.kernels.mv_bf16, w.rows); break;
    case q3x_model::MATRIX_Q8_0: launch(enc, model.kernels.mv_q8, w.rows); break;
    case q3x_model::MATRIX_Q4_0:
        launch(enc, model.kernels.mv_q4, w.rows, 32);
        break;
    }
}
void rms(id<MTLComputeCommandEncoder> enc, const Model& model, id<MTLBuffer> weights,
         Work& work, size_t weight) {
    const uint32_t n = model.config->H;
    [enc setBuffer:work.storage.buffer offset:work.hidden atIndex:0];
    [enc setBuffer:weights offset:weight atIndex:1];
    [enc setBuffer:work.storage.buffer offset:work.normalized atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    launch(enc, model.kernels.rms, 1);
}
void deltanet(id<MTLComputeCommandEncoder> enc, const Model& model, State& state,
              const Layer& layer, const LayerState& history) {
    const DeltaWeights& w = layer.delta;
    Work& work = state.work;
    const auto& c = *model.config;
    const uint32_t vh = c.VH, dqkv = 2 * c.KH * c.KD + c.VH * c.VD;
    mv(enc, model, layer.weights, w.qkv, work, work.normalized, work.delta_qkv);
    mv(enc, model, layer.weights, w.z, work, work.normalized, work.delta_z);
    mv(enc, model, layer.weights, w.a, work, work.normalized, work.delta_a);
    mv(enc, model, layer.weights, w.b, work, work.normalized, work.delta_b);
    [enc setBuffer:work.storage.buffer offset:work.delta_qkv atIndex:0];
    [enc setBuffer:layer.weights offset:w.conv atIndex:1];
    [enc setBuffer:state.recurrent.buffer offset:history.conv atIndex:2];
    [enc setBytes:&dqkv length:sizeof(dqkv) atIndex:3];
    launch(enc, model.kernels.conv, (dqkv + BLOCK - 1) / BLOCK);
    [enc setBuffer:work.storage.buffer offset:work.delta_qkv atIndex:0];
    [enc setBuffer:work.storage.buffer offset:work.delta_q atIndex:1];
    [enc setBuffer:work.storage.buffer offset:work.delta_k atIndex:2];
    [enc setBytes:&vh length:sizeof(vh) atIndex:3];
    launch(enc, model.kernels.prepare_delta_qk, vh);
    [enc setBuffer:work.storage.buffer offset:work.delta_q atIndex:0];
    [enc setBuffer:work.storage.buffer offset:work.delta_k atIndex:1];
    [enc setBuffer:work.storage.buffer offset:work.delta_qkv + 2ull * c.KH * c.KD * sizeof(float) atIndex:2];
    [enc setBuffer:work.storage.buffer offset:work.delta_a atIndex:3];
    [enc setBuffer:work.storage.buffer offset:work.delta_b atIndex:4];
    [enc setBuffer:layer.weights offset:w.alog atIndex:5];
    [enc setBuffer:layer.weights offset:w.dt atIndex:6];
    [enc setBuffer:state.recurrent.buffer offset:history.memory atIndex:7];
    [enc setBuffer:work.storage.buffer offset:work.delta_output atIndex:8];
    launch(enc, model.kernels.delta_rule, vh);
    [enc setBuffer:work.storage.buffer offset:work.delta_output atIndex:0];
    [enc setBuffer:layer.weights offset:w.norm atIndex:1];
    [enc setBuffer:work.storage.buffer offset:work.delta_z atIndex:2];
    launch(enc, model.kernels.gated_rms, vh);
    mv(enc, model, layer.weights, w.out, work, work.delta_output, work.hidden, true);
}
void attention(id<MTLComputeCommandEncoder> enc, const Model& model, State& state,
                const Layer& layer, const LayerState& history) {
    const AttentionWeights& w = layer.attention;
    Work& work = state.work;
    const auto& c = *model.config;
    const uint32_t position = state.position, kvw = c.KVH * c.AD;
    mv(enc, model, layer.weights, w.q, work, work.normalized, work.query_and_gate);
    mv(enc, model, layer.weights, w.k, work, work.normalized, work.key);
    mv(enc, model, layer.weights, w.v, work, work.normalized, work.value);
    [enc setBuffer:work.storage.buffer offset:work.query_and_gate atIndex:0];
    [enc setBuffer:layer.weights offset:w.qnorm atIndex:1];
    [enc setBuffer:work.storage.buffer offset:work.query atIndex:2];
    [enc setBuffer:work.storage.buffer offset:work.attention_gate atIndex:3];
    [enc setBytes:&position length:sizeof(position) atIndex:4];
    launch(enc, model.kernels.prepare_query, c.AH);
    [enc setBuffer:work.storage.buffer offset:work.key atIndex:0];
    [enc setBuffer:layer.weights offset:w.knorm atIndex:1];
    [enc setBytes:&position length:sizeof(position) atIndex:2];
    launch(enc, model.kernels.prepare_key, c.KVH);
    const uint32_t cache[] = {kvw, position};
    [enc setBuffer:work.storage.buffer offset:work.key atIndex:0];
    [enc setBuffer:work.storage.buffer offset:work.value atIndex:1];
    [enc setBuffer:state.kv.buffer offset:history.key atIndex:2];
    [enc setBuffer:state.kv.buffer offset:history.value atIndex:3];
    [enc setBytes:cache length:sizeof(cache) atIndex:4];
    launch(enc, model.kernels.store_kv, (kvw + BLOCK - 1) / BLOCK);
    const uint32_t p[] = {static_cast<uint32_t>(c.AH), static_cast<uint32_t>(c.KVH), position, 0};
    [enc setBuffer:work.storage.buffer offset:work.query atIndex:0];
    [enc setBuffer:work.storage.buffer offset:work.attention_gate atIndex:1];
    [enc setBuffer:state.kv.buffer offset:history.key atIndex:2];
    [enc setBuffer:state.kv.buffer offset:history.value atIndex:3];
    [enc setBuffer:work.storage.buffer offset:work.attention_output atIndex:4];
    [enc setBytes:p length:sizeof(p) atIndex:5];
    launch(enc, model.kernels.attention, c.AH);
    mv(enc, model, layer.weights, w.out, work, work.attention_output, work.hidden, true);
}
void ffn(id<MTLComputeCommandEncoder> enc, const Model& model, Work& work, const Layer& layer) {
    mv(enc, model, layer.weights, layer.gate, work, work.normalized, work.ffn_gate);
    mv(enc, model, layer.weights, layer.up, work, work.normalized, work.ffn_up);
    const uint32_t n = model.config->I;
    [enc setBuffer:work.storage.buffer offset:work.ffn_gate atIndex:0];
    [enc setBuffer:work.storage.buffer offset:work.ffn_up atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    launch(enc, model.kernels.swiglu, (n + BLOCK - 1) / BLOCK);
    mv(enc, model, layer.weights, layer.down, work, work.ffn_gate, work.hidden, true);
}

size_t batch_at(size_t base, int token, int width) {
    return base + static_cast<size_t>(token) * width * sizeof(float);
}
void batch_embed_q4(id<MTLComputeCommandEncoder> enc, const Model& model,
                    BatchWork& work, const int* tokens, int count) {
    const Linear& w = model.embedding;
    const uint32_t p[] = {static_cast<uint32_t>(w.cols),
                          static_cast<uint32_t>(count)};
    [enc setBuffer:model.weights offset:w.offset atIndex:0];
    [enc setBuffer:work.storage.buffer offset:work.hidden atIndex:1];
    [enc setBytes:tokens length:static_cast<size_t>(count) * sizeof(int) atIndex:2];
    [enc setBytes:p length:sizeof(p) atIndex:3];
    launch(enc, model.kernels.batch_embed_q4,
           (count * w.cols + BLOCK - 1) / BLOCK);
}
void batch_mv_q4(id<MTLComputeCommandEncoder> enc, const Model& model,
                 id<MTLBuffer> weights, const Linear& w, BatchWork& work,
                 size_t input, size_t output, int count, bool add = false) {
    Q3X_ASSERT(w.type == q3x_model::MATRIX_Q4_0, "Metal batch matrix type=%d",
               static_cast<int>(w.type));
    const uint32_t p[] = {static_cast<uint32_t>(w.rows),
                          static_cast<uint32_t>(w.cols),
                          static_cast<uint32_t>(count), add ? 1u : 0u};
    [enc setBuffer:weights offset:w.offset atIndex:0];
    [enc setBuffer:work.storage.buffer offset:input atIndex:1];
    [enc setBuffer:work.storage.buffer offset:output atIndex:2];
    [enc setBytes:p length:sizeof(p) atIndex:3];
    launch_grid(enc, model.kernels.batch_mv_q4,
                MTLSizeMake(w.rows, (count + 3) / 4, 1),
                MTLSizeMake(32, 1, 1));
}
void batch_rms(id<MTLComputeCommandEncoder> enc, const Model& model,
               id<MTLBuffer> weights, size_t weight, BatchWork& work,
               size_t input, size_t output, int width, int count) {
    const uint32_t p[] = {static_cast<uint32_t>(width),
                          static_cast<uint32_t>(count)};
    [enc setBuffer:work.storage.buffer offset:input atIndex:0];
    [enc setBuffer:weights offset:weight atIndex:1];
    [enc setBuffer:work.storage.buffer offset:output atIndex:2];
    [enc setBytes:p length:sizeof(p) atIndex:3];
    launch(enc, model.kernels.batch_rms, count);
}
void batch_deltanet_q4(id<MTLComputeCommandEncoder> enc, const Model& model,
                       State& state, const Layer& layer,
                       const LayerState& history, int count) {
    const DeltaWeights& w = layer.delta;
    BatchWork& work = state.batch;
    const auto& c = *model.config;
    const int DO = c.VH * c.VD, DQKV = 2 * c.KH * c.KD + DO;
    batch_mv_q4(enc, model, layer.weights, w.qkv, work,
                work.normalized, work.delta_qkv, count);
    batch_mv_q4(enc, model, layer.weights, w.z, work,
                work.normalized, work.delta_z, count);
    batch_mv_q4(enc, model, layer.weights, w.a, work,
                work.normalized, work.delta_a, count);
    batch_mv_q4(enc, model, layer.weights, w.b, work,
                work.normalized, work.delta_b, count);
    const uint32_t conv[] = {static_cast<uint32_t>(count),
                             static_cast<uint32_t>(DQKV)};
    [enc setBuffer:work.storage.buffer offset:work.delta_qkv atIndex:0];
    [enc setBuffer:layer.weights offset:w.conv atIndex:1];
    [enc setBuffer:state.recurrent.buffer offset:history.conv atIndex:2];
    [enc setBytes:conv length:sizeof(conv) atIndex:3];
    launch(enc, model.kernels.batch_conv, (DQKV + BLOCK - 1) / BLOCK);
    const uint32_t p[] = {static_cast<uint32_t>(count),
                          static_cast<uint32_t>(DQKV),
                          static_cast<uint32_t>(c.VH), 0};
    [enc setBuffer:work.storage.buffer offset:work.delta_qkv atIndex:0];
    [enc setBuffer:work.storage.buffer offset:work.delta_q atIndex:1];
    [enc setBuffer:work.storage.buffer offset:work.delta_k atIndex:2];
    [enc setBytes:p length:sizeof(p) atIndex:3];
    launch_grid(enc, model.kernels.batch_prepare_delta_qk,
                MTLSizeMake(c.VH, count, 1), MTLSizeMake(BLOCK, 1, 1));
    [enc setBuffer:work.storage.buffer offset:work.delta_q atIndex:0];
    [enc setBuffer:work.storage.buffer offset:work.delta_k atIndex:1];
    [enc setBuffer:work.storage.buffer offset:work.delta_qkv atIndex:2];
    [enc setBuffer:work.storage.buffer offset:work.delta_a atIndex:3];
    [enc setBuffer:work.storage.buffer offset:work.delta_b atIndex:4];
    [enc setBuffer:layer.weights offset:w.alog atIndex:5];
    [enc setBuffer:layer.weights offset:w.dt atIndex:6];
    [enc setBuffer:state.recurrent.buffer offset:history.memory atIndex:7];
    [enc setBuffer:work.storage.buffer offset:work.delta_output atIndex:8];
    [enc setBytes:p length:sizeof(p) atIndex:9];
    launch(enc, model.kernels.batch_delta_rule, c.VH);
    const uint32_t gated[] = {static_cast<uint32_t>(count),
                              static_cast<uint32_t>(c.VH)};
    [enc setBuffer:work.storage.buffer offset:work.delta_output atIndex:0];
    [enc setBuffer:layer.weights offset:w.norm atIndex:1];
    [enc setBuffer:work.storage.buffer offset:work.delta_z atIndex:2];
    [enc setBytes:gated length:sizeof(gated) atIndex:3];
    launch_grid(enc, model.kernels.batch_gated_rms,
                MTLSizeMake(c.VH, count, 1), MTLSizeMake(BLOCK, 1, 1));
    batch_mv_q4(enc, model, layer.weights, w.out, work,
                work.delta_output, work.hidden, count, true);
}
void batch_attention_q4(id<MTLComputeCommandEncoder> enc, const Model& model,
                        State& state, const Layer& layer,
                        const LayerState& history, int count) {
    const AttentionWeights& w = layer.attention;
    BatchWork& work = state.batch;
    const auto& c = *model.config;
    const int KVW = c.KVH * c.AD;
    batch_mv_q4(enc, model, layer.weights, w.q, work,
                work.normalized, work.query_and_gate, count);
    batch_mv_q4(enc, model, layer.weights, w.k, work,
                work.normalized, work.key, count);
    batch_mv_q4(enc, model, layer.weights, w.v, work,
                work.normalized, work.value, count);
    const uint32_t query[] = {static_cast<uint32_t>(state.position),
                              static_cast<uint32_t>(count),
                              static_cast<uint32_t>(c.AH), 0};
    [enc setBuffer:work.storage.buffer offset:work.query_and_gate atIndex:0];
    [enc setBuffer:layer.weights offset:w.qnorm atIndex:1];
    [enc setBuffer:work.storage.buffer offset:work.query atIndex:2];
    [enc setBuffer:work.storage.buffer offset:work.attention_gate atIndex:3];
    [enc setBytes:query length:sizeof(query) atIndex:4];
    launch_grid(enc, model.kernels.batch_prepare_query,
                MTLSizeMake(c.AH, count, 1), MTLSizeMake(BLOCK, 1, 1));
    const uint32_t key[] = {static_cast<uint32_t>(state.position),
                            static_cast<uint32_t>(count),
                            static_cast<uint32_t>(c.KVH), 0};
    [enc setBuffer:work.storage.buffer offset:work.key atIndex:0];
    [enc setBuffer:layer.weights offset:w.knorm atIndex:1];
    [enc setBytes:key length:sizeof(key) atIndex:2];
    launch_grid(enc, model.kernels.batch_prepare_key,
                MTLSizeMake(c.KVH, count, 1), MTLSizeMake(BLOCK, 1, 1));
    const uint32_t cache[] = {static_cast<uint32_t>(state.position),
                              static_cast<uint32_t>(count),
                              static_cast<uint32_t>(KVW), 0};
    [enc setBuffer:work.storage.buffer offset:work.key atIndex:0];
    [enc setBuffer:work.storage.buffer offset:work.value atIndex:1];
    [enc setBuffer:state.kv.buffer offset:history.key atIndex:2];
    [enc setBuffer:state.kv.buffer offset:history.value atIndex:3];
    [enc setBytes:cache length:sizeof(cache) atIndex:4];
    launch(enc, model.kernels.batch_store_kv,
           (count * KVW + BLOCK - 1) / BLOCK);
    const uint32_t p[] = {static_cast<uint32_t>(state.position),
                          static_cast<uint32_t>(count),
                          static_cast<uint32_t>(c.AH),
                          static_cast<uint32_t>(c.KVH)};
    [enc setBuffer:work.storage.buffer offset:work.query atIndex:0];
    [enc setBuffer:work.storage.buffer offset:work.attention_gate atIndex:1];
    [enc setBuffer:state.kv.buffer offset:history.key atIndex:2];
    [enc setBuffer:state.kv.buffer offset:history.value atIndex:3];
    [enc setBuffer:work.storage.buffer offset:work.attention_output atIndex:4];
    [enc setBytes:p length:sizeof(p) atIndex:5];
    launch_grid(enc, model.kernels.batch_attention,
                MTLSizeMake(c.AH, count, 1), MTLSizeMake(BLOCK, 1, 1));
    batch_mv_q4(enc, model, layer.weights, w.out, work,
                work.attention_output, work.hidden, count, true);
}
void batch_ffn_q4(id<MTLComputeCommandEncoder> enc, const Model& model,
                  BatchWork& work, const Layer& layer, int count) {
    batch_mv_q4(enc, model, layer.weights, layer.gate, work,
                work.normalized, work.ffn_gate, count);
    batch_mv_q4(enc, model, layer.weights, layer.up, work,
                work.normalized, work.ffn_up, count);
    const uint32_t p[] = {static_cast<uint32_t>(model.config->I),
                          static_cast<uint32_t>(count)};
    [enc setBuffer:work.storage.buffer offset:work.ffn_gate atIndex:0];
    [enc setBuffer:work.storage.buffer offset:work.ffn_up atIndex:1];
    [enc setBytes:p length:sizeof(p) atIndex:2];
    launch(enc, model.kernels.batch_swiglu,
           (count * model.config->I + BLOCK - 1) / BLOCK);
    batch_mv_q4(enc, model, layer.weights, layer.down, work,
                work.ffn_gate, work.hidden, count, true);
}
void prefill_q4(id<MTLComputeCommandEncoder> enc, const Model& model,
                State& state, const int* tokens, int count,
                bool compute_logits) {
    Q3X_ASSERT(count > 0 && count <= Q4_BATCH_TOKENS,
               "Metal Q4 prefill count=%d", count);
    const auto& c = *model.config;
    BatchWork& work = state.batch;
    batch_embed_q4(enc, model, work, tokens, count);
    for (int i = 0; i < c.N; ++i) {
        const Layer& layer = model.layer[i];
        batch_rms(enc, model, layer.weights, layer.input_norm, work,
                  work.hidden, work.normalized, c.H, count);
        if (i % c.AI != c.AI - 1)
            batch_deltanet_q4(enc, model, state, layer, state.layer[i], count);
        else
            batch_attention_q4(enc, model, state, layer, state.layer[i], count);
        batch_rms(enc, model, layer.weights, layer.post_norm, work,
                  work.hidden, work.normalized, c.H, count);
        batch_ffn_q4(enc, model, work, layer, count);
    }
    if (compute_logits) {
        const uint32_t n = c.H;
        [enc setBuffer:work.storage.buffer
             offset:batch_at(work.hidden, count - 1, c.H) atIndex:0];
        [enc setBuffer:model.weights offset:model.final_norm atIndex:1];
        [enc setBuffer:state.work.storage.buffer offset:state.work.normalized atIndex:2];
        [enc setBytes:&n length:sizeof(n) atIndex:3];
        launch(enc, model.kernels.rms, 1);
        mv(enc, model, model.weights, model.lm_head, state.work,
           state.work.normalized, state.work.logits);
    }
}
void forward(id<MTLComputeCommandEncoder> enc, const Model& model, State& state,
              int token, bool compute_logits) {
    const auto& c = *model.config;
    Work& work = state.work;
    Q3X_ASSERT(token >= 0 && token < c.V && state.position < state.capacity,
               "Metal forward token=%d position=%d", token, state.position);
    embed(enc, model, state, token);
    for (int i = 0; i < c.N; ++i) {
        const Layer& layer = model.layer[i];
        rms(enc, model, layer.weights, work, layer.input_norm);
        if (i % c.AI != c.AI - 1) deltanet(enc, model, state, layer, state.layer[i]);
        else attention(enc, model, state, layer, state.layer[i]);
        rms(enc, model, layer.weights, work, layer.post_norm);
        ffn(enc, model, work, layer);
    }
    if (compute_logits) {
        rms(enc, model, model.weights, work, model.final_norm);
        mv(enc, model, model.weights, model.lm_head, work, work.normalized, work.logits);
    }
    ++state.position;
}

bool Model::load(const char* path, const char** error) {
    auto fail = [&](const char* message) { *error = message; return false; };
    device = MTLCreateSystemDefaultDevice();
    if (!device) return fail("Metal device unavailable");
    if (!kernels.load(device, error)) return false;
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return fail("cannot open model.bin");
    struct stat info {};
    if (fstat(fd, &info) || info.st_size < static_cast<off_t>(q3x_model::HEADER_SIZE)) {
        close(fd); return fail("bad model.bin");
    }
    const size_t size = static_cast<size_t>(info.st_size);
    const uint8_t* file = static_cast<const uint8_t*>(mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));
    close(fd);
    if (file == MAP_FAILED) return fail("mmap model.bin failed");
    auto mapped_fail = [&](const char* message) { munmap(const_cast<uint8_t*>(file), size); return fail(message); };
    if (std::memcmp(file, "Q3XMODL\0", 8)) return mapped_fail("wrong model.bin magic");
    config = q3x_model::config_for_id(q3x_model::header_field(file, q3x_model::MODEL_ID));
    if (!config) return mapped_fail("unsupported Qwen model ID");
    if (!q3x_model::header_matches(file, size, *config))
        return mapped_fail("Qwen model.bin header mismatch");
    size_t cursor = q3x_model::HEADER_SIZE, base = 0;
    auto take = [&](size_t bytes) {
        if (*error) return size_t(0);
        const size_t padding = (64 - cursor % 64) % 64;
        if (cursor > size || padding > size - cursor || bytes > size - cursor - padding) {
            *error = "truncated model.bin"; return size_t(0);
        }
        cursor += padding; const size_t offset = cursor - base; cursor += bytes; return offset;
    };
    // Metal limits a single buffer independently of total memory. Prefix and
    // each layer are fixed contiguous regions; tensor order/alignment are unchanged.
    auto upload = [&](size_t begin, size_t end) -> id<MTLBuffer> {
        if (*error) return nil;
        if (end - begin > device.maxBufferLength) {
            *error = "model weight region exceeds Metal maxBufferLength"; return nil;
        }
        id<MTLBuffer> buffer = [device newBufferWithBytes:file + begin length:end - begin
                                                 options:MTLResourceStorageModeShared];
        if (!buffer) *error = "cannot allocate Metal model weights";
        return buffer;
    };
    const auto& c = *config;
    auto linear = [&](int rows, int cols) {
        Q3X_ASSERT(cols % 32 == 0, "Metal matrix cols=%d", cols);
        const size_t count = static_cast<size_t>(rows) * cols;
        size_t bytes = count * 2;
        if (c.matrix_type == q3x_model::MATRIX_Q8_0)
            bytes = count / q3x_q8::BLOCK_SIZE * sizeof(q3x_q8::Block);
        else if (c.matrix_type == q3x_model::MATRIX_Q4_0)
            bytes = count / q3x_q4::BLOCK_SIZE * sizeof(q3x_q4::Block);
        return Linear {take(bytes), rows, cols, c.matrix_type};
    };
    layer.reset(new (std::nothrow) Layer[c.N]);
    std::unique_ptr<size_t[]> regions(new (std::nothrow) size_t[c.N + 2]);
    if (!layer || !regions) return mapped_fail("cannot allocate model layer table");
    const int AS = c.AH * c.AD, KVW = c.KVH * c.AD;
    const int DO = c.VH * c.VD, DQKV = 2 * c.KH * c.KD + DO;
    embedding = linear(c.V, c.H); lm_head = c.tied_embeddings ? embedding : linear(c.V, c.H);
    final_norm = take(c.H * 2);
    regions[0] = 0; regions[1] = cursor;
    for (int i = 0; i < c.N; ++i) {
        base = cursor;
        Layer& l = layer[i]; l.input_norm = take(c.H * 2);
        if (i % c.AI != c.AI - 1) {
            l.delta.qkv = linear(DQKV, c.H); l.delta.z = linear(DO, c.H);
            l.delta.a = linear(c.VH, c.H); l.delta.b = linear(c.VH, c.H);
            l.delta.conv = take(static_cast<size_t>(DQKV) * c.CK * 2);
            l.delta.alog = take(c.VH * 4); l.delta.dt = take(c.VH * 2); l.delta.norm = take(c.VD * 4);
            l.delta.out = linear(c.H, DO);
        } else {
            l.attention.q = linear(2 * AS, c.H); l.attention.k = linear(KVW, c.H); l.attention.v = linear(KVW, c.H);
            l.attention.qnorm = take(c.AD * 2); l.attention.knorm = take(c.AD * 2);
            l.attention.out = linear(c.H, AS);
        }
        l.post_norm = take(c.H * 2); l.gate = linear(c.I, c.H);
        l.up = linear(c.I, c.H); l.down = linear(c.H, c.I);
        regions[i + 2] = cursor;
    }
    if (*error) return mapped_fail(*error);
    if (cursor != size) return mapped_fail("model.bin size does not match schema");
    weights = upload(regions[0], regions[1]);
    for (int i = 0; i < c.N; ++i) layer[i].weights = upload(regions[i + 1], regions[i + 2]);
    if (*error) return mapped_fail(*error);
    munmap(const_cast<uint8_t*>(file), size);
    LOG_INFO("Metal ready device=%s weights=%zu allocated=%llu max_buffer=%llu "
             "recommended_working_set=%llu",
             device.name.UTF8String, size,
             static_cast<unsigned long long>(device.currentAllocatedSize),
             static_cast<unsigned long long>(device.maxBufferLength),
             static_cast<unsigned long long>(device.recommendedMaxWorkingSetSize));
    return true;
}

Model* model_create(const char* path, char* err, size_t errlen) {
    @autoreleasepool {
        std::unique_ptr<Model> model(new (std::nothrow) Model());
        const char* message = nullptr;
        if (!model || !model->load(path, &message)) {
            if (err && errlen) std::snprintf(err, errlen, "%s", message ? message : "allocation failed");
            return nullptr;
        }
        if (err && errlen) err[0] = '\0';
        return model.release();
    }
}
void model_destroy(Model* model) { delete model; }
State* state_create(Model* model, int context) {
    Q3X_ASSERT(model && context > 0 && context <= q3x_model::MAX_CONTEXT, "Metal context=%d", context);
    @autoreleasepool { return new State(*model, context); }
}
void state_destroy(State* state) { delete state; }
void state_reset(State* state) {
    Q3X_ASSERT(state, "Metal reset null"); state->position = 0;
    std::memset(state->recurrent.data(), 0, state->recurrent.count * sizeof(float));
}
void state_forward(Model* model, State* state, const int* tokens, int count, bool logits) {
    Q3X_ASSERT(model && state && tokens && count > 0, "Metal forward count=%d", count);
    Q3X_ASSERT(state->position >= 0 && state->position + count <= state->capacity,
               "Metal forward position=%d count=%d capacity=%d",
               state->position, count, state->capacity);
    for (int i = 0; i < count; ++i)
        Q3X_ASSERT(tokens[i] >= 0 && tokens[i] < model->config->V,
                   "Metal forward token[%d]=%d vocabulary=%d",
                   i, tokens[i], model->config->V);
    if (model->config->matrix_type == q3x_model::MATRIX_Q4_0 && count > 1) {
        for (int start = 0; start < count; start += Q4_BATCH_TOKENS) {
            @autoreleasepool {
                const int chunk = std::min(Q4_BATCH_TOKENS, count - start);
                id<MTLCommandBuffer> command = [state->queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
                Q3X_ASSERT(command && enc, "Metal command allocation failed");
                prefill_q4(enc, *model, *state, tokens + start, chunk,
                           logits && start + chunk == count);
                [enc endEncoding]; [command commit]; [command waitUntilCompleted];
                Q3X_ASSERT(command.status == MTLCommandBufferStatusCompleted,
                           "Metal execution failed: %s",
                           command.error
                               ? command.error.localizedDescription.UTF8String
                               : "no error detail");
                state->position += chunk;
            }
        }
        return;
    }
    // Bound command-buffer memory for long prompts. Runtime already splits the
    // range at checkpoint_at; no encoded chunk crosses that boundary.
    for (int start = 0; start < count; start += ENCODE_TOKENS) {
        @autoreleasepool {
            id<MTLCommandBuffer> command = [state->queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
            Q3X_ASSERT(command && enc, "Metal command allocation failed");
            const int end = std::min(start + ENCODE_TOKENS, count);
            for (int i = start; i < end; ++i) forward(enc, *model, *state, tokens[i], logits && i + 1 == count);
            [enc endEncoding]; [command commit]; [command waitUntilCompleted];
            Q3X_ASSERT(command.status == MTLCommandBufferStatusCompleted, "Metal execution failed: %s",
                       command.error ? command.error.localizedDescription.UTF8String : "no error detail");
        }
    }
}
void state_checkpoint_save(State* state) {
    Q3X_ASSERT(state, "Metal checkpoint save null");
    state->checkpoint_position = state->position;
    std::memcpy(state->checkpoint.data(), state->recurrent.data(), state->recurrent.count * sizeof(float));
    std::memcpy(state->checkpoint.data(state->recurrent.count * sizeof(float)),
                state->work.storage.data(state->work.logits), state->config->V * sizeof(float));
}
void state_checkpoint_restore(State* state) {
    Q3X_ASSERT(state && state->checkpoint_position <= state->capacity, "Metal checkpoint restore invalid");
    std::memcpy(state->recurrent.data(), state->checkpoint.data(), state->recurrent.count * sizeof(float));
    std::memcpy(state->work.storage.data(state->work.logits),
                state->checkpoint.data(state->recurrent.count * sizeof(float)), state->config->V * sizeof(float));
    state->position = state->checkpoint_position;
}
int state_argmax(const State* state) {
    if (!state) return -1;
    const float* logits = state->work.storage.data(state->work.logits);
    return static_cast<int>(std::max_element(logits, logits + state->config->V) - logits);
}
void state_copy_logits(const State* state, float* output) {
    Q3X_ASSERT(state && output, "Metal copy logits null");
    std::memcpy(output, state->work.storage.data(state->work.logits), state->config->V * sizeof(float));
}
int vocab_size() { return q3x_model::QWEN35_08B.V; }
int max_context() { return q3x_model::MAX_CONTEXT; }
bool token_is_stop(int token) { return token == 248044 || token == 248046; }
uint32_t model_id(const Model* model) { return model->config->id; }

}  // namespace q3x_backend

// Qwen3.5 Metal correctness kernels. Activations/state stay FP32; weights stay
// BF16 or Q8_0. Every dispatch uses 256 threads per threadgroup.
#include <metal_stdlib>
using namespace metal;

constant uint BLOCK = 256, AD = 256, RD = 64, KH = 16, KD = 128, VD = 128, CK = 4;
constant float EPS = 1e-6f, THETA = 10000000.0f;
struct Q8Block { ushort scale; char values[32]; };
static_assert(sizeof(Q8Block) == 34, "Q8_0 block layout");

float bf16(ushort x) { return as_type<float>(uint(x) << 16); }
float sigmoid(float x) {
    return x >= 0.0f ? 1.0f / (1.0f + exp(-x)) : exp(x) / (1.0f + exp(x));
}
float silu(float x) { return x * sigmoid(x); }
float softplus(float x) {
    if (x > 20.0f) return x;
    const float e = exp(x), sum = 1.0f + e;
    // MSL has no log1p. Compensate rounding of 1+e instead of dropping the
    // negative tail to zero; A_log can amplify even a very small softplus.
    return sum == 1.0f ? e : log(sum) * (e / (sum - 1.0f));
}
float sum_group(float x, threadgroup float* shared, uint tid) {
    shared[tid] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = BLOCK / 2; step; step /= 2) {
        if (tid < step) shared[tid] += shared[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float result = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

kernel void embed_bf16(device const ushort* w [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       constant uint2& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.x) out[i] = bf16(w[ulong(p.y) * p.x + i]);
}
kernel void embed_q8(device const Q8Block* w [[buffer(0)]],
                     device float* out [[buffer(1)]],
                     constant uint2& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < p.x) {
        device const Q8Block& b = w[ulong(p.y) * (p.x / 32) + i / 32];
        out[i] = float(as_type<half>(b.scale)) * float(b.values[i % 32]);
    }
}
// p = {columns, add_to_output}. One group reduces one matrix row.
kernel void mv_bf16(device const ushort* w [[buffer(0)]],
                    device const float* x [[buffer(1)]], device float* out [[buffer(2)]],
                    constant uint2& p [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    float sum = 0.0f;
    for (uint i = tid; i < p.x; i += BLOCK) sum += bf16(w[ulong(row) * p.x + i]) * x[i];
    threadgroup float shared[BLOCK];
    const float total = sum_group(sum, shared, tid);
    if (tid == 0) out[row] = total + (p.y ? out[row] : 0.0f);
}
kernel void mv_q8(device const Q8Block* w [[buffer(0)]],
                  device const float* x [[buffer(1)]], device float* out [[buffer(2)]],
                  constant uint2& p [[buffer(3)]], uint row [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]]) {
    float sum = 0.0f;
    for (uint i = tid; i < p.x; i += BLOCK) {
        device const Q8Block& b = w[ulong(row) * (p.x / 32) + i / 32];
        sum += float(as_type<half>(b.scale)) * float(b.values[i % 32]) * x[i];
    }
    threadgroup float shared[BLOCK];
    const float total = sum_group(sum, shared, tid);
    if (tid == 0) out[row] = total + (p.y ? out[row] : 0.0f);
}
kernel void rms(device const float* x [[buffer(0)]], device const ushort* w [[buffer(1)]],
                 device float* out [[buffer(2)]], constant uint& n [[buffer(3)]],
                 uint tid [[thread_index_in_threadgroup]]) {
    float square = 0.0f;
    for (uint i = tid; i < n; i += BLOCK) square += x[i] * x[i];
    threadgroup float shared[BLOCK];
    const float scale = rsqrt(sum_group(square, shared, tid) / n + EPS);
    for (uint i = tid; i < n; i += BLOCK) out[i] = x[i] * scale * (1.0f + bf16(w[i]));
}
kernel void swiglu(device float* gate [[buffer(0)]], device const float* up [[buffer(1)]],
                    constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) gate[i] = silu(gate[i]) * up[i];
}
kernel void conv(device float* qkv [[buffer(0)]], device const ushort* w [[buffer(1)]],
                  device float* history [[buffer(2)]], constant uint& n [[buffer(3)]],
                  uint channel [[thread_position_in_grid]]) {
    if (channel >= n) return;
    device float* past = history + channel * (CK - 1);
    float sum = qkv[channel] * bf16(w[channel * CK + CK - 1]);
    for (uint i = 0; i < CK - 1; ++i) sum += past[i] * bf16(w[channel * CK + i]);
    for (uint i = 0; i < CK - 2; ++i) past[i] = past[i + 1];
    past[CK - 2] = qkv[channel];
    qkv[channel] = silu(sum);
}
kernel void prepare_delta_qk(device const float* qkv [[buffer(0)]],
                             device float* query [[buffer(1)]], device float* key [[buffer(2)]],
                             constant uint& vh [[buffer(3)]],
                             uint head [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
    const uint source = head / (vh / KH) * KD;
    const float q = tid < KD ? qkv[source + tid] : 0.0f;
    const float k = tid < KD ? qkv[KH * KD + source + tid] : 0.0f;
    threadgroup float shared[BLOCK];
    const float qs = rsqrt(sum_group(q * q, shared, tid) + EPS) / sqrt(float(KD));
    const float ks = rsqrt(sum_group(k * k, shared, tid) + EPS);
    if (tid < KD) { query[head * KD + tid] = q * qs; key[head * KD + tid] = k * ks; }
}
kernel void delta_rule(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
                       device const float* v [[buffer(2)]], device const float* a [[buffer(3)]],
                       device const float* b [[buffer(4)]], device const float* alog [[buffer(5)]],
                       device const ushort* dt [[buffer(6)]], device float* memory [[buffer(7)]],
                       device float* out [[buffer(8)]], uint head [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    // Each value lane owns its entire state column: no cross-lane state writes.
    if (tid >= VD) return;
    device float* state = memory + ulong(head) * KD * VD;
    const float biased = a[head] + bf16(dt[head]);
    const float decay = exp(-exp(alog[head]) * softplus(biased)), beta = sigmoid(b[head]);
    float predicted = 0.0f;
    for (uint i = 0; i < KD; ++i) {
        state[i * VD + tid] *= decay;
        predicted += k[head * KD + i] * state[i * VD + tid];
    }
    const float delta = beta * (v[head * VD + tid] - predicted);
    float result = 0.0f;
    for (uint i = 0; i < KD; ++i) {
        state[i * VD + tid] += k[head * KD + i] * delta;
        result += q[head * KD + i] * state[i * VD + tid];
    }
    out[head * VD + tid] = result;
}
kernel void gated_rms(device float* values [[buffer(0)]], device const float* w [[buffer(1)]],
                       device const float* gate [[buffer(2)]], uint head [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    const uint i = head * VD + tid;
    const float x = tid < VD ? values[i] : 0.0f;
    threadgroup float shared[BLOCK];
    const float scale = rsqrt(sum_group(x * x, shared, tid) / VD + EPS);
    if (tid < VD) values[i] = x * scale * w[tid] * silu(gate[i]);
}
kernel void prepare_query(device const float* packed [[buffer(0)]],
                           device const ushort* norm [[buffer(1)]],
                           device float* query [[buffer(2)]], device float* gate [[buffer(3)]],
                           constant uint& position [[buffer(4)]],
                           uint head [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shared[BLOCK], q[AD];
    const float x = packed[head * 2 * AD + tid];
    const float scale = rsqrt(sum_group(x * x, shared, tid) / AD + EPS);
    q[tid] = x * scale * (1.0f + bf16(norm[tid]));
    gate[head * AD + tid] = packed[head * 2 * AD + AD + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float value = q[tid];
    if (tid < RD) {
        const uint i = tid % (RD / 2);
        const float angle = position / pow(THETA, 2.0f * i / RD);
        value = tid < RD / 2 ? q[i] * cos(angle) - q[i + RD / 2] * sin(angle)
                            : q[i + RD / 2] * cos(angle) + q[i] * sin(angle);
    }
    query[head * AD + tid] = value;
}
kernel void prepare_key(device float* key [[buffer(0)]], device const ushort* norm [[buffer(1)]],
                         constant uint& position [[buffer(2)]],
                         uint head [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float shared[BLOCK], k[AD];
    const float x = key[head * AD + tid];
    const float scale = rsqrt(sum_group(x * x, shared, tid) / AD + EPS);
    k[tid] = x * scale * (1.0f + bf16(norm[tid]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float value = k[tid];
    if (tid < RD) {
        const uint i = tid % (RD / 2);
        const float angle = position / pow(THETA, 2.0f * i / RD);
        value = tid < RD / 2 ? k[i] * cos(angle) - k[i + RD / 2] * sin(angle)
                            : k[i + RD / 2] * cos(angle) + k[i] * sin(angle);
    }
    key[head * AD + tid] = value;
}
kernel void store_kv(device const float* key [[buffer(0)]], device const float* value [[buffer(1)]],
                      device float* kc [[buffer(2)]], device float* vc [[buffer(3)]],
                      constant uint2& p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i < p.x) { kc[ulong(p.y) * p.x + i] = key[i]; vc[ulong(p.y) * p.x + i] = value[i]; }
}
// Online stable softmax, one group per query head. KV is [token,KVH,AD].
kernel void attention(device const float* query [[buffer(0)]], device const float* gate [[buffer(1)]],
                       device const float* kc [[buffer(2)]], device const float* vc [[buffer(3)]],
                       device float* out [[buffer(4)]], constant uint4& p [[buffer(5)]],
                       uint head [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint kv_head = head / (p.x / p.y); // p = {AH, KVH, position, 0}
    const float q = query[head * AD + tid];
    float accumulator = 0.0f, maximum = -INFINITY, denominator = 0.0f;
    threadgroup float shared[BLOCK];
    for (uint token = 0; token <= p.z; ++token) {
        const ulong offset = (ulong(token) * p.y + kv_head) * AD;
        const float score = sum_group(q * kc[offset + tid], shared, tid) / sqrt(float(AD));
        const float next_maximum = max(maximum, score);
        const float alpha = exp(maximum - next_maximum), beta = exp(score - next_maximum);
        accumulator = accumulator * alpha + beta * vc[offset + tid];
        denominator = denominator * alpha + beta;
        maximum = next_maximum;
    }
    out[head * AD + tid] = accumulator / denominator * sigmoid(gate[head * AD + tid]);
}

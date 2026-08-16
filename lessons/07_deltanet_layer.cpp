// 第 07 课：把 DeltaNet 的零件组装成一个真正的 toy layer。
//
// 执行顺序与 Qwen 相同：
//   in_proj_qkv / z / a / b -> depthwise causal conv -> q/k L2 norm
//   -> gated delta recurrence -> gated RMSNorm -> out_proj。
// 所有维度都缩到 2；因此本课能运行，但每一行仍对应真实模型中的同一位置。
//
// 这不是一个可泛化的 DeltaNet 实现：权重固定在函数中，conv kernel 也特意取成
// [0, 1]。它的任务是把上一课的 recurrence 放回真实 layer 的前后投影、门控和
// normalization 之间，回答“一个 token 进入 DeltaNet layer 时具体经过什么”。

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace lesson07 {

constexpr int H = 2, QKV = 6, D = 2;
// H 是输入/输出 hidden size；QKV=Q(2)+K(2)+V(2)，D 是本例单 head 的宽度。

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float silu(float x) { return x * sigmoid(x); }
float softplus(float x) { return std::log1p(std::exp(x)); }

void mv(const float (*weight)[H], int rows, const float* x, float* y) {
    // 所有投影都写成同一件事 y = W @ x；真实权重只是 rows/cols 更大。
    for (int row = 0; row < rows; ++row) y[row] = weight[row][0] * x[0] + weight[row][1] * x[1];
}

void l2(float* x) {
    // DeltaNet 的 Q/K 使用 L2 norm；这和 layer 边界的 RMSNorm 是不同的归一化。
    const float scale = 1.0f / std::sqrt(x[0] * x[0] + x[1] * x[1] + 1e-6f);
    x[0] *= scale;
    x[1] *= scale;
}

struct State {
    // 每个 qkv channel 只保留一个卷积历史；S 是本 layer 唯一的 recurrent state。
    // 即使生成无限长文本，这两个数组也不会随 token 数增长。
    float conv_history[QKV] = {};
    float recurrent[D][D] = {};
};

// kernel=[0,1] 的最小 causal depthwise convolution：输出只看当前输入，
// 但仍明确写出 history 更新，以显示真实 CK=4 时 state 从哪里来。
void conv(float* x, State* state) {
    // 真实 kernel 是每通道 4 个参数，对当前和前三个 qkv input 做卷积后再 SiLU。
    for (int channel = 0; channel < QKV; ++channel) {
        const float current = x[channel];
        x[channel] = silu(current);
        state->conv_history[channel] = current;
    }
}

void delta_rule(const float* q, const float* k, const float* v, float log_decay,
                float beta, State* state, float* out) {
    // 与 lesson 06 完全相同的 S 更新，只是这里嵌在 layer 的其它操作中。
    for (int key = 0; key < D; ++key) {
        for (int value = 0; value < D; ++value) state->recurrent[key][value] *= std::exp(log_decay);
    }
    float memory[D] = {};
    for (int value = 0; value < D; ++value) {
        for (int key = 0; key < D; ++key) memory[value] += k[key] * state->recurrent[key][value];
    }
    for (int key = 0; key < D; ++key) {
        for (int value = 0; value < D; ++value) {
            state->recurrent[key][value] += k[key] * beta * (v[value] - memory[value]);
        }
    }
    for (int value = 0; value < D; ++value) {
        out[value] = 0.0f;
        for (int key = 0; key < D; ++key) out[value] += q[key] * state->recurrent[key][value];
    }
}

void delta_layer(const float* input, State* state, float* output) {
    // 前三行是 Q，接着 K，最后两行 V。真实 Qwen 将同样的拼接先过 conv。
    const float qkv_weight[QKV][H] = {
        {1, 0}, {0, 1}, {0.5f, 0}, {0, 0.5f}, {1, 0}, {0, 1},
    };
    const float z_weight[D][H] = {{1, 0}, {0, 1}};
    const float scalar_weight[1][H] = {{0, 0}};
    // z 是末尾 gated RMSNorm 的门；a 产生 decay，b 经 sigmoid 产生 beta。
    float qkv[QKV] = {}, z[D] = {}, a[1] = {}, b[1] = {};
    mv(qkv_weight, QKV, input, qkv);
    mv(z_weight, D, input, z);
    mv(scalar_weight, 1, input, a);
    mv(scalar_weight, 1, input, b);
    conv(qkv, state);

    // conv 已把 qkv 原地替换为激活后的值；按固定布局拆成 Q、K、V 三段。
    float q[D] = {qkv[0], qkv[1]};
    float k[D] = {qkv[2], qkv[3]};
    l2(q);
    l2(k);
    // Q 额外除 sqrt(D)，等价于实现中对 query 的缩放约定。
    for (float& value : q) value /= std::sqrt(static_cast<float>(D));

    float read[D] = {};
    const float log_decay = -softplus(a[0]);  // A_log=0、dt_bias=0 的 toy 情况。
    delta_rule(q, k, qkv + 4, log_decay, sigmoid(b[0]), state, read);

    // 真实 layer 在这里做 per-head gated RMSNorm；toy 的 norm.weight=1。
    // 因而输出 = RMS(read) * SiLU(z)，再由真实 out_proj 投回 hidden。
    const float rms = 1.0f / std::sqrt((read[0] * read[0] + read[1] * read[1]) / D + 1e-6f);
    for (int i = 0; i < D; ++i) output[i] = read[i] * rms * silu(z[i]);
}

void self_test() {
    State first, second;
    const float input[H] = {1.0f, 0.0f};
    float output_a[H] = {}, output_b[H] = {};
    delta_layer(input, &first, output_a);
    delta_layer(input, &second, output_b);
    // 两份零初始 state 对同一 token 必须产生同一输出；随后 state 已非零，说明
    // 这不是无状态前馈层。
    assert(std::isfinite(output_a[0]) && std::isfinite(output_a[1]));
    assert(std::fabs(output_a[0] - output_b[0]) < 1e-6f);
    assert(std::fabs(first.recurrent[0][0]) > 0.0f);
}

}  // namespace lesson07

int main() {
    // 只运行一个 token；多 token 时应复用同一个 State，才能观察 recurrence。
    lesson07::self_test();
    lesson07::State state;
    const float input[lesson07::H] = {1.0f, 0.0f};
    float output[lesson07::H] = {};
    lesson07::delta_layer(input, &state, output);
    std::printf("DeltaNet layer output: [%.6f, %.6f]\n", output[0], output[1]);
}

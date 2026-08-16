// 第 07 课：把 DeltaNet 的零件组装成一个真正的 toy layer。
//
// 执行顺序与 Qwen 相同：
//   in_proj_qkv / z / a / b -> depthwise causal conv -> q/k L2 norm
//   -> gated delta recurrence -> gated RMSNorm -> out_proj。
// 所有维度都缩到 2；因此本课能运行，但每一行仍对应真实模型中的同一位置。

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace lesson07 {

constexpr int H = 2, QKV = 6, D = 2;

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float silu(float x) { return x * sigmoid(x); }
float softplus(float x) { return std::log1p(std::exp(x)); }

void mv(const float (*weight)[H], int rows, const float* x, float* y) {
    for (int row = 0; row < rows; ++row) y[row] = weight[row][0] * x[0] + weight[row][1] * x[1];
}

void l2(float* x) {
    const float scale = 1.0f / std::sqrt(x[0] * x[0] + x[1] * x[1] + 1e-6f);
    x[0] *= scale;
    x[1] *= scale;
}

struct State {
    // 每个 qkv channel 只保留一个卷积历史；S 是本 layer 唯一的 recurrent state。
    float conv_history[QKV] = {};
    float recurrent[D][D] = {};
};

// kernel=[0,1] 的最小 causal depthwise convolution：输出只看当前输入，
// 但仍明确写出 history 更新，以显示真实 CK=4 时 state 从哪里来。
void conv(float* x, State* state) {
    for (int channel = 0; channel < QKV; ++channel) {
        const float current = x[channel];
        x[channel] = silu(current);
        state->conv_history[channel] = current;
    }
}

void delta_rule(const float* q, const float* k, const float* v, float log_decay,
                float beta, State* state, float* out) {
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
    float qkv[QKV] = {}, z[D] = {}, a[1] = {}, b[1] = {};
    mv(qkv_weight, QKV, input, qkv);
    mv(z_weight, D, input, z);
    mv(scalar_weight, 1, input, a);
    mv(scalar_weight, 1, input, b);
    conv(qkv, state);

    float q[D] = {qkv[0], qkv[1]};
    float k[D] = {qkv[2], qkv[3]};
    l2(q);
    l2(k);
    for (float& value : q) value /= std::sqrt(static_cast<float>(D));

    float read[D] = {};
    const float log_decay = -softplus(a[0]);  // A_log=0、dt_bias=0 的 toy 情况。
    delta_rule(q, k, qkv + 4, log_decay, sigmoid(b[0]), state, read);

    // 真实 layer 在这里做 per-head gated RMSNorm；toy 的 norm.weight=1。
    const float rms = 1.0f / std::sqrt((read[0] * read[0] + read[1] * read[1]) / D + 1e-6f);
    for (int i = 0; i < D; ++i) output[i] = read[i] * rms * silu(z[i]);
}

void self_test() {
    State first, second;
    const float input[H] = {1.0f, 0.0f};
    float output_a[H] = {}, output_b[H] = {};
    delta_layer(input, &first, output_a);
    delta_layer(input, &second, output_b);
    assert(std::isfinite(output_a[0]) && std::isfinite(output_a[1]));
    assert(std::fabs(output_a[0] - output_b[0]) < 1e-6f);
    assert(std::fabs(first.recurrent[0][0]) > 0.0f);
}

}  // namespace lesson07

int main() {
    lesson07::self_test();
    lesson07::State state;
    const float input[lesson07::H] = {1.0f, 0.0f};
    float output[lesson07::H] = {};
    lesson07::delta_layer(input, &state, output);
    std::printf("DeltaNet layer output: [%.6f, %.6f]\n", output[0], output[1]);
}

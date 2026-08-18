// 第 02 课：SwiGLU FFN 与 residual connection。
//
// 一个 Qwen FFN 的完整公式是：
//   ffn(x) = down_proj( SiLU(gate_proj(x)) * up_proj(x) )
//   output = x + ffn(x)
// 本课把两个上投影的结果直接写成 gate/up，先隔离理解非线性和 residual。
// 真实模型中 gate_proj 和 up_proj 都把 [hidden] 投到更宽的 [intermediate]，
// down_proj 再投回 [hidden]；这里让三个长度都为 3，只为让公式一眼可见。
//
// 白话记忆：FFN 不读取别的 token；它是每个 token 自己的“小型非线性思考器”。up
// 提供候选信息，gate 决定每个候选通道开多大，down 再把变宽后的结果压回 hidden。
// residual 的 x + ffn(x) 则表示“保留原句的表示，只加上本分支学到的修正”，不是覆盖 x。

#include <cassert>
#include <cmath>
#include <cstdio>

namespace lesson02 {

constexpr int kHidden = 3;

bool close(float left, float right) { return std::fabs(left - right) < 1e-5f; }

float sigmoid(float x) {
    // 数值稳定的 sigmoid；SwiGLU 里的 SiLU 会调用它。
    if (x >= 0.0f) return 1.0f / (1.0f + std::exp(-x));
    const float exp_x = std::exp(x);
    return exp_x / (1.0f + exp_x);
}

float silu(float x) { return x * sigmoid(x); }

// gate 与 up 的逐元素相乘正是 SwiGLU；输出长度保持为 intermediate size。
void swiglu(const float* gate, const float* up, float* output) {
    // SiLU(gate[i]) 不是 sigmoid(gate[i])：它额外乘回 gate[i]，保留正负幅度信息。
    for (int i = 0; i < kHidden; ++i) output[i] = silu(gate[i]) * up[i];
}

void residual_add(float* hidden, const float* branch) {
    // residual 不是拼接，而是同一位置逐元素相加。
    // 这使每个分支只需要学习对已有 hidden 的修正，也让深层网络保留直通路径。
    for (int i = 0; i < kHidden; ++i) hidden[i] += branch[i];
}

void self_test() {
    const float gate[kHidden] = {0.0f, 1.0f, -2.0f};
    const float up[kHidden] = {2.0f, 3.0f, 4.0f};
    const float original_hidden[kHidden] = {1.0f, -2.0f, 0.5f};
    float ffn[kHidden] = {};
    float hidden[kHidden] = {original_hidden[0], original_hidden[1], original_hidden[2]};

    swiglu(gate, up, ffn);
    residual_add(hidden, ffn);

    // gate=0 时 SiLU(0)=0，因此即使 up 非零，这一个 intermediate 通道也关闭。
    assert(close(ffn[0], 0.0f));
    assert(close(ffn[1], silu(1.0f) * 3.0f));
    assert(close(hidden[0], original_hidden[0]));
    assert(close(hidden[2], original_hidden[2] + silu(-2.0f) * 4.0f));
}

}  // namespace lesson02

int main() {
    // 这里打印的是 down_proj 之前的 SwiGLU 分支；真正的 layer 会再线性投回 hidden。
    lesson02::self_test();

    const float gate[lesson02::kHidden] = {0.0f, 1.0f, -2.0f};
    const float up[lesson02::kHidden] = {2.0f, 3.0f, 4.0f};
    float ffn[lesson02::kHidden] = {};
    float hidden[lesson02::kHidden] = {1.0f, -2.0f, 0.5f};
    lesson02::swiglu(gate, up, ffn);
    lesson02::residual_add(hidden, ffn);

    std::printf("SwiGLU branch: [%.6f, %.6f, %.6f]\n", ffn[0], ffn[1], ffn[2]);
    std::printf("after residual: [%.6f, %.6f, %.6f]\n", hidden[0], hidden[1], hidden[2]);
}

// 第 01 课：Qwen 的 ordinary RMSNorm 与一个线性层。
//
// 阅读路线：
//   已经会：一个 token 已经是 hidden 向量；00 课用 dot 组成 lm_head，让它变成 logits。
//   本课只加：RMSNorm 稳定这排数字，再用 linear 让各 hidden dimension 相互混合。
//   运行后看：同一个输入在 RMSNorm 前后长度改变，linear 则从全部输入维度计算每个输出维度。
//   下一课：把两条 linear 支路接成每个 token 自己的非线性 FFN。
//
// 上一课的 embedding 只是“取一行”。真实模型随后反复执行：
//   y = RMSNorm(x) -> W * y
// 这里保留最小向量，读者可以用纸笔复算每个数。
//
// RMSNorm 只按向量的均方根缩放，不像 LayerNorm 那样先减去均值。它不混合
// hidden dimension；真正混合维度的是后面的 linear。Qwen 的 layer 是 pre-norm：
// 先 norm，再把结果送入 attention/DeltaNet 或 FFN 分支。
//
// 白话记忆：hidden 是当前 token 的一排特征数字。RMSNorm 先把这排数字的整体音量
// 调到稳定范围；linear 再让每个输出特征读取所有输入特征，并把它们混合成新特征。
// 本课不会让 token 互相通信；“看前文”是 attention/DeltaNet 在后续课程负责的事。

#include <cassert>
#include <cmath>
#include <cstdio>

namespace lesson01 {

constexpr int kHidden = 2;
constexpr int kOutput = 2;
constexpr float kEpsilon = 1e-6f;

// 目的/直觉：测试浮点计算结果时允许极小的舍入误差，不能直接用 left == right。
// 数学：      close(a, b) = |a - b| < 1e-5。
// 实现：      std::fabs 求绝对值，再与固定误差阈值比较。
bool close(float left, float right) { return std::fabs(left - right) < 1e-5f; }

// 目的/直觉：沿用第 00 课的 dot，衡量两个等长向量对应维度的总体匹配程度。
// 数学：      dot(a, b) = sum_{i=0}^{N-1}(a_i * b_i) -> scalar。
// 实现：      遍历 N 个位置，逐项相乘并累加到 sum。
float dot(const float* left, const float* right, int count) {
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) sum += left[i] * right[i];
    return sum;
}

// 目的/直觉：把 hidden x[H] 的整体“音量”调到稳定范围，再让每个维度乘自己的
//             可学习缩放；它不让不同维度相加，所以不负责混合特征。
// 数学：      m   = (1 / H) * sum_{i=0}^{H-1}(x_i^2)
//             r   = 1 / sqrt(m + eps)
//             y_i = x_i * r * (1 + w_i),  i = 0, 1, ..., H-1
//             输入 x[H]、checkpoint weight w[H]，输出 y[H]。
// 实现：      第一遍循环求平方和并除以 H；计算一次 r；第二遍循环逐维写出 y_i。
//             Qwen3.5/3.8 ordinary RMSNorm 使用 (1 + w_i)，不是直接使用 w_i。
void qwen_rmsnorm(const float* input, const float* weight, float* output) {
    // m = (x_0^2 + x_1^2 + ... + x_(H-1)^2) / H
    float mean_square = 0.0f;
    for (int i = 0; i < kHidden; ++i) mean_square += input[i] * input[i];
    mean_square /= kHidden;

    // r = 1 / sqrt(m + eps)
    const float inverse_rms = 1.0f / std::sqrt(mean_square + kEpsilon);

    // 对每一维 i：y_i = x_i * r * (1 + w_i)
    for (int i = 0; i < kHidden; ++i) output[i] = input[i] * inverse_rms * (1.0f + weight[i]);
}

// 目的/直觉：把 input[H] 投影成 output[O]；每个输出维度读取全部输入维度，
//             因而 linear 才是本课真正混合 hidden dimension 的操作。
// 数学：      y = W @ x
//             W[O,H] @ x[H] -> y[O]
//             y_o = dot(W[o], x) = sum_{h=0}^{H-1}(W[o,h] * x_h)
// 实现：      按输出行 o 遍历 W；每一行 W[o] 与 input 做一次第 00 课学过的 dot。
//             当前投影没有 bias，所以不需要再加 b_o。
void linear(const float weight[kOutput][kHidden], const float* input, float* output) {
    for (int output_row = 0; output_row < kOutput; ++output_row) {
        output[output_row] = dot(weight[output_row], input, kHidden);
    }
}

// 目的/直觉：用可以手算的固定输入，锁定 RMSNorm 和 linear 的预期行为。
// 数学：      x=[3,4] 时 m=(3^2+4^2)/2=12.5；再验证 y=W@RMSNorm(x)。
// 实现：      运行两个函数，并用 close + assert 比较实际值与手算公式。
void self_test() {
    const float input[kHidden] = {3.0f, 4.0f};
    const float norm_weight[kHidden] = {0.0f, 0.0f};
    const float projection[kOutput][kHidden] = {{1.0f, 2.0f}, {3.0f, -1.0f}};
    float normalized[kHidden] = {};
    float output[kOutput] = {};

    qwen_rmsnorm(input, norm_weight, normalized);
    linear(projection, normalized, output);

    // mean(x^2) = (9 + 16) / 2 = 12.5；weight=0 表示 RMSNorm 的缩放恰好是 1。
    const float inverse_rms = 1.0f / std::sqrt(12.5f + kEpsilon);
    assert(close(normalized[0], 3.0f * inverse_rms));
    assert(close(normalized[1], 4.0f * inverse_rms));
    assert(close(output[0], normalized[0] + 2.0f * normalized[1]));
    assert(close(output[1], 3.0f * normalized[0] - normalized[1]));
}

}  // namespace lesson01

// 目的/直觉：把自测中的中间结果打印出来，让读者直接看到 norm 与 linear 是两个操作。
// 数学：      normalized=RMSNorm(input)，output=projection@normalized。
// 实现：      先运行 self_test，再重复同一个例子并打印两个输出向量。
int main() {
    lesson01::self_test();

    const float input[lesson01::kHidden] = {3.0f, 4.0f};
    const float norm_weight[lesson01::kHidden] = {0.0f, 0.0f};
    const float projection[lesson01::kOutput][lesson01::kHidden] = {
        {1.0f, 2.0f}, {3.0f, -1.0f}};
    float normalized[lesson01::kHidden] = {};
    float output[lesson01::kOutput] = {};
    lesson01::qwen_rmsnorm(input, norm_weight, normalized);
    lesson01::linear(projection, normalized, output);

    std::printf("RMSNorm: [%.6f, %.6f]\n", normalized[0], normalized[1]);
    std::printf("linear:  [%.6f, %.6f]\n", output[0], output[1]);
}

// 第 01 课：Qwen 的 ordinary RMSNorm 与一个线性层。
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
constexpr float kEpsilon = 1e-6f;

bool close(float left, float right) { return std::fabs(left - right) < 1e-5f; }

// Qwen3.5/3.8 ordinary RMSNorm 的 scale 是 (1 + weight)，不是 weight 本身。
void qwen_rmsnorm(const float* input, const float* weight, float* output) {
    // RMS(x) = sqrt(mean_i(x_i^2) + eps)。eps 防止全零输入除零。
    float mean_square = 0.0f;
    for (int i = 0; i < kHidden; ++i) mean_square += input[i] * input[i];
    mean_square /= kHidden;

    const float inverse_rms = 1.0f / std::sqrt(mean_square + kEpsilon);
    // checkpoint 保存的是 weight；Qwen 的 ordinary RMSNorm 实际乘 1 + weight。
    for (int i = 0; i < kHidden; ++i) output[i] = input[i] * inverse_rms * (1.0f + weight[i]);
}

// 行主序权重 W[out_dim][in_dim] 与输入向量相乘。
void linear(const float weight[2][kHidden], const float* input, float* output) {
    // y[row] = sum_col W[row,col] * x[col]。无 bias：Qwen 的这些投影没有 bias。
    for (int output_row = 0; output_row < 2; ++output_row) {
        output[output_row] = 0.0f;
        for (int input_column = 0; input_column < kHidden; ++input_column) {
            output[output_row] += weight[output_row][input_column] * input[input_column];
        }
    }
}

void self_test() {
    const float input[kHidden] = {3.0f, 4.0f};
    const float norm_weight[kHidden] = {0.0f, 0.0f};
    const float projection[2][kHidden] = {{1.0f, 2.0f}, {3.0f, -1.0f}};
    float normalized[kHidden] = {};
    float output[2] = {};

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

int main() {
    // 输出先显示 norm 后的向量，再显示 W @ norm；两者不应混淆成同一个操作。
    lesson01::self_test();

    const float input[lesson01::kHidden] = {3.0f, 4.0f};
    const float norm_weight[lesson01::kHidden] = {0.0f, 0.0f};
    const float projection[2][lesson01::kHidden] = {{1.0f, 2.0f}, {3.0f, -1.0f}};
    float normalized[lesson01::kHidden] = {};
    float output[2] = {};
    lesson01::qwen_rmsnorm(input, norm_weight, normalized);
    lesson01::linear(projection, normalized, output);

    std::printf("RMSNorm: [%.6f, %.6f]\n", normalized[0], normalized[1]);
    std::printf("linear:  [%.6f, %.6f]\n", output[0], output[1]);
}

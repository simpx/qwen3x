// 第 04 课：一个 head 的 causal attention。
//
// 当前 token 的 query 只能读取当前位置及其之前的 key/value：
//   score[t] = dot(q, k[t]) / sqrt(head_dim)
//   p = softmax(score)
//   output = sum_t p[t] * v[t]
// “causal” 在这个最小 decode 版本中很直白：传入的 key/value 列表就是过去
// 与当前 token，未来 token 从来不会传进来。
// 在 prompt 的 prefill 中，实际实现可一次计算整个下三角 attention matrix；本课
// 用逐 token 的 decode 写法表达同一条因果规则，也正是 KV cache 所需的形式。

#include <cassert>
#include <cmath>
#include <cstdio>

namespace lesson04 {

constexpr int kHeadDim = 2;

bool close(float left, float right) { return std::fabs(left - right) < 1e-5f; }

float dot(const float* left, const float* right) {
    // 这是单 head 的 q·k；多 head 的版本只是在每个 head 上独立重复它。
    float sum = 0.0f;
    for (int i = 0; i < kHeadDim; ++i) sum += left[i] * right[i];
    return sum;
}

// keys/values 的行数是已有 token 数，每一行是一个 head_dim 向量。
void causal_attention(const float* query, const float (*keys)[kHeadDim],
                      const float (*values)[kHeadDim], int token_count, float* output) {
    assert(token_count > 0);
    float scores[8] = {};  // 教学模型最多演示 8 个历史 token。
    assert(token_count <= 8);

    // 除 sqrt(head_dim) 控制 dot product 的尺度，避免 softmax 随维度变尖。
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    float maximum = -INFINITY;
    for (int token = 0; token < token_count; ++token) {
        scores[token] = dot(query, keys[token]) * scale;
        if (scores[token] > maximum) maximum = scores[token];
    }

    // 减去最大值是稳定 softmax 的标准写法：exp(score-maximum) 不会溢出，
    // 且归一化后概率完全不变。
    float denominator = 0.0f;
    for (int token = 0; token < token_count; ++token) {
        scores[token] = std::exp(scores[token] - maximum);
        denominator += scores[token];
    }

    for (int dimension = 0; dimension < kHeadDim; ++dimension) output[dimension] = 0.0f;
    for (int token = 0; token < token_count; ++token) {
        const float probability = scores[token] / denominator;  // 所有 token 概率和为 1。
        for (int dimension = 0; dimension < kHeadDim; ++dimension) {
            output[dimension] += probability * values[token][dimension];
        }
    }
}

void self_test() {
    const float query[kHeadDim] = {1.0f, 0.0f};
    const float keys[2][kHeadDim] = {{1.0f, 0.0f}, {0.0f, 1.0f}};
    const float values[2][kHeadDim] = {{10.0f, 0.0f}, {0.0f, 20.0f}};
    float output[kHeadDim] = {};

    causal_attention(query, keys, values, 2, output);

    // q 更像第 0 个 key。两个 score 分别是 1/sqrt(2) 和 0，
    // 所以可直接手算 softmax 后的两个输出分量。
    const float p0 = std::exp(1.0f / std::sqrt(2.0f)) / (std::exp(1.0f / std::sqrt(2.0f)) + 1.0f);
    assert(close(output[0], 10.0f * p0));
    assert(close(output[1], 20.0f * (1.0f - p0)));
}

}  // namespace lesson04

int main() {
    // 输出是对 value 的加权平均，而不是对 key 或 score 的输出。
    lesson04::self_test();

    const float query[lesson04::kHeadDim] = {1.0f, 0.0f};
    const float keys[2][lesson04::kHeadDim] = {{1.0f, 0.0f}, {0.0f, 1.0f}};
    const float values[2][lesson04::kHeadDim] = {{10.0f, 0.0f}, {0.0f, 20.0f}};
    float output[lesson04::kHeadDim] = {};
    lesson04::causal_attention(query, keys, values, 2, output);
    std::printf("attention output: [%.6f, %.6f]\n", output[0], output[1]);
}

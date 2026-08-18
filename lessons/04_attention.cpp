// 第 04 课：一个 head 的 causal attention。
//
// 阅读路线：
//   已经会：Q/K 可以带上位置；每个 token 也能经 FFN 独立处理自己。
//   本课只加：当前 Q 对历史 K 打分，用 softmax 得到权重，再从历史 V 取回信息。
//   运行后看：输出是多个 V 的加权平均，不是选择一个 V，也不会读取未来 token。
//   下一课：把每个旧 token 的 K/V 留下来，避免每次生成都从头计算。
//
// 当前 token 的 query 只能读取当前位置及其之前的 key/value：
//   score[t] = dot(q, k[t]) / sqrt(head_dim)
//   p = softmax(score)
//   output = sum_t p[t] * v[t]
// “causal” 在这个最小 decode 版本中很直白：传入的 key/value 列表就是过去
// 与当前 token，未来 token 从来不会传进来。
// 在 prompt 的 prefill 中，实际实现可一次计算整个下三角 attention matrix；本课
// 用逐 token 的 decode 写法表达同一条因果规则，也正是 KV cache 所需的形式。
//
// 白话记忆：attention 分两步。query（Q）是当前 token 想找什么；每个 key（K）像一张
// 可匹配的地址卡；softmax 根据 Q 和各 K 的匹配程度给历史位置分配权重，最后取这些位置
// value（V，真正要带回来的内容）的加权平均。Q 用来“找”，V 用来“读”。

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

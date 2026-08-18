// 第 03 课：RoPE（Rotary Position Embedding）。
//
// 阅读路线：
//   已经会：每个 token 的 hidden 会经过 linear 产生不同用途的向量。
//   本课只加：在 attention 读取前文之前，按 token position 旋转它的 Q/K 向量。
//   运行后看：position 0 不旋转；position 约等于 pi/2 时二维向量转了四分之一圈。
//   下一课：把带位置的 Q/K 真正用于 causal attention 的“找”和“读”。
//
// attention 本身只比较向量内容，不知道 token 在第几个位置。Qwen 在每层
// attention 的 Q 与 K 上做 RoPE，把 position 变成二维平面的旋转角度。
// 本课只处理一个 head 的 4 个通道，且使用 Qwen 的 half-rotation 布局：
// [x0, x1, x2, x3] 中 (x0,x2) 与 (x1,x3) 分别组成两个旋转平面。
// RoPE 不额外把 position 加到 hidden 上；它改写 Q/K，使 dot(q_p, k_t) 天然依赖
// 相对位置 p-t。只有 attention 层的 Q/K 使用 RoPE；DeltaNet 层不走这条路径。
//
// 白话记忆：如果没有位置，两个内容相同的 token 在第 1 位或第 100 位看起来完全一样。
// RoPE 不是把“位置编号”塞进向量末尾，而是按位置把 Q/K 的每一对数字转一个不同角度；
// 之后做 q·k 时，结果自然会随两个 token 的相对距离改变。

#include <cassert>
#include <cmath>
#include <cstdio>

namespace lesson03 {

constexpr int kHeadDim = 4;
constexpr int kRotaryDim = 4;
constexpr float kRopeTheta = 10000000.0f;

bool close(float left, float right) { return std::fabs(left - right) < 1e-5f; }

// 为一个 token position 计算每个旋转平面的 cos/sin。
void make_rope_frequencies(int position, float* cosine, float* sine) {
    constexpr int half = kRotaryDim / 2;
    for (int i = 0; i < half; ++i) {
        // 第 i 个二维平面的角频率是 theta^(-2i/rotary_dim)。低 i 转得快，高 i 转得慢。
        const float inverse_frequency = 1.0f / std::pow(kRopeTheta, 2.0f * i / kRotaryDim);
        const float angle = position * inverse_frequency;
        // Qwen 的 half-rotation 需要把同一频率复制到前、后两个半区。
        cosine[i] = cosine[i + half] = std::cos(angle);
        sine[i] = sine[i + half] = std::sin(angle);
    }
}

void apply_rope(float* vector, const float* cosine, const float* sine) {
    constexpr int half = kRotaryDim / 2;
    // 原地旋转会覆盖同一对的另一个输入，因此先备份 rotary 部分。
    float original[kRotaryDim] = {};
    for (int i = 0; i < kRotaryDim; ++i) original[i] = vector[i];

    for (int i = 0; i < half; ++i) {
        vector[i] = original[i] * cosine[i] - original[i + half] * sine[i];
        vector[i + half] = original[i + half] * cosine[i + half] + original[i] * sine[i + half];
    }
}

void self_test() {
    float position_zero_cos[kRotaryDim] = {};
    float position_zero_sin[kRotaryDim] = {};
    float identity[kHeadDim] = {1.0f, 2.0f, 3.0f, 4.0f};
    make_rope_frequencies(0, position_zero_cos, position_zero_sin);
    apply_rope(identity, position_zero_cos, position_zero_sin);
    for (int i = 0; i < kHeadDim; ++i) assert(close(identity[i], static_cast<float>(i + 1)));

    // 手工给出 cos=0、sin=1，相当于每个二维平面旋转 90 度。
    // 这比比较一串 sin/cos 近似数更直接地验证旋转的符号和 half-rotation 配对。
    float quarter_turn[kHeadDim] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float cosine[kRotaryDim] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float sine[kRotaryDim] = {1.0f, 1.0f, 1.0f, 1.0f};
    apply_rope(quarter_turn, cosine, sine);
    assert(close(quarter_turn[0], -3.0f));
    assert(close(quarter_turn[1], -4.0f));
    assert(close(quarter_turn[2], 1.0f));
    assert(close(quarter_turn[3], 2.0f));
}

}  // namespace lesson03

int main() {
    // position=1 的结果会是浮点近似；position=0 在 self_test 中已经验证为恒等变换。
    lesson03::self_test();

    float vector[lesson03::kHeadDim] = {1.0f, 2.0f, 3.0f, 4.0f};
    float cosine[lesson03::kRotaryDim] = {};
    float sine[lesson03::kRotaryDim] = {};
    lesson03::make_rope_frequencies(1, cosine, sine);
    lesson03::apply_rope(vector, cosine, sine);

    std::printf("position 1 RoPE: [%.6f, %.6f, %.6f, %.6f]\n",
                vector[0], vector[1], vector[2], vector[3]);
}

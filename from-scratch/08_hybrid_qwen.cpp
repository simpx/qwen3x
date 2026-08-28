// 第 08 课：Qwen hybrid stack 的控制流。
//
// 阅读路线：
//   已经会：RMSNorm/SwiGLU/residual、RoPE/attention cache、DeltaNet recurrent state。
//   本课只加：每四层选择 3 个 DeltaNet + 1 个 attention，并把两类历史放进同一个 State。
//   运行后看：同样的 token embedding 连续 forward 两次，第二次因旧 state 得到不同输出。
//   下一课：保持同一控制流，把 toy 维度和手写权重替换为真实 Qwen3.5-0.8B。
//
// 一个 token 穿过每层时都执行同一副骨架：
//   n      = RMSNorm(hidden)
//   branch = DeltaNet(n, delta_state) 或 Attention(n, kv_cache)
//   hidden = hidden + branch
//   n      = RMSNorm(hidden)
//   hidden = hidden + SwiGLU_FFN(n)
//
// 本课所有 hidden/head/intermediate width 都固定为 2；projection 也使用容易手算的
// identity weight。这样 mixer 和 FFN 仍执行前面课程的真实公式，而本课只突出 layer
// 调度与 state 生命周期，不再使用标量均值或 0.1*x 占位符。

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace lesson08 {

constexpr int kHidden = 2;
constexpr int kLayers = 4;
constexpr int kDeltaLayers = 3;
constexpr float kEpsilon = 1e-6f;

using Vector = std::array<float, kHidden>;
using Matrix = std::array<Vector, kHidden>;

constexpr Matrix kIdentity = {{{1.0f, 0.0f}, {0.0f, 1.0f}}};

// 目的/直觉：比较多步 stateful forward 的浮点结果时容许微小误差。
// 数学：      close(a,b)=|a-b|<1e-5。
// 实现：      对差值取绝对值，再与固定阈值比较。
bool close(float left, float right) { return std::fabs(left - right) < 1e-5f; }

// 目的/直觉：复用第 00 课，为 linear、attention score 和 state read 提供点积。
// 数学：      dot(a,b)=sum_{i=0}^{H-1}(a_i*b_i)。
// 实现：      遍历 H 个维度，逐项相乘并累加。
float dot(const Vector& left, const Vector& right) {
    float sum = 0.0f;
    for (int i = 0; i < kHidden; ++i) sum += left[i] * right[i];
    return sum;
}

// 目的/直觉：复用第 01 课，把一个 hidden 向量投影成另一个向量。
// 数学：      y=W@x；W[H,H]@x[H]->y[H]；y_o=dot(W[o],x)。
// 实现：      遍历 W 的每一行，每行调用一次 dot；本课所有 projection 无 bias。
Vector linear(const Matrix& weight, const Vector& input) {
    Vector output {};
    for (int row = 0; row < kHidden; ++row) output[row] = dot(weight[row], input);
    return output;
}

// 目的/直觉：复用第 01 课，在每个 mixer/FFN 分支前稳定 hidden 的整体大小。
// 数学：      y_i=x_i/sqrt(mean(x^2)+eps)；本课省略 learned (1+w_i)，等价于 w_i=0。
// 实现：      先求 H 个分量的均方与 inverse RMS，再用同一 scale 缩放所有维度。
Vector rmsnorm(const Vector& input) {
    float square = 0.0f;
    for (float value : input) square += value * value;
    const float scale = 1.0f / std::sqrt(square / kHidden + kEpsilon);
    return {input[0] * scale, input[1] * scale};
}

// 目的/直觉：复用第 02 课，为 SwiGLU 提供 0..1 的平滑软开关。
// 数学：      sigmoid(x)=1/(1+exp(-x))。
// 实现：      按 x 符号选择等价公式，避免指数溢出。
float sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + std::exp(-x));
    const float exp_x = std::exp(x);
    return exp_x / (1.0f + exp_x);
}

// 目的/直觉：复用第 02 课的平滑 gate 激活。
// 数学：      SiLU(x)=x*sigmoid(x)。
// 实现：      调用 sigmoid 后乘回 x。
float silu(float x) { return x * sigmoid(x); }

// 目的/直觉：复用第 03 课，让 attention 的 Q/K 方向携带当前 token position。
// 数学：      [x0,x1] -> [x0*cos(p)-x1*sin(p), x1*cos(p)+x0*sin(p)]。
// 实现：      先保存原值，再按二维旋转公式原地更新；V 不调用此函数。
void rope(Vector* vector, int position) {
    const float cosine = std::cos(static_cast<float>(position));
    const float sine = std::sin(static_cast<float>(position));
    const float x0 = (*vector)[0], x1 = (*vector)[1];
    (*vector)[0] = x0 * cosine - x1 * sine;
    (*vector)[1] = x1 * cosine + x0 * sine;
}

// 目的/直觉：复用第 06 课，只保留 Q/K 的方向，不让长度任意改变 state 读写强度。
// 数学：      y=x/sqrt(dot(x,x)+eps)，所以 ||y||_2 约等于 1。
// 实现：      计算一次 scale，再原地缩放两个维度。
void l2(Vector* vector) {
    const float scale = 1.0f / std::sqrt(dot(*vector, *vector) + kEpsilon);
    for (float& value : *vector) value *= scale;
}

struct DeltaState {
    Matrix recurrent {};  // 一个 Delta layer 的固定 S[H,H]，不会随 token 数增长。
};

struct AttentionState {
    std::vector<Vector> keys;    // 逻辑 shape [tokens,H]，每个 token append 一个 K。
    std::vector<Vector> values;  // 逻辑 shape [tokens,H]，与 keys 同步增长。
};

struct State {
    std::array<DeltaState, kDeltaLayers> delta;
    AttentionState attention;
    int position = 0;
};

// 目的/直觉：复用第 06/07 课，让一个 Delta layer 用固定矩阵 S 压缩并读取历史。
// 数学：      q=norm(Wq@x)，k=norm(Wk@x)，v=Wv@x；S=0.5S；
//             memory=k^T@S；S+=k outer 0.5(v-memory)；out=q^T@S。
// 实现：      identity projection 产生 q/k/v；随后严格按 decay、read、outer write、
//             query read 的顺序更新该 layer 自己的 recurrent matrix。
Vector delta_mixer(const Vector& input, DeltaState* state) {
    Vector query = linear(kIdentity, input);
    Vector key = linear(kIdentity, input);
    const Vector value = linear(kIdentity, input);
    l2(&query);
    l2(&key);

    for (Vector& row : state->recurrent) {
        for (float& cell : row) cell *= 0.5f;
    }

    Vector memory {};
    for (int value_dimension = 0; value_dimension < kHidden; ++value_dimension) {
        for (int key_dimension = 0; key_dimension < kHidden; ++key_dimension) {
            memory[value_dimension] +=
                key[key_dimension] * state->recurrent[key_dimension][value_dimension];
        }
    }
    for (int key_dimension = 0; key_dimension < kHidden; ++key_dimension) {
        for (int value_dimension = 0; value_dimension < kHidden; ++value_dimension) {
            state->recurrent[key_dimension][value_dimension] +=
                key[key_dimension] * 0.5f * (value[value_dimension] - memory[value_dimension]);
        }
    }

    Vector output {};
    for (int value_dimension = 0; value_dimension < kHidden; ++value_dimension) {
        for (int key_dimension = 0; key_dimension < kHidden; ++key_dimension) {
            output[value_dimension] +=
                query[key_dimension] * state->recurrent[key_dimension][value_dimension];
        }
    }
    return output;
}

// 目的/直觉：复用第 03～05 课，把当前 K/V append 到 cache，再让当前 Q 对全部历史
//             做 scaled dot-product softmax attention。
// 数学：      q=RoPE(Wq@x,p)，k=RoPE(Wk@x,p)，v=Wv@x；cache.append(k,v)；
//             score_t=dot(q,K_t)/sqrt(H)，prob=softmax(score)，out=sum_t(prob_t*V_t)。
// 实现：      identity projection 后旋转 Q/K；先 append 以允许看自己；再分三遍计算
//             score/max、stable softmax 分母、value 加权和。
Vector attention_mixer(const Vector& input, int position, AttentionState* state) {
    Vector query = linear(kIdentity, input);
    Vector key = linear(kIdentity, input);
    const Vector value = linear(kIdentity, input);
    rope(&query, position);
    rope(&key, position);
    state->keys.push_back(key);
    state->values.push_back(value);

    const float scale = 1.0f / std::sqrt(static_cast<float>(kHidden));
    std::vector<float> scores(state->keys.size());
    float maximum = -std::numeric_limits<float>::infinity();
    for (size_t token = 0; token < state->keys.size(); ++token) {
        scores[token] = dot(query, state->keys[token]) * scale;
        maximum = std::max(maximum, scores[token]);
    }
    float denominator = 0.0f;
    for (float& score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
    }

    Vector output {};
    for (size_t token = 0; token < scores.size(); ++token) {
        const float probability = scores[token] / denominator;
        for (int i = 0; i < kHidden; ++i) output[i] += probability * state->values[token][i];
    }
    return output;
}

// 目的/直觉：复用第 02 课，让每层 mixer 后都运行同一种真正的 SwiGLU FFN。
// 数学：      gate=Wg@x，up=Wu@x，mixed_i=SiLU(gate_i)*up_i，out=Wd@mixed。
// 实现：      三个 projection 使用 identity，逐维计算 SwiGLU，再执行 down projection。
Vector ffn(const Vector& input) {
    Vector gate = linear(kIdentity, input);
    const Vector up = linear(kIdentity, input);
    for (int i = 0; i < kHidden; ++i) gate[i] = silu(gate[i]) * up[i];
    return linear(kIdentity, gate);
}

// 目的/直觉：复用第 02 课，不覆盖主干 hidden，只把 mixer/FFN 的修正量加回去。
// 数学：      hidden'_i=hidden_i+branch_i。
// 实现：      遍历 H 个维度原地逐元素相加。
void residual_add(Vector* hidden, const Vector& branch) {
    for (int i = 0; i < kHidden; ++i) (*hidden)[i] += branch[i];
}

// 目的/直觉：展示 Qwen hybrid 的唯一新知识：每层先按 3:1 选择 mixer，再执行共同 FFN，
//             并让同一 State 跨 token 存活。
// 数学：      layer 0,1,2 用 DeltaNet，layer 3 用 attention；每层都是
//             h=h+mixer(RMSNorm(h))；h=h+FFN(RMSNorm(h))。
// 实现：      遍历四层，根据 layer%4 分支到对应 state；完成全部层后 position 加一。
Vector forward_token(const Vector& token_embedding, State* state) {
    Vector hidden = token_embedding;
    for (int layer = 0; layer < kLayers; ++layer) {
        Vector normalized = rmsnorm(hidden);
        const Vector mixer = layer % 4 == 3
            ? attention_mixer(normalized, state->position, &state->attention)
            : delta_mixer(normalized, &state->delta[layer]);
        residual_add(&hidden, mixer);

        normalized = rmsnorm(hidden);
        residual_add(&hidden, ffn(normalized));
    }
    ++state->position;
    return hidden;
}

// 目的/直觉：证明两类 state 都跨 token 生效，而不是每次 forward 重新清零。
// 数学：      reused_state 的第二次输出应不同于 fresh_state 的第一次输出；cache T=2。
// 实现：      同一 embedding 连续跑两次，再与全新 State 的一次结果比较并检查两类 state。
void self_test() {
    const Vector embedding = {1.0f, 0.5f};
    State reused;
    const Vector first = forward_token(embedding, &reused);
    const Vector second = forward_token(embedding, &reused);
    State fresh;
    const Vector without_history = forward_token(embedding, &fresh);

    assert(close(first[0], without_history[0]) && close(first[1], without_history[1]));
    assert(!close(second[0], without_history[0]) || !close(second[1], without_history[1]));
    assert(reused.attention.keys.size() == 2 && reused.attention.values.size() == 2);
    assert(std::fabs(reused.delta[0].recurrent[0][0]) > 0.0f);
}

}  // namespace lesson08

// 目的/直觉：打印同一 token 两次穿过 hybrid stack 的结果及 attention cache 长度。
// 数学：      output_t=HybridForward(embedding,state_{t-1})，state_t 继续供下一 token 使用。
// 实现：      先跑 self_test，再复用一个 State 调用 forward_token 两次并打印。
int main() {
    lesson08::self_test();
    lesson08::State state;
    const lesson08::Vector embedding = {1.0f, 0.5f};
    const lesson08::Vector first = lesson08::forward_token(embedding, &state);
    const lesson08::Vector second = lesson08::forward_token(embedding, &state);
    std::printf("token 0 output: [%.6f, %.6f]\n", first[0], first[1]);
    std::printf("token 1 output: [%.6f, %.6f]\n", second[0], second[1]);
    std::printf("attention cache tokens: %zu\n", state.attention.keys.size());
}

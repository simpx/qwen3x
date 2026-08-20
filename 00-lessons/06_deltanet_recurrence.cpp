// 第 06 课：Gated DeltaNet 的 recurrent state。
//
// 阅读路线：
//   已经会：attention 用会随文本增长的 KV cache 记住历史。
//   本课只加：DeltaNet 用固定大小的矩阵 S 写入、遗忘并读回前文。
//   运行后看：无论已处理多少 token，S 始终只有 key_dim * value_dim 个数。
//   下一课：把这条 recurrence 放回真实 DeltaNet layer 的投影、卷积、门控和输出投影之间。
//
// attention 的 KV cache 会随着 token 数增长；DeltaNet 不保存全部历史，
// 而是为每个 head 保存一个固定大小矩阵 S[key_dim][value_dim]。
// 每个 token 的核心更新是：
//   S      = exp(log_decay) * S
//   memory = k^T * S
//   S      = S + k outer ( beta * (v - memory) )
//   out    = q^T * S
//
// 真正 Qwen 还会在这之前计算 q/k 的 L2 norm、beta 和 log_decay；这里先把
// 这四条 recurrence 公式单独讲清楚。
//
// 维度约定：k、q 长度是 key_dim，v、out 长度是 value_dim；S 是
// [key_dim, value_dim]。所以 k^T @ S 读出一个 value 向量，k outer delta
// 把一个 value 向量写进 S。本课把 batch、head 都固定为 1，真实模型在每个
// value head 上独立维护一份 S。
//
// 白话记忆：DeltaNet 也是“写记忆、读记忆”，但不把每个旧 token 排成越来越长的列表。
// S 是固定大小的压缩笔记本：key 决定往笔记本的哪个地址写，value 是写入的内容，query
// 决定从哪里读；decay 是遗忘速度，beta 是这次改写笔记本的力度。这些数都由模型在每个
// token 动态产生，不是训练时的优化器学习率。

#include <cassert>
#include <cmath>
#include <cstdio>

namespace lesson06 {

constexpr int kKeyDim = 2;
constexpr int kValueDim = 2;

// 目的/直觉：比较 recurrent state 的浮点结果时容许微小误差。
// 数学：      close(a,b)=|a-b|<1e-5。
// 实现：      对差值取绝对值，再与固定阈值比较。
bool close(float left, float right) { return std::fabs(left - right) < 1e-5f; }

struct RecurrentState {
    // 这就是一个 head 的 S。context 增长时它的大小永远不变。
    float matrix[kKeyDim][kValueDim] = {};
};

// 目的/直觉：用固定大小矩阵 S 压缩历史。每个 token 先遗忘旧信息，再比较“当前 key
//             已能读回什么”，只写入 value 的误差，最后用 query 读取更新后的记忆。
// 数学：      d=exp(log_decay)
//             S=d*S
//             memory=k^T@S                         [K]@[K,V] -> [V]
//             delta=beta*(v-memory)                [V]
//             S=S+k outer delta                    [K,V]
//             out=q^T@S                            [K]@[K,V] -> [V]。
// 实现：      严格按上述顺序执行四组循环；S 原地更新并跨 token 保留，临时 memory
//             每一步重新创建。先写后读使当前 token 能影响自己的 output。
void delta_step(const float* query, const float* key, const float* value,
                float log_decay, float beta, RecurrentState* state, float* output) {
    // 顺序不能交换：先 decay，再根据衰减后的 memory 计算 delta，最后从更新后的
    // state 读取。这就保证 token t 的输出能包含 token t 自己的写入。

    // 1. 忘记一部分旧历史。log_decay 通常为负，故 decay 在 (0, 1]。
    const float decay = std::exp(log_decay);
    for (int key_dimension = 0; key_dimension < kKeyDim; ++key_dimension) {
        for (int value_dimension = 0; value_dimension < kValueDim; ++value_dimension) {
            state->matrix[key_dimension][value_dimension] *= decay;
        }
    }

    // 2. 从衰减后的 state 读出当前 key 已经“记住”了什么：memory = k^T @ S。
    float memory[kValueDim] = {};
    for (int value_dimension = 0; value_dimension < kValueDim; ++value_dimension) {
        for (int key_dimension = 0; key_dimension < kKeyDim; ++key_dimension) {
            memory[value_dimension] += key[key_dimension] * state->matrix[key_dimension][value_dimension];
        }
    }

    // 3. 写入当前 value 与 memory 的差；beta 决定本次写入力度。
    //    若 S 已能用 k 读回 v，delta=0，不会重复写入同一条记忆。
    for (int key_dimension = 0; key_dimension < kKeyDim; ++key_dimension) {
        for (int value_dimension = 0; value_dimension < kValueDim; ++value_dimension) {
            state->matrix[key_dimension][value_dimension] +=
                key[key_dimension] * beta * (value[value_dimension] - memory[value_dimension]);
        }
    }

    // 4. 由当前 query 从更新后的 state 读取输出：out = q^T @ S。
    for (int value_dimension = 0; value_dimension < kValueDim; ++value_dimension) {
        output[value_dimension] = 0.0f;
        for (int key_dimension = 0; key_dimension < kKeyDim; ++key_dimension) {
            output[value_dimension] += query[key_dimension] * state->matrix[key_dimension][value_dimension];
        }
    }
}

// 目的/直觉：用相同 token 连续更新同一份 S，观察 state 大小不变但记忆逐步接近 value。
// 数学：      初始 S=0、decay=0.5、beta=0.5；第一次 out=0.5*v，第二次 out=0.625*v。
// 实现：      对同一 RecurrentState 调用 delta_step 两次，并断言两次可手算输出。
void self_test() {
    RecurrentState state;
    const float query[kKeyDim] = {1.0f, 0.0f};
    const float key[kKeyDim] = {1.0f, 0.0f};
    const float value[kValueDim] = {3.0f, -2.0f};
    float output[kValueDim] = {};

    // 初始 S=0、beta=0.5，所以第一次写入正好得到 0.5 * value。
    delta_step(query, key, value, std::log(0.5f), 0.5f, &state, output);
    assert(close(output[0], 1.5f));
    assert(close(output[1], -1.0f));

    // 第二次先衰减一半，再按 delta rule 纠正到更接近 value 的状态；它展示了
    // state 的大小不变，但内容会随每个 token 更新。
    delta_step(query, key, value, std::log(0.5f), 0.5f, &state, output);
    assert(close(output[0], 1.875f));
    assert(close(output[1], -1.25f));
}

}  // namespace lesson06

// 目的/直觉：展示一次 Delta rule 后的输出和实际保存下来的固定矩阵 S。
// 数学：      q=k=[1,0]，所以本例只写入并读取 S 的第 0 行。
// 实现：      运行 self_test，再做一步 delta_step 并打印 output 与 state row 0。
int main() {
    lesson06::self_test();

    lesson06::RecurrentState state;
    const float query[lesson06::kKeyDim] = {1.0f, 0.0f};
    const float key[lesson06::kKeyDim] = {1.0f, 0.0f};
    const float value[lesson06::kValueDim] = {3.0f, -2.0f};
    float output[lesson06::kValueDim] = {};
    lesson06::delta_step(query, key, value, std::log(0.5f), 0.5f, &state, output);

    std::printf("DeltaNet output: [%.6f, %.6f]\n", output[0], output[1]);
    std::printf("state row 0:    [%.6f, %.6f]\n", state.matrix[0][0], state.matrix[0][1]);
}

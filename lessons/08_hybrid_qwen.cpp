// 第 08 课：Qwen hybrid stack 的控制流。
//
// 阅读路线：
//   已经会：attention 和 DeltaNet 都能读取前文，只是分别保存 KV cache 与固定 recurrent state。
//   本课只加：Qwen 每四层选择 3 个 DeltaNet + 1 个 attention；两类 state 都属于 generation State。
//   运行后看：同样输入的第二 token 会因复用两类旧 state 而得到不同输出。
//   下一课：拿掉玩具数字，按同样的控制流运行真实 Qwen3.5-0.8B 权重。
//
// lessons 04-07 已分别给出 attention 与 DeltaNet 的数学。本课只把焦点放在
// “一次 token 如何穿过四层”：三个 DeltaNet layer 的 state 不断更新，第四层
// 的 attention 则追加 KV cache。真正权重和完整张量维度在 capstone 中出现。
// 这里的 mixer/ffn 都是故意简化的标量函数；本课的唯一新知识是“每层选择哪类
// mixer，以及两类 state 为什么必须都放在 generation State 中”。
//
// 白话记忆：一层 Qwen 不会同时跑 attention 和 DeltaNet；它选择其中一个 mixer，然后
// 再跑同样的 FFN。当前 token 穿过多层时，DeltaNet 的固定大小 state 和 attention 的可增长
// KV cache 都必须随它一起保存到下一个 token。丢掉 State 就等于模型每个字都失忆。

#include <cassert>
#include <cstdio>
#include <vector>

namespace lesson08 {

constexpr int Layers = 4;

struct State {
    float delta_memory[3] = {};  // 三个 DeltaNet layer 的固定大小 memory。
    // 第四层才有、随 context 增长的 KV cache 简化视图；真实 cache 同时保存 K 和 V。
    std::vector<float> attention_values;
};

float delta_mixer(float input, float* memory) {
    // 这是 q=k=1 时 delta recurrence 的一维特例：固定 state，而不是 token 列表。
    *memory = 0.5f * *memory + 0.5f * input;
    return *memory;
}

float attention_mixer(float input, std::vector<float>* cache) {
    // 这里用均值代替 softmax，目的是只观察 cache 生命周期；完整 attention 在第 04/05 课。
    cache->push_back(input);
    float sum = 0.0f;
    for (float value : *cache) sum += value;
    return sum / cache->size();
}

float ffn(float x) { return 0.1f * x; }  // 本课把已学的 SwiGLU FFN 缩成可读占位。

float forward_token(float token_embedding, State* state) {
    // generation 时每个 token 都调用一次本函数，并把同一 State 继续传进来。
    float hidden = token_embedding;
    for (int layer = 0; layer < Layers; ++layer) {
        // 每个分支都是 pre-norm：真实路径此处会把 hidden 经过 RMSNorm。
        const float normalized = hidden;
        float mixer = 0.0f;
        if (layer % 4 != 3) mixer = delta_mixer(normalized, &state->delta_memory[layer]);
        else mixer = attention_mixer(normalized, &state->attention_values);
        hidden += mixer;          // 第一个 residual：hidden = hidden + mixer(normalized)。
        // 真正路径会再次 RMSNorm；这里保留 FFN 位于 mixer residual 之后的顺序。
        hidden += ffn(hidden);    // 第二个 residual。
    }
    return hidden;
}

void self_test() {
    State state;
    const float first = forward_token(1.0f, &state);
    const float second = forward_token(1.0f, &state);
    // 第二 token 使用了第一 token 留下的两类 state，因此轨迹应与第一 token 不同。
    assert(first > 1.0f && second > first);
    assert(state.attention_values.size() == 2);
}

}  // namespace lesson08

int main() {
    // 两次调用明确模拟 prefill/decoding 时“state 跨 token 存活”的事实。
    lesson08::self_test();
    lesson08::State state;
    std::printf("token 0 output: %.6f\n", lesson08::forward_token(1.0f, &state));
    std::printf("token 1 output: %.6f\n", lesson08::forward_token(1.0f, &state));
    std::printf("attention cache tokens: %zu\n", state.attention_values.size());
}

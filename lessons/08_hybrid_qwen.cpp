// 第 08 课：Qwen hybrid stack 的控制流。
//
// lessons 04-07 已分别给出 attention 与 DeltaNet 的数学。本课只把焦点放在
// “一次 token 如何穿过四层”：三个 DeltaNet layer 的 state 不断更新，第四层
// 的 attention 则追加 KV cache。真正权重和完整张量维度在 capstone 中出现。

#include <cassert>
#include <cstdio>
#include <vector>

namespace lesson08 {

constexpr int Layers = 4;

struct State {
    float delta_memory[3] = {};  // 三个 DeltaNet layer 的固定大小 memory。
    std::vector<float> attention_values;  // 第四层才有、随 context 增长的 KV cache 简化视图。
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
    float hidden = token_embedding;
    for (int layer = 0; layer < Layers; ++layer) {
        const float normalized = hidden;  // 省略数值细节；真实路径此处是 RMSNorm。
        float mixer = 0.0f;
        if (layer % 4 != 3) mixer = delta_mixer(normalized, &state->delta_memory[layer]);
        else mixer = attention_mixer(normalized, &state->attention_values);
        hidden += mixer;          // 第一个 residual。
        hidden += ffn(hidden);    // post-norm + FFN + 第二个 residual 的结构位置。
    }
    return hidden;
}

void self_test() {
    State state;
    const float first = forward_token(1.0f, &state);
    const float second = forward_token(1.0f, &state);
    assert(first > 1.0f && second > first);
    assert(state.attention_values.size() == 2);
}

}  // namespace lesson08

int main() {
    lesson08::self_test();
    lesson08::State state;
    std::printf("token 0 output: %.6f\n", lesson08::forward_token(1.0f, &state));
    std::printf("token 1 output: %.6f\n", lesson08::forward_token(1.0f, &state));
    std::printf("attention cache tokens: %zu\n", state.attention_values.size());
}

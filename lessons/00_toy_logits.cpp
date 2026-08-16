// 第 00 课：最小 language model 的一整步。
//
// 真正模型的第一个和最后一个操作也是这里的两件事：
//   token id -> embedding 向量 -> lm_head -> vocabulary logits -> 选一个 token。
// 为了只学习这一条数据流，本课没有 layer、attention 或 tokenizer。

#include <cassert>
#include <cmath>
#include <cstdio>

namespace lesson00 {

constexpr int kVocabSize = 4;  // 玩具词表中只有 4 个 token。
constexpr int kHiddenSize = 3; // 每个 token 用 3 个 float 表示。

// 行号就是 token id，列是 hidden dimension。
// 真正 Qwen 的 embedding 有 248320 行、1024 或 5120 列；数学完全一样。
const float kEmbedding[kVocabSize][kHiddenSize] = {
    {1.0f, 0.0f, 0.0f},  // token 0
    {0.0f, 1.0f, 0.0f},  // token 1
    {0.0f, 0.0f, 1.0f},  // token 2
    {0.5f, 0.5f, 0.5f},  // token 3
};

void embedding_lookup(int token, float* hidden) {
    assert(token >= 0 && token < kVocabSize);
    for (int i = 0; i < kHiddenSize; ++i) hidden[i] = kEmbedding[token][i];
}

// 本课模拟 tied word embeddings：lm_head 的每一行复用 embedding 的同一行。
// 因而 logits[v] = dot(hidden, embedding[v])。
void tied_lm_head(const float* hidden, float* logits) {
    for (int vocabulary_id = 0; vocabulary_id < kVocabSize; ++vocabulary_id) {
        float sum = 0.0f;
        for (int i = 0; i < kHiddenSize; ++i) sum += hidden[i] * kEmbedding[vocabulary_id][i];
        logits[vocabulary_id] = sum;
    }
}

int argmax(const float* values, int count) {
    int best = 0;
    for (int i = 1; i < count; ++i) {
        if (values[i] > values[best]) best = i;
    }
    return best;
}

void self_test() {
    float hidden[kHiddenSize] = {};
    float logits[kVocabSize] = {};
    embedding_lookup(2, hidden);
    tied_lm_head(hidden, logits);

    // token 2 的 embedding 是 [0, 0, 1]；与第 2 行点积为 1，是最大值。
    assert(std::fabs(logits[0] - 0.0f) < 1e-6f);
    assert(std::fabs(logits[2] - 1.0f) < 1e-6f);
    assert(argmax(logits, kVocabSize) == 2);
}

}  // namespace lesson00

int main() {
    lesson00::self_test();

    float hidden[lesson00::kHiddenSize] = {};
    float logits[lesson00::kVocabSize] = {};
    const int input_token = 2;
    lesson00::embedding_lookup(input_token, hidden);
    lesson00::tied_lm_head(hidden, logits);

    std::printf("input token: %d\nembedding:", input_token);
    for (float value : hidden) std::printf(" %.1f", value);
    std::printf("\nlogits:");
    for (float value : logits) std::printf(" %.1f", value);
    std::printf("\nnext token (argmax): %d\n", lesson00::argmax(logits, lesson00::kVocabSize));
}

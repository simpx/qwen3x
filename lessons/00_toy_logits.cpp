// 第 00 课：最小 language model 的一整步。
//
// 真正模型的第一个和最后一个操作也是这里的两件事：
//   token id -> embedding 向量 -> lm_head -> vocabulary logits -> 选一个 token。
// 为了只学习这一条数据流，本课没有 layer、attention 或 tokenizer。
//
// 阅读提示：tokenizer 属于本文件外的文字<->整数转换器；进入模型之后，模型只
// 看 token id。logit 也还不是概率：它只是词表中每个候选 token 的未归一化分数。
// 下一课会从这里得到的 hidden 向量开始，加入第一层 RMSNorm 和矩阵乘法。

#include <cassert>
#include <cmath>
#include <cstdio>

namespace lesson00 {

constexpr int kVocabSize = 4;  // 玩具词表中只有 4 个 token，即 4 个候选输出。
constexpr int kHiddenSize = 3; // hidden vector 的宽度；真实模型会大得多。

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
    // 这不是矩阵乘法：one-hot(token) @ embedding 的结果恰好就是取第 token 行。
    for (int i = 0; i < kHiddenSize; ++i) hidden[i] = kEmbedding[token][i];
}

// 本课模拟 tied word embeddings：lm_head 的每一行复用 embedding 的同一行。
// 因而 logits[v] = dot(hidden, embedding[v])。
void tied_lm_head(const float* hidden, float* logits) {
    // 输出 shape 是 [vocab]。这里没有 softmax，因为 argmax 前 softmax 不会改变
    // 最大值的位置；真正 sampling 才会在 logits 上做 temperature/softmax。
    for (int vocabulary_id = 0; vocabulary_id < kVocabSize; ++vocabulary_id) {
        float sum = 0.0f;
        for (int i = 0; i < kHiddenSize; ++i) sum += hidden[i] * kEmbedding[vocabulary_id][i];
        logits[vocabulary_id] = sum;
    }
}

int argmax(const float* values, int count) {
    // 贪婪解码：永远选最大 logit。它可重复、好测试，但不等同于随机 sampling。
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
    // 这个断言同时固定了 embedding lookup、tied lm_head 和 argmax 三个概念。
    assert(std::fabs(logits[0] - 0.0f) < 1e-6f);
    assert(std::fabs(logits[2] - 1.0f) < 1e-6f);
    assert(argmax(logits, kVocabSize) == 2);
}

}  // namespace lesson00

int main() {
    // main 故意重复 self_test 的小例子并打印中间值；博客读者可把它当作一次
    // "给定上一个 token，模型预测下一个 token" 的完整可见轨迹。
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

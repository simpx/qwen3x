// 第 00 课：最小 language model 的一整步。
//
// 阅读路线：
//   已经会：文字已在课程外由 tokenizer 变成一个 token id。
//   本课只加：id 查 embedding、tied lm_head 打分、argmax 选下一个 id。
//   运行后看：输入 id 2 如何得到 4 个 logits，最后为什么选 id 2。
//   下一课：从这里的 hidden 向量出发，先学习一层内部最常见的 RMSNorm 与 linear。
//
// 真正模型的第一个和最后一个操作也是这里的两件事：
//   token id -> embedding 向量 -> （许多 layers）-> lm_head -> vocabulary logits -> 选一个 token。
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

// 行号就是 token id，列是 hidden dimension。它可看成一个“token -> 小数向量”的查表字典：
// token 2 就取第 2 行 [0, 0, 1]。真正 Qwen 的表是 [248320, 1024]；数学完全一样。
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

// dot（点积）把两个同长度向量压成一个数：每一对对应位置相乘，再把结果相加。
// 它回答的不是“两个向量是否完全相同”，而是“它们的方向和大小有多匹配”。例如：
//   dot([1, 2], [3, 4]) = 1*3 + 2*4 = 11。
// 第一个参数只是数组开头的指针，本身不携带长度，所以和 argmax 一样需要 count
// 告诉循环何时停止。真正模型里的矩阵乘法，本质上也是很多次这样的点积。
float dot(const float* left, const float* right, int count) {
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) sum += left[i] * right[i];
    return sum;
}

// lm_head 是模型的最后一层：它把一个 hidden[hidden_size] 向量，变成 vocabulary 个分数。
// 换言之，它要回答“当前上下文之后，词表中每一个 token 分别有多合适？”
//
// 正常（untied）写法会另存一个很大的输出矩阵 W[vocab_size, hidden_size]，并计算
//   logits[v] = dot(hidden, W[v])。
// Qwen3.5-0.8B 采用 tied lm_head：不另存 W，而是直接重用输入 embedding 表作为 W。
// 因而同一张表既负责：
//   输入时：token id -> embedding 的第 token 行；
//   输出时：hidden 和 embedding 的每一行点积 -> 该 token 的 logit。
// 这会少存一份约 V*H 的权重，也让“适合作为下一个 token”的方向和 token embedding
// 共用同一套坐标系。它不是在做严格的 cosine similarity：向量长度也会影响分数。
//
// 公式是 logits[v] = dot(hidden, embedding[v])。v 是候选的下一个 token id。
void tied_lm_head(const float* hidden, float* logits) {
    // 输出 shape 是 [vocab]。这里没有 softmax，因为 argmax 前 softmax 不会改变
    // 最大值的位置；真正 sampling 才会在 logits 上做 temperature/softmax。
    for (int vocabulary_id = 0; vocabulary_id < kVocabSize; ++vocabulary_id) {
        // 固定一个候选 token，计算“当前 hidden”和该 token embedding 有多匹配。
        logits[vocabulary_id] = dot(hidden, kEmbedding[vocabulary_id], kHiddenSize);
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
    const float left[2] = {1.0f, 2.0f};
    const float right[2] = {3.0f, 4.0f};
    assert(std::fabs(dot(left, right, 2) - 11.0f) < 1e-6f);

    float hidden[kHiddenSize] = {};
    float logits[kVocabSize] = {};
    embedding_lookup(2, hidden);
    tied_lm_head(hidden, logits);

    // token 2 的 embedding 是 [0, 0, 1]；它和第 2 行点积为 1，是最大值。
    // 所以这个特意构造的玩具模型会选 token 2。真实模型并不是简单复制输入 token：
    // 真实 hidden 已经由所有 Transformer/DeltaNet layer 根据整个前文改写过了。
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

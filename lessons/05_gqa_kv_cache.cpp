// 第 05 课：GQA 与 KV cache。
//
// 在 decode 阶段，旧 token 的 K/V 不应每次重算。每层保存：
//   key_cache[position][kv_head][head_dim]
//   value_cache[position][kv_head][head_dim]
// 新 token 只投影一次 K/V 并 append；它的每个 Q head 从整个历史 cache 读取。
// GQA 的关键是多个 Q head 共用一个 KV head，从而缩小 cache。
//
// 这里 query_heads=2、kv_heads=1，所以每个 KV head 服务 2 个 Q head。真实 Qwen
// 的映射也一样简单：q_head / (query_heads / kv_heads) 就是它应读取的 KV head。
// 注意 cache 保存的是投影并 RoPE 后的 K、以及未旋转的 V，不是原始 hidden。
//
// 白话记忆：prompt 的 prefill 是把已有文本逐 token 写进 cache；随后每生成一个 token，
// decode 只为新 token 算一次 Q/K/V，再读取旧 cache。没有 cache 时，每次生成一个字都要
// 从头重新计算整段 prompt。GQA 的“grouped”只表示多个不同的 Q 共用较少的 K/V 存储；
// 它们仍可以因为 Q 不同而关注不同历史位置。

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

namespace lesson05 {

constexpr int kQueryHeads = 2;
constexpr int kKvHeads = 1;
constexpr int kHeadDim = 2;
constexpr int kQueriesPerKv = kQueryHeads / kKvHeads;

bool close(float left, float right) { return std::fabs(left - right) < 1e-5f; }

struct KvCache {
    // 连续布局等价于 [token][kv_head][head_dim]。
    std::vector<float> keys;
    std::vector<float> values;

    int token_count() const {
        // keys 和 values 总是同步 append，因此只需从 keys 的总元素数还原 token 数。
        return static_cast<int>(keys.size() / (kKvHeads * kHeadDim));
    }

    void append(const float new_keys[kKvHeads][kHeadDim], const float new_values[kKvHeads][kHeadDim]) {
        // append 的时机是“当前 token 的 K/V 已经投影好，当前 token 的 Q 要开始
        // attention”之前；因此当前 token 可以看到自己，也只能看到自己之前的位置。
        for (int kv_head = 0; kv_head < kKvHeads; ++kv_head) {
            for (int dimension = 0; dimension < kHeadDim; ++dimension) {
                keys.push_back(new_keys[kv_head][dimension]);
                values.push_back(new_values[kv_head][dimension]);
            }
        }
    }

    const float* key(int token, int kv_head) const {
        // 平铺数组 [token][kv_head][dimension] 的手写 offset，避免引入 Tensor 类。
        return &keys[(token * kKvHeads + kv_head) * kHeadDim];
    }

    const float* value(int token, int kv_head) const {
        return &values[(token * kKvHeads + kv_head) * kHeadDim];
    }
};

float dot(const float* left, const float* right) {
    float sum = 0.0f;
    for (int i = 0; i < kHeadDim; ++i) sum += left[i] * right[i];
    return sum;
}

void gqa_decode(const float query[kQueryHeads][kHeadDim], const KvCache& cache,
                float output[kQueryHeads][kHeadDim]) {
    assert(cache.token_count() > 0);
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));

    for (int q_head = 0; q_head < kQueryHeads; ++q_head) {
        // q_head 0、1 都映射到 kv_head 0；若有 4 个 Q / 2 个 KV，则映射为 0,0,1,1。
        const int kv_head = q_head / kQueriesPerKv;
        // score 的长度随 context 增长；这正是 attention decode 的 O(context) 工作量。
        std::vector<float> scores(cache.token_count());
        float maximum = -INFINITY;
        for (int token = 0; token < cache.token_count(); ++token) {
            scores[token] = dot(query[q_head], cache.key(token, kv_head)) * scale;
            if (scores[token] > maximum) maximum = scores[token];
        }

        float denominator = 0.0f;
        for (float& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }

        for (int dimension = 0; dimension < kHeadDim; ++dimension) output[q_head][dimension] = 0.0f;
        for (int token = 0; token < cache.token_count(); ++token) {
            const float probability = scores[token] / denominator;
            const float* value = cache.value(token, kv_head);
            for (int dimension = 0; dimension < kHeadDim; ++dimension) {
                output[q_head][dimension] += probability * value[dimension];
            }
        }
    }
}

void self_test() {
    KvCache cache;
    const float key0[kKvHeads][kHeadDim] = {{1.0f, 0.0f}};
    const float value0[kKvHeads][kHeadDim] = {{10.0f, 0.0f}};
    const float key1[kKvHeads][kHeadDim] = {{0.0f, 1.0f}};
    const float value1[kKvHeads][kHeadDim] = {{0.0f, 20.0f}};

    // 这两次 append 对应两 token prompt 的 prefill。真正 prefill 与 decode 的差别
    // 只在于循环次数：两者最终都向同一份 cache 写入每个位置的 K/V。
    cache.append(key0, value0);
    cache.append(key1, value1);
    assert(cache.token_count() == 2);

    // 两个 Q head 共享唯一的 KV head，但各自用不同 query 读取它。
    const float query[kQueryHeads][kHeadDim] = {{1.0f, 0.0f}, {0.0f, 1.0f}};
    float output[kQueryHeads][kHeadDim] = {};
    gqa_decode(query, cache, output);

    assert(output[0][0] > output[0][1]);
    assert(output[1][1] > output[1][0]);
    assert(close(cache.key(0, 0)[0], 1.0f));
}

}  // namespace lesson05

int main() {
    // 本例刻意让两个 Q head 偏好不同 token，证明“共享 KV”不代表输出相同。
    lesson05::self_test();

    lesson05::KvCache cache;
    const float key0[lesson05::kKvHeads][lesson05::kHeadDim] = {{1.0f, 0.0f}};
    const float value0[lesson05::kKvHeads][lesson05::kHeadDim] = {{10.0f, 0.0f}};
    const float key1[lesson05::kKvHeads][lesson05::kHeadDim] = {{0.0f, 1.0f}};
    const float value1[lesson05::kKvHeads][lesson05::kHeadDim] = {{0.0f, 20.0f}};
    const float query[lesson05::kQueryHeads][lesson05::kHeadDim] = {{1.0f, 0.0f}, {0.0f, 1.0f}};
    float output[lesson05::kQueryHeads][lesson05::kHeadDim] = {};
    cache.append(key0, value0);
    cache.append(key1, value1);
    lesson05::gqa_decode(query, cache, output);

    std::printf("cache tokens: %d\n", cache.token_count());
    std::printf("Q head 0: [%.6f, %.6f]\n", output[0][0], output[0][1]);
    std::printf("Q head 1: [%.6f, %.6f]\n", output[1][0], output[1][1]);
}

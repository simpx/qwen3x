// 第 05 课：GQA 与 KV cache。
//
// 在 decode 阶段，旧 token 的 K/V 不应每次重算。每层保存：
//   key_cache[position][kv_head][head_dim]
//   value_cache[position][kv_head][head_dim]
// 新 token 只投影一次 K/V 并 append；它的每个 Q head 从整个历史 cache 读取。
// GQA 的关键是多个 Q head 共用一个 KV head，从而缩小 cache。

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
        return static_cast<int>(keys.size() / (kKvHeads * kHeadDim));
    }

    void append(const float new_keys[kKvHeads][kHeadDim], const float new_values[kKvHeads][kHeadDim]) {
        for (int kv_head = 0; kv_head < kKvHeads; ++kv_head) {
            for (int dimension = 0; dimension < kHeadDim; ++dimension) {
                keys.push_back(new_keys[kv_head][dimension]);
                values.push_back(new_values[kv_head][dimension]);
            }
        }
    }

    const float* key(int token, int kv_head) const {
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
        const int kv_head = q_head / kQueriesPerKv;
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

    // 这两次 append 对应两 token prompt 的 prefill。
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

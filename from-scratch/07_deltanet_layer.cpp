// 第 07 课：把 DeltaNet 的零件组装成一个真正的 toy layer。
//
// 阅读路线：
//   已经会：DeltaNet 的 S 如何按 Q/K/V、decay 和 beta 逐 token 更新。
//   本课分两段：先手算 current_qkv[6]+history[6,2]->mixed_qkv[6]；
//                 再看同一 hidden 如何投影成 q/k/v/z/a/b，走完 DeltaNet 分支。
//   运行后看：同一个 State 连续处理三个 token，第三次的 conv 同时使用前两个 token。
//   下一课：把 DeltaNet layer 和 attention layer 按 Qwen 的固定顺序放进同一个模型。
//
// 执行顺序与 Qwen 相同：
//   in_proj_qkv / z / a / b -> depthwise causal conv -> q/k L2 norm
//   -> gated delta recurrence -> gated RMSNorm -> out_proj。
// 所有维度都缩到 2；因此本课能运行，但每一行仍对应真实模型中的同一位置。
//
// 这不是一个可泛化的 DeltaNet 实现：权重固定在函数中，conv 也特意统一使用
// current + 0.5*previous + 0.25*older。它的任务是把上一课的 recurrence
// 放回真实 layer 的前后投影、门控和
// normalization 之间，回答“一个 token 进入 DeltaNet layer 时具体经过什么”。
//
// 白话记忆：本课的 input 是“已经经过前面 layer 的当前 token hidden”。四个投影不是四份
// 不同 token：qkv 产生读/写记忆所需的 Q、K、V；z 控制最终输出门；a 控制遗忘；b 控制
// 写入力度。它们都只是从同一个 input 向量线性算出的不同视角。

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace lesson07 {

constexpr int H = 2, QKV = 6, D = 2, CONV_HISTORY = 2;
// H 是输入/输出 hidden size；QKV=Q(2)+K(2)+V(2)，D 是本例单 head 的宽度。
// CONV_HISTORY=2 表示每个 qkv 位置保存前两个 token 的值。

// 本课 conv 的全部计算（先看计算，再记术语）：
//
//   token t 当前投影：current_qkv[6] = [q0,q1,k0,k1,v0,v1]
//   前两个 token：      history[6,2]，每个位置都有 [older,previous]
//
//   对 position=0..5 分别做：
//   mixed_qkv[position]
//     = current_qkv[position]
//     + 0.5 * previous_qkv[position]
//     + 0.25 * older_qkv[position]
//
//   current_qkv[6] + history[6,2] -> mixed_qkv[6]
//
// 名字只是这个计算的简写：一组乘法权重叫 kernel；整个局部乘加叫 conv；
// 只用当前和过去、不用未来叫 causal。conv 不改变 qkv 的 shape。

// 目的/直觉：把门控参数压到 0..1，供 DeltaNet 的 beta 使用。
// 数学：      sigmoid(x)=1/(1+exp(-x))。
// 实现：      按 x 的符号选择等价公式，避免 exp 对大数溢出。
float sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + std::exp(-x));
    const float exp_x = std::exp(x);
    return exp_x / (1.0f + exp_x);
}

// 目的/直觉：复用第 02 课的平滑非线性，供 causal conv 和输出 gate 使用。
// 数学：      SiLU(x)=x*sigmoid(x)。
// 实现：      调用 sigmoid 后乘回 x。
float silu(float x) { return x * sigmoid(x); }

// 目的/直觉：把任意 a 参数变成正数，再取负号构造 log_decay<=0。
// 数学：      softplus(x)=log(1+exp(x)) > 0。
// 实现：      大正数直接返回 x 避免 exp 溢出，其余用 log1p 保留精度。
float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

// 目的/直觉：复用第 00 课的 dot，为 matrix-vector multiplication 算一个输出。
// 数学：      dot(a,b)=sum_{i=0}^{H-1}(a_i*b_i)。
// 实现：      遍历 H 个通道，逐项相乘并累加。
float dot(const float* left, const float* right) {
    float sum = 0.0f;
    for (int i = 0; i < H; ++i) sum += left[i] * right[i];
    return sum;
}

// 目的/直觉：复用第 01 课的 linear，让同一 input 投影出 q/k/v/z/a/b 等不同视角。
// 数学：      W[rows,H] @ x[H] -> y[rows]；y_r=dot(W[r],x)。
// 实现：      遍历权重的 rows 行，每行调用一次 dot；这些 toy projection 无 bias。
void mv(const float (*weight)[H], int rows, const float* x, float* y) {
    for (int row = 0; row < rows; ++row) y[row] = dot(weight[row], x);
}

// 目的/直觉：只保留 Q/K 的方向，避免其长度任意放大 DeltaNet 的读写强度。
// 数学：      y=x/sqrt(sum_i(x_i^2)+eps)，所以 ||y||_2 约等于 1。
// 实现：      先求两个分量平方和的倒数平方根，再原地缩放两个分量。
void l2(float* x) {
    const float scale = 1.0f / std::sqrt(x[0] * x[0] + x[1] * x[1] + 1e-6f);
    x[0] *= scale;
    x[1] *= scale;
}

struct State {
    // conv_history[QKV=6][2]：每个 qkv 位置保留前两个 token。
    //   [position][0] = older，也就是 t-2
    //   [position][1] = previous，也就是 t-1
    // recurrent[D=2][D=2]：就是第 06 课的 S[Dk,Dv]。
    // 即使生成无限长文本，这两个数组也不会随 token 数增长。
    float conv_history[QKV][CONV_HISTORY] = {};
    float recurrent[D][D] = {};
};

// 目的/直觉：让当前 qkv[6] 在进入 recurrence 前，混入前两个 token
//             在同一位置上的数值。
// 数学与 shape：
//   current[6] + history[6,2] -> mixed[6] -> SiLU -> x[6]
//   mixed_t[i] = current_t[i] + 0.5*previous[i] + 0.25*older[i]。
// 实现：对 6 个位置分别计算；算完后丢掉 t-2，把 [t-1,t]
//             作为下一步的两个历史。
void conv(float* x, State* state) {
    // 真实 Qwen3.5-0.8B 使用当前+前三个；本课先展示当前+前两个。
    for (int position = 0; position < QKV; ++position) {
        const float current = x[position];
        const float older = state->conv_history[position][0];
        const float previous = state->conv_history[position][1];
        const float mixed = current + 0.5f * previous + 0.25f * older;
        x[position] = silu(mixed);

        // [older, previous] = [t-2, t-1] 向前移动成 [t-1, t]。
        state->conv_history[position][0] = previous;
        state->conv_history[position][1] = current;
    }
}

// 目的/直觉：单独验证并打印第三个 token 如何同时使用前两个 token，
//             不让这件事被后面的 Delta rule 和门控掩盖。
// 数学与 shape：对每个位置，token2_mixed=token2+0.5*token1+0.25*token0。
// 实现：用同一个 State 连续调用 conv 三次，对第三次结果逐位断言。
void conv_self_test() {
    State state;
    float token0[QKV] = {1, 2, 3, 4, 5, 6};
    float token1[QKV] = {10, 20, 30, 40, 50, 60};
    float token2[QKV] = {100, 200, 300, 400, 500, 600};
    conv(token0, &state);
    conv(token1, &state);
    conv(token2, &state);

    for (int position = 0; position < QKV; ++position) {
        const float raw0 = static_cast<float>(position + 1);
        const float raw1 = 10.0f * raw0;
        const float raw2 = 100.0f * raw0;
        const float expected = silu(raw2 + 0.5f * raw1 + 0.25f * raw0);
        assert(std::fabs(token2[position] - expected) < 1e-5f);
        assert(std::fabs(state.conv_history[position][0] - raw1) < 1e-6f);
        assert(std::fabs(state.conv_history[position][1] - raw2) < 1e-6f);
    }

    const float position0_mixed = 100.0f + 0.5f * 10.0f + 0.25f * 1.0f;
    std::printf("conv shape: current[6] + history[6,2] -> mixed[6]\n");
    std::printf("token 2 position 0: 100 + 0.5*10 + 0.25*1 = %.2f; SiLU -> %.2f\n",
                position0_mixed, token2[0]);
}

// 目的/直觉：复用第 06 课，用一个 head 的固定矩阵 S 完成遗忘、误差写入和读取。
// 数学：      S=exp(log_decay)S；memory=k^T@S；S+=k outer beta(v-memory)；out=q^T@S。
// 实现：      按公式顺序执行 decay、memory、outer-product update、query read 四组循环。
void delta_rule(const float* q, const float* k, const float* v, float log_decay,
                float beta, State* state, float* out) {
    for (int key = 0; key < D; ++key) {
        for (int value = 0; value < D; ++value) state->recurrent[key][value] *= std::exp(log_decay);
    }
    float memory[D] = {};
    for (int value = 0; value < D; ++value) {
        for (int key = 0; key < D; ++key) memory[value] += k[key] * state->recurrent[key][value];
    }
    for (int key = 0; key < D; ++key) {
        for (int value = 0; value < D; ++value) {
            state->recurrent[key][value] += k[key] * beta * (v[value] - memory[value]);
        }
    }
    for (int value = 0; value < D; ++value) {
        out[value] = 0.0f;
        for (int key = 0; key < D; ++key) out[value] += q[key] * state->recurrent[key][value];
    }
}

// 目的/直觉：对 Delta readout 做每 head 的幅度稳定和 z 门控，再交给 out_proj。
// 数学：      y_i = read_i / sqrt(mean(read^2)+eps) * norm_i * SiLU(z_i)。
// 实现：      先求整个 head 共用的 inverse RMS，再逐维乘 norm weight 与 SiLU gate。
void gated_rms(const float* read, const float* norm_weight, const float* z, float* output) {
    float square = 0.0f;
    for (int i = 0; i < D; ++i) square += read[i] * read[i];
    const float scale = 1.0f / std::sqrt(square / D + 1e-6f);
    for (int i = 0; i < D; ++i) output[i] = read[i] * scale * norm_weight[i] * silu(z[i]);
}

// 目的/直觉：把同一个 hidden 完整送过 Qwen DeltaNet mixer：先产生 q/k/v 和三个门，
//             再做 causal conv、固定 state recurrence、gated RMSNorm 与 out projection。
// 数学：      qkv=Wqkv@x，z=Wz@x，a=Wa@x，b=Wb@x；
//             read=DeltaRule(norm(q),norm(k),v,-softplus(a),sigmoid(b),S)；
//             output=Wout@GatedRMS(read,z)。
// 实现：      所有 toy 权重写在函数内；按真实 forward 顺序调用 mv/conv/l2/delta_rule/
//             gated_rms/mv，state 中的 conv_history 和 recurrent 跨 token 保留。
void delta_layer(const float* input, State* state, float* output) {
    // 前三行是 Q，接着 K，最后两行 V。真实 Qwen 将同样的拼接先过 conv。
    const float qkv_weight[QKV][H] = {
        {1, 0}, {0, 1}, {0.5f, 0}, {0, 0.5f}, {1, 0}, {0, 1},
    };
    const float z_weight[D][H] = {{1, 0}, {0, 1}};
    const float scalar_weight[1][H] = {{0, 0}};
    const float norm_weight[D] = {1.0f, 1.0f};
    const float out_weight[H][H] = {{1, 0}, {0, 1}};
    // z 是末尾 gated RMSNorm 的门；a 产生 decay，b 经 sigmoid 产生 beta。
    float qkv[QKV] = {}, z[D] = {}, a[1] = {}, b[1] = {};
    mv(qkv_weight, QKV, input, qkv);
    mv(z_weight, D, input, z);
    mv(scalar_weight, 1, input, a);
    mv(scalar_weight, 1, input, b);
    conv(qkv, state);

    // conv 已把 qkv 原地替换为激活后的值；按固定布局拆成 Q、K、V 三段。
    float q[D] = {qkv[0], qkv[1]};
    float k[D] = {qkv[2], qkv[3]};
    l2(q);
    l2(k);
    // Q 额外除 sqrt(D)，等价于实现中对 query 的缩放约定。
    for (float& value : q) value /= std::sqrt(static_cast<float>(D));

    float read[D] = {};
    const float log_decay = -softplus(a[0]);  // A_log=0、dt_bias=0 的 toy 情况。
    delta_rule(q, k, qkv + 4, log_decay, sigmoid(b[0]), state, read);

    float gated[D] = {};
    gated_rms(read, norm_weight, z, gated);
    mv(out_weight, H, gated, output);
}

// 目的/直觉：证明第二个 token 的输出确实读取了第一个 token 留在 S 中的记忆。
// 数学：      DeltaLayer(x1, DeltaLayer(x0,S=0).state) 应不同于 DeltaLayer(x1,S=0)。
// 实现：      一条路径连续跑 x0,x1，另一条路径只用全新 state 跑 x1；比较第二步输出。
void self_test() {
    conv_self_test();
    State with_history, fresh;
    const float first_input[H] = {1.0f, 0.0f};
    const float second_input[H] = {1.0f, 1.0f};
    float first_output[H] = {}, with_history_output[H] = {}, fresh_output[H] = {};
    delta_layer(first_input, &with_history, first_output);
    delta_layer(second_input, &with_history, with_history_output);
    delta_layer(second_input, &fresh, fresh_output);

    assert(std::isfinite(first_output[0]) && std::isfinite(first_output[1]));
    assert(std::fabs(with_history.recurrent[0][0]) > 0.0f);
    assert(std::fabs(with_history_output[0] - fresh_output[0]) > 1e-5f ||
           std::fabs(with_history_output[1] - fresh_output[1]) > 1e-5f);
}

}  // namespace lesson07

// 目的/直觉：打印三个连续 token 的输出；第三步的 conv 同时混入前两步。
// 数学：      第三步 conv 使用 current + 0.5*previous + 0.25*older。
// 实现：      运行 self_test，再复用同一 State 依次处理三个 hidden 并打印。
int main() {
    lesson07::self_test();
    lesson07::State state;
    const float first_input[lesson07::H] = {1.0f, 0.0f};
    const float second_input[lesson07::H] = {1.0f, 1.0f};
    const float third_input[lesson07::H] = {0.0f, 1.0f};
    float first_output[lesson07::H] = {};
    float second_output[lesson07::H] = {};
    float third_output[lesson07::H] = {};
    lesson07::delta_layer(first_input, &state, first_output);
    lesson07::delta_layer(second_input, &state, second_output);
    lesson07::delta_layer(third_input, &state, third_output);
    std::printf("token 0 DeltaNet output: [%.6f, %.6f]\n", first_output[0], first_output[1]);
    std::printf("token 1 DeltaNet output: [%.6f, %.6f]\n", second_output[0], second_output[1]);
    std::printf("token 2 DeltaNet output: [%.6f, %.6f]\n", third_output[0], third_output[1]);
}
